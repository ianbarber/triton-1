#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCESCANCOMMON_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCESCANCOMMON_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

//
#include "mlir/IR/TypeUtilities.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <iterator>
#include <type_traits>

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
class ReduceOp;
class ScanOp;

// === NPOT pow2-rounding phantom register detection (shared reduce/scan) ===
//
// When the true contiguous-per-thread run on the reduction/scan axis is not a
// power of two (e.g. a [rows, 96] tensor at num_warps=1: lane stride 3 on the
// axis), the LinearLayout rounds the per-thread register dim up to the next
// power of two (register axis bases [1, 2] -> 4 register slots with axis
// offsets 0,1,2,3). The extra register slot is NOT out-of-range: with lane
// stride == the true contig, register i of lane L aliases a lower register of
// lane L+1 (offset 3 of lane L == offset 0 of lane L+1), so it DUPLICATES a
// real element. Folding all register rows within the thread therefore counts
// that duplicate in two lanes' partials -> it is double-counted after the
// cross-lane fold (an all-ones [rows, 96] reduce at num_warps=1 yields 128
// instead of 96; the out-of-range masker alone leaves 127 -- it catches only
// the single lane whose duplicate lands past dimSize).
//
// NOTE: the "true contig" here is the lane stride on the axis (the smallest
// positive lane-basis axis offset), NOT getContigPerThread() -- that walks the
// register bases and reports the pow2-rounded run (4 above), which hides the
// duplicate. Use getReductionAxisLaneStride() below.
//
// The out-of-range masker (offset >= dimSize) misses these because they are
// IN-range duplicates. This helper flags them instead: a register index is a
// pow2-rounding phantom when its within-tile axis position lies in
// [contigPerThread, NextPow2(contigPerThread)). Callers identity-fill those
// slots (for all lanes) before the within-thread fold.
//
// `offsets` must be emitOffsetForLayout(srcLayout, type): offsets[i][axis] is
// the axis offset of register i at lane=warp=0. `contigPerThread` is the lane
// stride on the axis. Returns the register indices (into the per-thread value
// vector) that are phantoms. Empty when contigPerThread is a power of two (or
// 0) -> pow2/N=48/64/192 stay byte-identical.
//
// TODO: ScanOpToLLVM (scan/cumsum) has the same pow2-rounding duplicate on its
// within-thread scan and should reuse this (with getReductionAxisLaneStride) to
// identity-fill the phantom registers before scanning.
inline llvm::SmallDenseSet<unsigned>
getPow2RoundingPhantomRegisters(ArrayRef<SmallVector<unsigned>> offsets,
                                unsigned axis, unsigned contigPerThread) {
  llvm::SmallDenseSet<unsigned> phantom;
  if (contigPerThread == 0)
    return phantom;
  unsigned pow2Contig = llvm::PowerOf2Ceil(contigPerThread);
  // Power-of-two contig: no rounding, so no phantom slots (byte-identical).
  if (pow2Contig == contigPerThread)
    return phantom;
  for (unsigned i = 0; i < offsets.size(); ++i) {
    unsigned withinTile = offsets[i][axis] % pow2Contig;
    if (withinTile >= contigPerThread)
      phantom.insert(i);
  }
  return phantom;
}

// The true contiguous-per-thread run on `axis`: the smallest positive lane
// basis axis offset (the stride to the neighbouring thread on the axis).
// Returns 0 when no lane basis moves along the axis (the axis lives entirely in
// registers/warps, so there is no register-vs-lane pow2-rounding duplicate).
// `ll` is the source LinearLayout, toLinearLayout(srcShape, srcEncoding).
inline unsigned getReductionAxisLaneStride(const LinearLayout &ll,
                                           unsigned axis) {
  unsigned stride = 0;
  for (const auto &[inDim, bases] : ll.getBases()) {
    if (inDim.str() != "lane")
      continue;
    for (const auto &basis : bases) {
      unsigned off = static_cast<unsigned>(basis[axis]);
      if (off > 0 && (stride == 0 || off < stride))
        stride = off;
    }
  }
  return stride;
}

inline SmallVector<Value>
inlineCombineBlock(ConversionPatternRewriter &rewriter, Block &combineBlock,
                   Block *insertionBlock, Block::iterator insertionPoint,
                   ValueRange combineArgs) {
  auto returnOp = combineBlock.getTerminator();
  rewriter.inlineBlockBefore(&combineBlock, insertionBlock, insertionPoint,
                             combineArgs);

  auto results = SmallVector<Value>(returnOp->getOperands());

  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
  return results;
}

inline SmallVector<Value> applyCombineOp(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Region &combineOp, ValueRange acc,
                                         ValueRange cur, Value pred = {}) {
  // Allows for passing an uninitialized acc and use cur as the neutral element
  if (acc.size() == 0) {
    return cur;
  }
  assert(cur.size() == acc.size());

  // Create a new copy of the combine block, and try to speculatively inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();

  rewriter.cloneRegionBefore(combineOp, parent,
                             std::next(currentBlock->getIterator()));
  Block &newCombine = *currentBlock->getNextNode();

  llvm::SmallVector<Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  auto isRegionSpeculatable =
      std::all_of(newCombine.begin(), newCombine.end(),
                  [](auto &op) { return isSpeculatable(&op); });

  if (!pred || isRegionSpeculatable) {
    // Fast path, region has no side effects so we can unconditionally execute
    return inlineCombineBlock(rewriter, newCombine, currentBlock,
                              rewriter.getInsertionPoint(), combineArgs);
  }

  // Slow case, create an if to only execute region when pred is true
  // #currentBlock
  // if (pred) {
  //   #newCombine
  //   results = combineOp(cur, acc)
  //   yield results
  // } else {
  //    yield undef
  // }
  // #thenBlock
  Block *thenBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  auto returnOp = newCombine.getTerminator();
  auto results = SmallVector<Value>(returnOp->getOperands());

  rewriter.setInsertionPointToEnd(currentBlock);
  SmallVector<Value> thenBlockArgs;
  thenBlockArgs.reserve(results.size());
  for (auto result : results) {
    auto ty = result.getType();
    auto undef = LLVM::UndefOp::create(rewriter, loc, ty);
    thenBlockArgs.push_back(undef);
    thenBlock->addArgument(ty, loc);
  }
  LLVM::CondBrOp::create(rewriter, loc, pred, &newCombine, combineArgs,
                         thenBlock, thenBlockArgs);

  // Split a block after the call.
  rewriter.setInsertionPointToEnd(&newCombine);
  rewriter.replaceOpWithNewOp<LLVM::BrOp>(returnOp, results, thenBlock);
  rewriter.setInsertionPointToStart(thenBlock);
  return SmallVector<Value>(thenBlock->getArguments());
}

} // namespace mlir::triton

template <typename SourceOp>
class ConvertTritonGPUReduceScanToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp> {
public:
  // Make sure the class is only instantiated with Reduce and Scan
  static_assert(std::is_same_v<SourceOp, ReduceOp> ||
                std::is_same_v<SourceOp, ScanOp>);

  using ConvertOpToLLVMPattern<SourceOp>::getTypeConverter;
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(SourceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  SmallVector<Value> getSmemBases(SourceOp op, unsigned elems,
                                  ConversionPatternRewriter &rewriter,
                                  const TargetInfoBase &targetInfo) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    auto basePtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    indexToBase[indices[0]] = basePtr;
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] =
          b.gep(basePtr.getType(), getElementType(op, indices[i - 1]),
                indexToBase[indices[i - 1]], b.i32_val(elems));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }
};

#endif
