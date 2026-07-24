// RUN: TRITON_ALLOW_NPOT=1 triton-opt --split-input-file %s -verify-diagnostics

// The pre-Ampere dot path falls back to FMA lowering, which does not support
// modular dot-operand layouts. Reject the NPOT type instead of leaving a silent
// miscompile route open.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sm75_dot_operand(%arg0: tensor<16x48xf16, #dot0>) {
    // expected-error @+1 {{NPOT dot/MMA layout is not yet supported for NVIDIA compute capability 75}}
    %0 = ttg.convert_layout %arg0 : tensor<16x48xf16, #dot0> -> tensor<16x48xf16, #blocked>
    tt.return
  }
}

// -----

// An explicitly constructed legacy NVIDIA MMA layout is rejected as well.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sm75_mma(%arg0: tensor<16x48xf16, #mma>) {
    // expected-error @+1 {{NPOT dot/MMA layout is not yet supported for NVIDIA compute capability 75}}
    %0 = ttg.convert_layout %arg0 : tensor<16x48xf16, #mma> -> tensor<16x48xf16, #mma>
    tt.return
  }
}

// -----

// sm80+ remains admitted.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sm80_dot_operand(%arg0: tensor<16x48xf16, #dot0>) {
    %0 = ttg.convert_layout %arg0 : tensor<16x48xf16, #dot0> -> tensor<16x48xf16, #blocked>
    tt.return
  }
}

// -----

// The gate is specific to dot/MMA layouts; blocked elementwise NPOT remains
// available at sm75.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sm75_blocked(%arg0: tensor<16x48xf16, #blocked>) {
    %0 = arith.addf %arg0, %arg0 : tensor<16x48xf16, #blocked>
    tt.return
  }
}
