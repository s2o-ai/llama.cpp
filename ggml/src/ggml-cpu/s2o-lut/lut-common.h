// S2O LUT-based INT4 inference backend — shared types and utilities
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

// ============================================================================
// LUT Concept
// ============================================================================
//
// For INT4 quantization, each weight nibble is in [0,15]. Given a quantized
// block with scale `d`, the dequantized value of nibble `q` is:
//
//     w = d * (q - 8)                   (for Q4_0)
//
// The dot product of a group of weights with FP32 activations can be rewritten:
//
//     sum_i  w_i * x_i  =  sum_i  d * (q_i - 8) * x_i
//                        =  d * [ sum_i  LUT[q_i]  -  8 * sum_i x_i ]
//
// where LUT[q] = q * x_i is precomputed for each activation value.
//
// For a group of 32 elements (one Q4_0 block), we build a 16-entry LUT
// (one entry per possible nibble value 0-15), where each entry accumulates
// the partial dot product for that nibble value across a sub-group of
// activations processed in a SIMD lane.
//
// With VPSHUFB (AVX2/AVX-512), we can perform 32/64 parallel LUT lookups
// in a single instruction, making this extremely fast.
//
// ============================================================================

// Q4_0 block: 32 values, stored as 16 bytes of nibble pairs + fp16 scale
// QK4_0 = 32 (defined in ggml)
#ifndef QK4_0
#define QK4_0 32
#endif

// Q4_K super-block: 256 values (8 sub-blocks of 32)
#ifndef QK_K
#define QK_K 256
#endif

// Default L2 tile size in bytes (256 KB). Adjustable at runtime.
#define S2O_LUT_DEFAULT_L2_TILE_BYTES (256 * 1024)

// Alignment for repacked weight buffers
#define S2O_LUT_ALIGN 64

// ============================================================================
// Quantized types supported by S2O LUT kernels
// ============================================================================

inline bool s2o_lut_supports_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0;
    // Q4_K support will be added in a follow-up
}

// ============================================================================
// LUT kernel function signatures
// ============================================================================

// GEMV: single-row vector-matrix multiply
//   dst[j] = dot(activation[0..K-1], weights[j][0..K-1])  for j in [j_start, j_end)
//
// Parameters:
//   dst       - output float array, length >= j_end
//   src_act   - activation vector, float[K]
//   src_wt    - quantized weight matrix (Q4_0 blocks, row-major, K elements per row)
//   K         - inner dimension (number of elements per dot product)
//   j_start   - first output column
//   j_end     - one past last output column
//   nb        - stride in bytes between consecutive weight rows (columns of the weight tensor in ggml layout)
typedef void (*s2o_lut_gemv_fn)(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb
);

// GEMM: batched matrix multiply (for prompt processing)
//   dst[i][j] = dot(activation[i][0..K-1], weights[j][0..K-1])
//
// Parameters:
//   dst       - output float matrix, row-major, stride dst_stride floats
//   src_act   - activation matrix, float[M][K], row-major, stride act_stride floats
//   src_wt    - quantized weight matrix
//   M         - number of activation rows (batch/prompt tokens)
//   K         - inner dimension
//   j_start   - first output column
//   j_end     - one past last output column
//   nb        - stride in bytes between consecutive weight rows
//   dst_stride - stride of dst in floats
//   act_stride - stride of activation in floats
typedef void (*s2o_lut_gemm_fn)(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       M,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb,
    int64_t       dst_stride,
    int64_t       act_stride
);

// ============================================================================
// Kernel dispatch table
// ============================================================================

struct s2o_lut_kernels {
    const char     * name;       // e.g. "avx512", "avx2"
    s2o_lut_gemv_fn  gemv_q4_0;  // single-row Q4_0
    s2o_lut_gemm_fn  gemm_q4_0;  // batched Q4_0
};

// Implemented in architecture-specific files:
#if defined(__AVX512F__) && defined(__AVX512BW__)
extern const s2o_lut_kernels s2o_lut_kernels_avx512;
#endif

#if defined(__AVX2__)
extern const s2o_lut_kernels s2o_lut_kernels_avx2;
#endif

#if defined(__ARM_NEON)
extern const s2o_lut_kernels s2o_lut_kernels_neon;
extern const s2o_lut_kernels s2o_lut_kernels_neon_dotprod;
#endif

// Select best available kernel set for current compilation target
inline const s2o_lut_kernels * s2o_lut_select_kernels(void) {
#if defined(__ARM_NEON)
    // On ARM, DOTPROD variant is always defined (stubs to baseline if not compiled with DOTPROD).
    // Runtime HWCAP check would go here for dynamic selection — for now, use compile-time.
#if defined(__ARM_FEATURE_DOTPROD)
    return &s2o_lut_kernels_neon_dotprod;
#else
    return &s2o_lut_kernels_neon;
#endif
#elif defined(__AVX512F__) && defined(__AVX512BW__)
    return &s2o_lut_kernels_avx512;
#elif defined(__AVX2__)
    return &s2o_lut_kernels_avx2;
#else
    return nullptr;
#endif
}
