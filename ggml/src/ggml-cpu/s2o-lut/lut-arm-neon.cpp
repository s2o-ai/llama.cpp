// S2O LUT-based INT4 GEMV/GEMM — ARM NEON implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ARM NEON integer dot product path (same strategy as x86 AVX2):
//   1. Quantize FP32 activations to INT8 on the fly (32 values per block)
//   2. For each output column:
//      a. Unpack Q4_0 nibbles: low nibble → elems 0-15, high nibble → elems 16-31
//      b. Subtract 8 to center in [-8, +7]
//      c. INT8 × INT8 dot product (SDOT if available, else SMULL+SADALP)
//      d. Scale by (d_weight * d_activation), accumulate
//
// Two variants compiled in the same file:
//   - Baseline NEON: vmull_s8 + vpaddlq_s16 (all ARMv8-A)
//   - DOTPROD:       vdotq_s32 (ARMv8.2-A+dotprod, ~4x fewer instructions)
//
// The dispatch table selector in lut-common.h picks at runtime based on HWCAP.

#if defined(__ARM_NEON)

#include "lut-common.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// Nibble unpacking: 16 bytes → 32 INT8 values in [-8, +7]
// ============================================================================
// ggml Q4_0 layout: qs[j] low nibble = elem j (0-15), high nibble = elem j+16

static inline void s2o_unpack_q4_0_neon(
    const uint8_t * qs,
    int8x16_t & out_lo,    // elems 0-15
    int8x16_t & out_hi     // elems 16-31
) {
    const uint8x16_t raw = vld1q_u8(qs);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t  off  = vdupq_n_s8(8);

    uint8x16_t lo = vandq_u8(raw, mask);         // low nibbles: elems 0-15
    uint8x16_t hi = vshrq_n_u8(raw, 4);          // high nibbles: elems 16-31

    out_lo = vsubq_s8(vreinterpretq_s8_u8(lo), off);  // center: [-8, +7]
    out_hi = vsubq_s8(vreinterpretq_s8_u8(hi), off);
}

// ============================================================================
// On-the-fly FP32 → INT8 quantization (32 elements per Q4_0 block)
// ============================================================================
// Returns scale d such that original ≈ d * quantized

static inline float s2o_quantize_block_f32_to_i8_neon(const float * src, int8_t * dst) {
    // Load 32 floats in 8 groups of 4
    float32x4_t v0 = vld1q_f32(src);
    float32x4_t v1 = vld1q_f32(src + 4);
    float32x4_t v2 = vld1q_f32(src + 8);
    float32x4_t v3 = vld1q_f32(src + 12);
    float32x4_t v4 = vld1q_f32(src + 16);
    float32x4_t v5 = vld1q_f32(src + 20);
    float32x4_t v6 = vld1q_f32(src + 24);
    float32x4_t v7 = vld1q_f32(src + 28);

    // Absolute values
    float32x4_t a0 = vabsq_f32(v0);
    float32x4_t a1 = vabsq_f32(v1);
    float32x4_t a2 = vabsq_f32(v2);
    float32x4_t a3 = vabsq_f32(v3);
    float32x4_t a4 = vabsq_f32(v4);
    float32x4_t a5 = vabsq_f32(v5);
    float32x4_t a6 = vabsq_f32(v6);
    float32x4_t a7 = vabsq_f32(v7);

    // Pairwise max reduction
    float32x4_t m01 = vmaxq_f32(a0, a1);
    float32x4_t m23 = vmaxq_f32(a2, a3);
    float32x4_t m45 = vmaxq_f32(a4, a5);
    float32x4_t m67 = vmaxq_f32(a6, a7);
    float32x4_t m0123 = vmaxq_f32(m01, m23);
    float32x4_t m4567 = vmaxq_f32(m45, m67);
    float32x4_t mall = vmaxq_f32(m0123, m4567);

    // Horizontal max of 4 floats (AArch64)
    float amax = vmaxvq_f32(mall);

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

    float32x4_t vid = vdupq_n_f32(id);

    // Quantize: round(x * id), then narrow INT32 → INT16 → INT8 with saturation
    int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(v0, vid));
    int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(v1, vid));
    int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(v2, vid));
    int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(v3, vid));
    int32x4_t q4 = vcvtnq_s32_f32(vmulq_f32(v4, vid));
    int32x4_t q5 = vcvtnq_s32_f32(vmulq_f32(v5, vid));
    int32x4_t q6 = vcvtnq_s32_f32(vmulq_f32(v6, vid));
    int32x4_t q7 = vcvtnq_s32_f32(vmulq_f32(v7, vid));

    // Narrow: INT32 → INT16 (saturating)
    int16x4_t n0 = vqmovn_s32(q0);
    int16x4_t n1 = vqmovn_s32(q1);
    int16x4_t n2 = vqmovn_s32(q2);
    int16x4_t n3 = vqmovn_s32(q3);
    int16x4_t n4 = vqmovn_s32(q4);
    int16x4_t n5 = vqmovn_s32(q5);
    int16x4_t n6 = vqmovn_s32(q6);
    int16x4_t n7 = vqmovn_s32(q7);

    // Combine pairs: INT16x4 → INT16x8
    int16x8_t h01 = vcombine_s16(n0, n1);
    int16x8_t h23 = vcombine_s16(n2, n3);
    int16x8_t h45 = vcombine_s16(n4, n5);
    int16x8_t h67 = vcombine_s16(n6, n7);

    // Narrow: INT16 → INT8 (saturating)
    int8x8_t b01 = vqmovn_s16(h01);
    int8x8_t b23 = vqmovn_s16(h23);
    int8x8_t b45 = vqmovn_s16(h45);
    int8x8_t b67 = vqmovn_s16(h67);

    // Store 32 INT8 values
    vst1_s8(dst,      b01);
    vst1_s8(dst + 8,  b23);
    vst1_s8(dst + 16, b45);
    vst1_s8(dst + 24, b67);

    return d;
}

// ============================================================================
// INT8 × INT8 dot product — baseline NEON (no DOTPROD)
// ============================================================================
// Uses vmull_s8 (signed 8×8→16 multiply) + vpaddlq_s16 (pairwise add to INT32)
// Processes 16 elements at a time (two int8x8 halves)

static inline int32x4_t s2o_dot_i8_neon(int8x16_t a, int8x16_t b) {
    // Multiply low and high halves: 8×8 → 16-bit
    int16x8_t prod_lo = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    // Pairwise add INT16 → INT32 and accumulate
    int32x4_t sum = vpaddlq_s16(prod_lo);
    sum = vpadalq_s16(sum, prod_hi);
    return sum;
}

// ============================================================================
// Q4_0 GEMV — baseline NEON
// ============================================================================

static void s2o_lut_gemv_q4_0_neon(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb
) {
    const int64_t nb_k = K / QK4_0;

    // Pre-quantize all activation blocks to INT8
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8_neon(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    for (int64_t j = j_start; j < j_end; j++) {
        const block_q4_0 * wr = (const block_q4_0 *)((const char *)src_wt + j * nb);

        float sum = 0.0f;

        for (int64_t b = 0; b < nb_k; b++) {
            const float combined_d = GGML_FP16_TO_FP32(wr[b].d) * act_d[b];

            // Unpack Q4_0 weights
            int8x16_t qw_lo, qw_hi;
            s2o_unpack_q4_0_neon(wr[b].qs, qw_lo, qw_hi);

            // Load pre-quantized INT8 activations
            int8x16_t qa_lo = vld1q_s8(act_q8 + b * QK4_0);       // elems 0-15
            int8x16_t qa_hi = vld1q_s8(act_q8 + b * QK4_0 + 16);  // elems 16-31

            // INT8 dot products (16 elements each)
            int32x4_t dot_lo = s2o_dot_i8_neon(qw_lo, qa_lo);
            int32x4_t dot_hi = s2o_dot_i8_neon(qw_hi, qa_hi);

            // Sum all 8 INT32 accumulators → single float
            int32x4_t dot_sum = vaddq_s32(dot_lo, dot_hi);
            int32_t total = vaddvq_s32(dot_sum);

            sum += combined_d * (float)total;
        }

        dst[j] = sum;
    }
}

// ============================================================================
// Q4_0 GEMM — baseline NEON (batched, for prompt processing)
// ============================================================================

static void s2o_lut_gemm_q4_0_neon(
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
) {
    for (int64_t i = 0; i < M; i++) {
        s2o_lut_gemv_q4_0_neon(
            dst + i * dst_stride,
            src_act + i * act_stride,
            src_wt,
            K, j_start, j_end, nb
        );
    }
}

// ============================================================================
// Dispatch table: baseline NEON
// ============================================================================

const s2o_lut_kernels s2o_lut_kernels_neon = {
    /* .name      = */ "neon",
    /* .gemv_q4_0 = */ s2o_lut_gemv_q4_0_neon,
    /* .gemm_q4_0 = */ s2o_lut_gemm_q4_0_neon,
};

// ============================================================================
// DOTPROD variant: uses vdotq_s32 for ~4x fewer instructions in inner loop
// ============================================================================

#if defined(__ARM_FEATURE_DOTPROD)

static inline int32x4_t s2o_dot_i8_dotprod(int8x16_t a, int8x16_t b) {
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, a, b);
    return acc;
}

static void s2o_lut_gemv_q4_0_neon_dotprod(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb
) {
    const int64_t nb_k = K / QK4_0;

    // Pre-quantize activations
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8_neon(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    for (int64_t j = j_start; j < j_end; j++) {
        const block_q4_0 * wr = (const block_q4_0 *)((const char *)src_wt + j * nb);

        float sum = 0.0f;

        for (int64_t b = 0; b < nb_k; b++) {
            const float combined_d = GGML_FP16_TO_FP32(wr[b].d) * act_d[b];

            int8x16_t qw_lo, qw_hi;
            s2o_unpack_q4_0_neon(wr[b].qs, qw_lo, qw_hi);

            int8x16_t qa_lo = vld1q_s8(act_q8 + b * QK4_0);
            int8x16_t qa_hi = vld1q_s8(act_q8 + b * QK4_0 + 16);

            // DOTPROD: 4 multiply-accumulates per instruction (vs 1 in baseline)
            int32x4_t dot_lo = s2o_dot_i8_dotprod(qw_lo, qa_lo);
            int32x4_t dot_hi = s2o_dot_i8_dotprod(qw_hi, qa_hi);

            int32x4_t dot_sum = vaddq_s32(dot_lo, dot_hi);
            int32_t total = vaddvq_s32(dot_sum);

            sum += combined_d * (float)total;
        }

        dst[j] = sum;
    }
}

static void s2o_lut_gemm_q4_0_neon_dotprod(
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
) {
    for (int64_t i = 0; i < M; i++) {
        s2o_lut_gemv_q4_0_neon_dotprod(
            dst + i * dst_stride,
            src_act + i * act_stride,
            src_wt,
            K, j_start, j_end, nb
        );
    }
}

const s2o_lut_kernels s2o_lut_kernels_neon_dotprod = {
    /* .name      = */ "neon_dotprod",
    /* .gemv_q4_0 = */ s2o_lut_gemv_q4_0_neon_dotprod,
    /* .gemm_q4_0 = */ s2o_lut_gemm_q4_0_neon_dotprod,
};

#else

// Stub when compiled without DOTPROD — dispatch will use baseline NEON
const s2o_lut_kernels s2o_lut_kernels_neon_dotprod = {
    /* .name      = */ "neon_dotprod",
    /* .gemv_q4_0 = */ s2o_lut_gemv_q4_0_neon,
    /* .gemm_q4_0 = */ s2o_lut_gemm_q4_0_neon,
};

#endif // __ARM_FEATURE_DOTPROD

#endif // __ARM_NEON
