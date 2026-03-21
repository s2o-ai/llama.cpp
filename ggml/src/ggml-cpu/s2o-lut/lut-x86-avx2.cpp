// S2O LUT-based INT4 GEMV/GEMM — AVX2 implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// High-performance Q4_0 × FP32 dot product for AVX2.
//
// Strategy:
//   1. Quantize FP32 activations to INT8 on the fly (per Q4_0 block of 32)
//   2. Unpack Q4_0 nibbles to 32 bytes using bytes_from_nibbles_32 pattern
//   3. Subtract bias of 8 to center weights in [-8..+7]
//   4. Use VPMADDUBSW + VPMADDWD for integer dot product → INT32
//   5. Convert to FP32, scale by (d_weight * d_activation), accumulate
//
// This matches ggml's own Q4_0 × Q8_0 fast path, except we quantize
// the activation on the fly instead of requiring pre-quantized input.

#if defined(__AVX2__)

#include "lut-common.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// AVX2 helpers (same patterns as ggml arch/x86/quants.c)
// ============================================================================

// Unpack 16 nibble-pair bytes into 32 bytes in [0..15]
static inline __m256i s2o_bytes_from_nibbles_32(const uint8_t * qs) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)qs);
    // low 128: original bytes (low nibbles after mask)
    // high 128: bytes >> 4 (high nibbles after mask)
    const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0x0F), bytes);
}

// INT8 × INT8 multiply, pairwise add to INT16, then pairwise add to INT32, convert to FP32
// Uses VPMADDUBSW (unsigned × signed → INT16) + VPMADDWD (INT16 → INT32)
static inline __m256 s2o_mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of y to compensate
    const __m256i sy = _mm256_sign_epi8(y, x);
    // VPMADDUBSW: unsigned(ax) × signed(sy) → 16 INT16 values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    // VPMADDWD: pairwise add INT16 → 8 INT32, then convert to FP32
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed = _mm256_madd_epi16(ones, dot);
    return _mm256_cvtepi32_ps(summed);
}

// Horizontal sum of 8 floats
static inline float s2o_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ============================================================================
// On-the-fly FP32 → INT8 quantization for a block of 32 activations
// ============================================================================
// Returns the inverse scale (1/d) used, and writes 32 INT8 values to dst.
// This mimics ggml's Q8_0 quantization but just for a single block.

static inline float s2o_quantize_block_f32_to_i8(const float * src, int8_t * dst) {
    // Find abs max across 32 floats using AVX2
    __m256 v0 = _mm256_loadu_ps(src);
    __m256 v1 = _mm256_loadu_ps(src + 8);
    __m256 v2 = _mm256_loadu_ps(src + 16);
    __m256 v3 = _mm256_loadu_ps(src + 24);

    // Absolute values
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 a0 = _mm256_andnot_ps(sign_mask, v0);
    __m256 a1 = _mm256_andnot_ps(sign_mask, v1);
    __m256 a2 = _mm256_andnot_ps(sign_mask, v2);
    __m256 a3 = _mm256_andnot_ps(sign_mask, v3);

    // Max across all 32
    __m256 mx = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
    // Horizontal max of 8 floats
    __m128 hi = _mm256_extractf128_ps(mx, 1);
    __m128 lo = _mm256_castps256_ps128(mx);
    __m128 m128 = _mm_max_ps(lo, hi);
    m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
    float amax = _mm_cvtss_f32(m128);

    // Scale: map [-amax, amax] to [-127, 127]
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

    // Quantize: round(x * id), clamp to [-128, 127]
    __m256 vid = _mm256_set1_ps(id);
    __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
    __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
    __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
    __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));

    // Pack INT32 → INT16 → INT8
    __m256i q01 = _mm256_packs_epi32(q0, q1);  // 16 INT16
    __m256i q23 = _mm256_packs_epi32(q2, q3);  // 16 INT16
    __m256i q_i8 = _mm256_packs_epi16(q01, q23); // 32 INT8

    // AVX2 packs operates within 128-bit lanes, so we need to permute
    // to get the correct byte order: [0,1,4,5,2,3,6,7] → [0,1,2,3,4,5,6,7]
    q_i8 = _mm256_permutevar8x32_epi32(q_i8,
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

    _mm256_storeu_si256((__m256i *)dst, q_i8);

    return d;
}

// ============================================================================
// Q4_0 GEMV — AVX2 (high-performance integer dot product path)
// ============================================================================

static void s2o_lut_gemv_q4_0_avx2(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb
) {
    const int64_t nb_k = K / QK4_0;
    const __m256i off = _mm256_set1_epi8(8);

    // Pre-quantize all activation blocks to INT8
    // This is done once and reused across all output columns
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    for (int64_t j = j_start; j < j_end; j++) {
        const block_q4_0 * wt_row = (const block_q4_0 *)((const char *)src_wt + j * nb);

        __m256 acc = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            // Combined scale = weight_d * activation_d
            const float combined_d = GGML_FP16_TO_FP32(wt_row[b].d) * act_d[b];
            const __m256 vd = _mm256_set1_ps(combined_d);

            // Unpack Q4_0 nibbles to 32 bytes in [0..15]
            __m256i qw = s2o_bytes_from_nibbles_32(wt_row[b].qs);

            // Subtract 8 to center in [-8..+7]
            qw = _mm256_sub_epi8(qw, off);

            // Load pre-quantized INT8 activations
            __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            // Integer dot product: 32 INT8 × INT8 → 8 FP32
            const __m256 q = s2o_mul_sum_i8_pairs_float(qw, qa);

            // Scale and accumulate
            acc = _mm256_fmadd_ps(vd, q, acc);
        }

        dst[j] = s2o_hsum_float_8(acc);
    }
}

// ============================================================================
// Q4_0 GEMM — AVX2 (batched, for prompt processing)
// ============================================================================

static void s2o_lut_gemm_q4_0_avx2(
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
        s2o_lut_gemv_q4_0_avx2(
            dst + i * dst_stride,
            src_act + i * act_stride,
            src_wt,
            K, j_start, j_end, nb
        );
    }
}

// ============================================================================
// Kernel dispatch table
// ============================================================================

const s2o_lut_kernels s2o_lut_kernels_avx2 = {
    /* .name      = */ "avx2",
    /* .gemv_q4_0 = */ s2o_lut_gemv_q4_0_avx2,
    /* .gemm_q4_0 = */ s2o_lut_gemm_q4_0_avx2,
};

#endif // defined(__AVX2__)
