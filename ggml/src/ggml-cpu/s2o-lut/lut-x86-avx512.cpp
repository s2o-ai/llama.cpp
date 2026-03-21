// S2O LUT-based INT4 GEMV/GEMM — AVX-512 implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 dual-column processing: processes 2 output columns per iteration
// using 512-bit integer dot product, doubling throughput vs AVX2.
//
// Strategy (same integer dot product as AVX2, but wider):
//   1. Quantize FP32 activations to INT8 on the fly (16 floats/instr with AVX-512)
//   2. For each pair of output columns:
//      a. Unpack Q4_0 nibbles to 32 bytes per column (256-bit)
//      b. Combine weights from 2 columns into one 512-bit register
//      c. Duplicate INT8 activations into 512-bit register
//      d. 512-bit VPMADDUBSW + VPMADDWD integer dot product
//      e. Split result back into per-column accumulators
//   3. Scale by (d_weight * d_activation), accumulate
//
// Uses only AVX512F + AVX512BW intrinsics (no DQ dependency).
// VPMOVDB for efficient INT32→INT8 packing during quantization.
// Mask-based sign handling replaces VPSIGNB (which only exists for 128/256-bit).

#if defined(__AVX512F__) && defined(__AVX512BW__)

#include "lut-common.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// 512-bit lane helpers (AVX512F only, no DQ required)
// ============================================================================

// Combine two __m256i into __m512i: [lo | hi]
// Uses VSHUFI32X4 (AVX512F) instead of VINSERTI64X4 (AVX512DQ)
static inline __m512i s2o_set_m256i(const __m256i lo, const __m256i hi) {
    const __m512i lo_512 = _mm512_castsi256_si512(lo);
    const __m512i hi_512 = _mm512_castsi256_si512(hi);
    // imm 0x44 = 01_00_01_00: lane0=a[0], lane1=a[1], lane2=b[0], lane3=b[1]
    return _mm512_shuffle_i32x4(lo_512, hi_512, 0x44);
}

// Extract upper 256 bits of __m512 as __m256
// Uses VSHUFF32X4 (AVX512F) instead of VEXTRACTF32X8 (AVX512DQ)
static inline __m256 s2o_extract_hi_ps(const __m512 v) {
    // imm 0x4E = 01_00_11_10: swaps [2,3] to low, [0,1] to high
    return _mm512_castps512_ps256(_mm512_shuffle_f32x4(v, v, 0x4E));
}

// ============================================================================
// AVX2 helpers (for per-block 256-bit operations)
// ============================================================================

// Unpack 16 nibble-pair bytes into 32 bytes in [0..15]
static inline __m256i s2o_bytes_from_nibbles_32(const uint8_t * qs) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)qs);
    const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0x0F), bytes);
}

// INT8 × INT8 → FP32, 256-bit (for trailing odd column)
static inline __m256 s2o_mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed = _mm256_madd_epi16(ones, dot);
    return _mm256_cvtepi32_ps(summed);
}

// INT8 × INT8 → FP32, 512-bit (for dual-column processing)
// AVX-512 lacks VPSIGNB, so we use mask + conditional negate.
static inline __m512 s2o_mul_sum_i8_pairs_float_512(const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    // Mask where x < 0
    const __mmask64 neg = _mm512_cmpgt_epi8_mask(zero, x);
    // |x| (unsigned operand for VPMADDUBSW)
    const __m512i ax = _mm512_abs_epi8(x);
    // Conditionally negate y where x was negative
    const __m512i ny = _mm512_sub_epi8(zero, y);
    const __m512i sy = _mm512_mask_blend_epi8(neg, y, ny);
    // VPMADDUBSW: unsigned(|x|) × signed(sy) → 32 INT16
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    // VPMADDWD: pairwise add INT16 → 16 INT32
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i summed = _mm512_madd_epi16(ones, dot);
    return _mm512_cvtepi32_ps(summed);
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
// On-the-fly FP32 → INT8 quantization — AVX-512 accelerated
// ============================================================================
// 16 floats per instruction (2x throughput vs AVX2).
// VPMOVDB for direct INT32→INT8 packing (no lane reorder needed).

static inline float s2o_quantize_block_f32_to_i8(const float * src, int8_t * dst) {
    // Load 32 floats as 2 × __m512 (16 each)
    __m512 v0 = _mm512_loadu_ps(src);
    __m512 v1 = _mm512_loadu_ps(src + 16);

    // Absolute values via integer AND (avoids AVX512DQ _mm512_andnot_ps)
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v0), abs_mask));
    __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v1), abs_mask));

    // Max absolute value across all 32 elements
    float amax = _mm512_reduce_max_ps(_mm512_max_ps(a0, a1));

    // Scale: map [-amax, amax] to [-127, 127]
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

    // Quantize: round(x * id), clamp to [-128, 127]
    __m512 vid = _mm512_set1_ps(id);
    __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid));
    __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid));

    // Pack INT32 → INT8 with saturation (VPMOVDB), 16 values per instruction
    _mm_storeu_si128((__m128i *)dst,        _mm512_cvtsepi32_epi8(q0));
    _mm_storeu_si128((__m128i *)(dst + 16), _mm512_cvtsepi32_epi8(q1));

    return d;
}

// ============================================================================
// Q4_0 GEMV — AVX-512 (dual-column integer dot product)
// ============================================================================

static void s2o_lut_gemv_q4_0_avx512(
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

    // Pre-quantize all activation blocks to INT8 (done once, reused across columns)
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    // Process 2 columns at a time using 512-bit integer dot product
    int64_t j = j_start;
    for (; j + 1 < j_end; j += 2) {
        const block_q4_0 * wr0 = (const block_q4_0 *)((const char *)src_wt + j * nb);
        const block_q4_0 * wr1 = (const block_q4_0 *)((const char *)src_wt + (j + 1) * nb);

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const float d0 = GGML_FP16_TO_FP32(wr0[b].d) * act_d[b];
            const float d1 = GGML_FP16_TO_FP32(wr1[b].d) * act_d[b];

            // Unpack Q4_0 nibbles → 32 bytes, subtract 8 to center in [-8..+7]
            __m256i qw0 = _mm256_sub_epi8(s2o_bytes_from_nibbles_32(wr0[b].qs), off);
            __m256i qw1 = _mm256_sub_epi8(s2o_bytes_from_nibbles_32(wr1[b].qs), off);

            // Combine into 512-bit: [col_j weights | col_j+1 weights]
            __m512i qw = s2o_set_m256i(qw0, qw1);

            // Duplicate INT8 activations into 512-bit
            __m256i qa256 = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));
            __m512i qa = s2o_set_m256i(qa256, qa256);

            // 512-bit integer dot product → 16 FP32
            __m512 q = s2o_mul_sum_i8_pairs_float_512(qw, qa);

            // Lower 8 FP32 → col j partial, upper 8 → col j+1
            __m256 q_lo = _mm512_castps512_ps256(q);
            __m256 q_hi = s2o_extract_hi_ps(q);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), q_lo, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), q_hi, acc1);
        }

        dst[j]     = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
    }

    // Handle trailing odd column with 256-bit path
    if (j < j_end) {
        const block_q4_0 * wr = (const block_q4_0 *)((const char *)src_wt + j * nb);

        __m256 acc = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const float combined_d = GGML_FP16_TO_FP32(wr[b].d) * act_d[b];
            __m256i qw = _mm256_sub_epi8(s2o_bytes_from_nibbles_32(wr[b].qs), off);
            __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));
            __m256 q = s2o_mul_sum_i8_pairs_float(qw, qa);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(combined_d), q, acc);
        }

        dst[j] = s2o_hsum_float_8(acc);
    }
}

// ============================================================================
// Q4_0 GEMM — AVX-512 (batched, for prompt processing)
// ============================================================================

static void s2o_lut_gemm_q4_0_avx512(
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
        s2o_lut_gemv_q4_0_avx512(
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

const s2o_lut_kernels s2o_lut_kernels_avx512 = {
    /* .name      = */ "avx512",
    /* .gemv_q4_0 = */ s2o_lut_gemv_q4_0_avx512,
    /* .gemm_q4_0 = */ s2o_lut_gemm_q4_0_avx512,
};

#endif // defined(__AVX512F__) && defined(__AVX512BW__)
