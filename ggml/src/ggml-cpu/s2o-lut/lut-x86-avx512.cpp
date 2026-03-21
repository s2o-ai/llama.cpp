// S2O LUT-based INT4 GEMV/GEMM — AVX-512 implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 4-wide column processing with VPSHUFB dequantization:
//   - Processes 4 output columns per GEMV iteration (2 pairs of 2, each pair
//     combined into a 512-bit register for dual-column dot product)
//   - VPSHUFB replaces nibble-unpack arithmetic with 16-entry LUT
//   - Software prefetch for weight blocks
//   - L2 cache-aware tiling for GEMM
//
// Uses only AVX512F + AVX512BW intrinsics (no DQ dependency).

#if defined(__AVX512F__) && defined(__AVX512BW__)

#include "lut-common.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// 512-bit lane helpers (AVX512F only, no DQ required)
// ============================================================================

static inline __m512i s2o_set_m256i(const __m256i lo, const __m256i hi) {
    const __m512i lo_512 = _mm512_castsi256_si512(lo);
    const __m512i hi_512 = _mm512_castsi256_si512(hi);
    return _mm512_shuffle_i32x4(lo_512, hi_512, 0x44);
}

static inline __m256 s2o_extract_hi_ps(const __m512 v) {
    return _mm512_castps512_ps256(_mm512_shuffle_f32x4(v, v, 0x4E));
}

// ============================================================================
// VPSHUFB dequantization: nibble → signed INT8 via constant LUT
// ============================================================================

static inline __m256i s2o_vpshufb_dequant_q4_0(const uint8_t * qs) {
    const __m128i lut  = _mm_load_si128((const __m128i *)s2o_q4_dequant_table);
    const __m128i m4b  = _mm_set1_epi8(0x0F);
    const __m128i raw  = _mm_loadu_si128((const __m128i *)qs);

    const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m4b));
    const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m4b));

    return _mm256_set_m128i(hi, lo);
}

// ============================================================================
// INT8 × INT8 → FP32 dot products (256-bit and 512-bit)
// ============================================================================

// 256-bit path (for trailing odd column)
static inline __m256 s2o_mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed = _mm256_madd_epi16(ones, dot);
    return _mm256_cvtepi32_ps(summed);
}

// 512-bit path (for dual-column processing)
static inline __m512 s2o_mul_sum_i8_pairs_float_512(const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    const __mmask64 neg = _mm512_cmpgt_epi8_mask(zero, x);
    const __m512i ax = _mm512_abs_epi8(x);
    const __m512i ny = _mm512_sub_epi8(zero, y);
    const __m512i sy = _mm512_mask_blend_epi8(neg, y, ny);
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
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

static inline float s2o_quantize_block_f32_to_i8(const float * src, int8_t * dst) {
    __m512 v0 = _mm512_loadu_ps(src);
    __m512 v1 = _mm512_loadu_ps(src + 16);

    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v0), abs_mask));
    __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v1), abs_mask));

    float amax = _mm512_reduce_max_ps(_mm512_max_ps(a0, a1));

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

    __m512 vid = _mm512_set1_ps(id);
    __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid));
    __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid));

    _mm_storeu_si128((__m128i *)dst,        _mm512_cvtsepi32_epi8(q0));
    _mm_storeu_si128((__m128i *)(dst + 16), _mm512_cvtsepi32_epi8(q1));

    return d;
}

// ============================================================================
// Q4_0 GEMV — AVX-512, 4-wide column processing
// ============================================================================
// Processes 4 columns per iteration as 2 pairs, each pair using 512-bit
// dual-column dot product. This is 2x the throughput of the previous 2-wide.

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

    // Pre-quantize all activation blocks to INT8
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    // ---- 4-wide main loop (2 pairs of dual-column 512-bit) ----
    int64_t j = j_start;
    for (; j + 3 < j_end; j += 4) {
        const block_q4_0 * wr0 = (const block_q4_0 *)((const char *)src_wt + (j + 0) * nb);
        const block_q4_0 * wr1 = (const block_q4_0 *)((const char *)src_wt + (j + 1) * nb);
        const block_q4_0 * wr2 = (const block_q4_0 *)((const char *)src_wt + (j + 2) * nb);
        const block_q4_0 * wr3 = (const block_q4_0 *)((const char *)src_wt + (j + 3) * nb);

        // 4 accumulators (256-bit each, extracted from 512-bit results)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            // Prefetch
            if (b + S2O_LUT_PREFETCH_DIST < nb_k) {
                _mm_prefetch((const char *)&wr0[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char *)&wr1[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char *)&wr2[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char *)&wr3[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
            }

            const float d0 = GGML_FP16_TO_FP32(wr0[b].d) * act_d[b];
            const float d1 = GGML_FP16_TO_FP32(wr1[b].d) * act_d[b];
            const float d2 = GGML_FP16_TO_FP32(wr2[b].d) * act_d[b];
            const float d3 = GGML_FP16_TO_FP32(wr3[b].d) * act_d[b];

            // VPSHUFB dequantize 4 weight blocks → 4 × 256-bit signed INT8
            const __m256i qw0 = s2o_vpshufb_dequant_q4_0(wr0[b].qs);
            const __m256i qw1 = s2o_vpshufb_dequant_q4_0(wr1[b].qs);
            const __m256i qw2 = s2o_vpshufb_dequant_q4_0(wr2[b].qs);
            const __m256i qw3 = s2o_vpshufb_dequant_q4_0(wr3[b].qs);

            // Activations as 256-bit and 512-bit
            const __m256i qa256 = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            // Pair 1: columns j+0 and j+1 via 512-bit dual-column
            {
                const __m512i qw_pair = s2o_set_m256i(qw0, qw1);
                const __m512i qa_pair = s2o_set_m256i(qa256, qa256);
                const __m512 q = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);

                const __m256 q_lo = _mm512_castps512_ps256(q);
                const __m256 q_hi = s2o_extract_hi_ps(q);

                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), q_lo, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), q_hi, acc1);
            }

            // Pair 2: columns j+2 and j+3 via 512-bit dual-column
            {
                const __m512i qw_pair = s2o_set_m256i(qw2, qw3);
                const __m512i qa_pair = s2o_set_m256i(qa256, qa256);
                const __m512 q = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);

                const __m256 q_lo = _mm512_castps512_ps256(q);
                const __m256 q_hi = s2o_extract_hi_ps(q);

                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), q_lo, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), q_hi, acc3);
            }
        }

        dst[j + 0] = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
        dst[j + 2] = s2o_hsum_float_8(acc2);
        dst[j + 3] = s2o_hsum_float_8(acc3);
    }

    // ---- 2-wide remainder (512-bit dual-column) ----
    if (j + 1 < j_end) {
        const block_q4_0 * wr0 = (const block_q4_0 *)((const char *)src_wt + j * nb);
        const block_q4_0 * wr1 = (const block_q4_0 *)((const char *)src_wt + (j + 1) * nb);

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const float d0 = GGML_FP16_TO_FP32(wr0[b].d) * act_d[b];
            const float d1 = GGML_FP16_TO_FP32(wr1[b].d) * act_d[b];

            const __m256i qw0 = s2o_vpshufb_dequant_q4_0(wr0[b].qs);
            const __m256i qw1 = s2o_vpshufb_dequant_q4_0(wr1[b].qs);
            const __m256i qa256 = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            const __m512i qw_pair = s2o_set_m256i(qw0, qw1);
            const __m512i qa_pair = s2o_set_m256i(qa256, qa256);
            const __m512 q = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm512_castps512_ps256(q), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s2o_extract_hi_ps(q), acc1);
        }

        dst[j]     = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
        j += 2;
    }

    // ---- 1-wide trailing column (256-bit) ----
    if (j < j_end) {
        const block_q4_0 * wr = (const block_q4_0 *)((const char *)src_wt + j * nb);
        __m256 acc = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const float combined_d = GGML_FP16_TO_FP32(wr[b].d) * act_d[b];
            const __m256i qw = s2o_vpshufb_dequant_q4_0(wr[b].qs);
            const __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(combined_d), s2o_mul_sum_i8_pairs_float(qw, qa), acc);
        }

        dst[j] = s2o_hsum_float_8(acc);
    }
}

// ============================================================================
// Q4_0 GEMM — AVX-512 (L2 cache-aware tiling)
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
    const int64_t bytes_per_col = (K / QK4_0) * sizeof(block_q4_0);
    int64_t tile_n = S2O_LUT_DEFAULT_L2_TILE_BYTES / bytes_per_col;
    if (tile_n < 4) tile_n = 4;
    tile_n = (tile_n / 4) * 4;

    const int64_t N = j_end - j_start;

    if (M <= 1 || N <= tile_n) {
        for (int64_t i = 0; i < M; i++) {
            s2o_lut_gemv_q4_0_avx512(
                dst + i * dst_stride,
                src_act + i * act_stride,
                src_wt,
                K, j_start, j_end, nb
            );
        }
        return;
    }

    for (int64_t j_tile = j_start; j_tile < j_end; j_tile += tile_n) {
        const int64_t j_tile_end = std::min(j_tile + tile_n, j_end);

        for (int64_t i = 0; i < M; i++) {
            s2o_lut_gemv_q4_0_avx512(
                dst + i * dst_stride,
                src_act + i * act_stride,
                src_wt,
                K, j_tile, j_tile_end, nb
            );
        }
    }
}

// ============================================================================
// Q4_0 GEMV — AVX-512, packed (column-interleaved) layout
// ============================================================================
// Same 4-wide dual-pair algorithm, but weight blocks for each 4-column group
// are contiguous: group_base[b*4 + c] for block b, column c in [0,3].

static void s2o_lut_gemv_q4_0_packed_avx512(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb
) {
    (void)nb;
    const int64_t nb_k = K / QK4_0;
    const block_q4_0 * wt = (const block_q4_0 *)src_wt;

    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    int64_t j = j_start;

    // Handle unaligned prefix
    const int64_t j_aligned = ((j_start + 3) / 4) * 4;
    for (; j < std::min(j_aligned, j_end); j++) {
        const int64_t group = j / 4;
        const int64_t c = j % 4;
        const block_q4_0 * group_base = wt + group * nb_k * 4;

        __m256 acc = _mm256_setzero_ps();
        for (int64_t b = 0; b < nb_k; b++) {
            const block_q4_0 * wr = &group_base[b * 4 + c];
            const float combined_d = GGML_FP16_TO_FP32(wr->d) * act_d[b];
            const __m256i qw = s2o_vpshufb_dequant_q4_0(wr->qs);
            const __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(combined_d), s2o_mul_sum_i8_pairs_float(qw, qa), acc);
        }
        dst[j] = s2o_hsum_float_8(acc);
    }

    // ---- 4-wide packed loop (2 pairs of dual-column 512-bit) ----
    for (; j + 3 < j_end; j += 4) {
        const int64_t group = j / 4;
        const block_q4_0 * group_base = wt + group * nb_k * 4;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            // Contiguous: 4 blocks at group_base[b*4 + 0..3]
            const block_q4_0 * wr0 = &group_base[b * 4 + 0];
            const block_q4_0 * wr1 = &group_base[b * 4 + 1];
            const block_q4_0 * wr2 = &group_base[b * 4 + 2];
            const block_q4_0 * wr3 = &group_base[b * 4 + 3];

            if (b + S2O_LUT_PREFETCH_DIST < nb_k) {
                _mm_prefetch((const char *)&group_base[(b + S2O_LUT_PREFETCH_DIST) * 4], _MM_HINT_T0);
            }

            const float d0 = GGML_FP16_TO_FP32(wr0->d) * act_d[b];
            const float d1 = GGML_FP16_TO_FP32(wr1->d) * act_d[b];
            const float d2 = GGML_FP16_TO_FP32(wr2->d) * act_d[b];
            const float d3 = GGML_FP16_TO_FP32(wr3->d) * act_d[b];

            const __m256i qw0 = s2o_vpshufb_dequant_q4_0(wr0->qs);
            const __m256i qw1 = s2o_vpshufb_dequant_q4_0(wr1->qs);
            const __m256i qw2 = s2o_vpshufb_dequant_q4_0(wr2->qs);
            const __m256i qw3 = s2o_vpshufb_dequant_q4_0(wr3->qs);

            const __m256i qa256 = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            // Pair 1: columns 0 and 1 via 512-bit
            {
                const __m512i qw_pair = s2o_set_m256i(qw0, qw1);
                const __m512i qa_pair = s2o_set_m256i(qa256, qa256);
                const __m512 q = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm512_castps512_ps256(q), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s2o_extract_hi_ps(q), acc1);
            }

            // Pair 2: columns 2 and 3 via 512-bit
            {
                const __m512i qw_pair = s2o_set_m256i(qw2, qw3);
                const __m512i qa_pair = s2o_set_m256i(qa256, qa256);
                const __m512 q = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm512_castps512_ps256(q), acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), s2o_extract_hi_ps(q), acc3);
            }
        }

        dst[j + 0] = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
        dst[j + 2] = s2o_hsum_float_8(acc2);
        dst[j + 3] = s2o_hsum_float_8(acc3);
    }

    // ---- 2-wide remainder ----
    if (j + 1 < j_end) {
        const int64_t group = j / 4;
        const int64_t c = j % 4;
        const block_q4_0 * group_base = wt + group * nb_k * 4;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const block_q4_0 * wr0 = &group_base[b * 4 + c];
            const block_q4_0 * wr1 = &group_base[b * 4 + c + 1];

            const float d0 = GGML_FP16_TO_FP32(wr0->d) * act_d[b];
            const float d1 = GGML_FP16_TO_FP32(wr1->d) * act_d[b];

            const __m256i qw0 = s2o_vpshufb_dequant_q4_0(wr0->qs);
            const __m256i qw1 = s2o_vpshufb_dequant_q4_0(wr1->qs);
            const __m256i qa256 = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            const __m512i qw_pair = s2o_set_m256i(qw0, qw1);
            const __m512i qa_pair = s2o_set_m256i(qa256, qa256);
            const __m512 q = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm512_castps512_ps256(q), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s2o_extract_hi_ps(q), acc1);
        }

        dst[j]     = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
        j += 2;
    }

    // ---- 1-wide trailing ----
    if (j < j_end) {
        const int64_t group = j / 4;
        const int64_t c = j % 4;
        const block_q4_0 * group_base = wt + group * nb_k * 4;

        __m256 acc = _mm256_setzero_ps();
        for (int64_t b = 0; b < nb_k; b++) {
            const block_q4_0 * wr = &group_base[b * 4 + c];
            const float combined_d = GGML_FP16_TO_FP32(wr->d) * act_d[b];
            const __m256i qw = s2o_vpshufb_dequant_q4_0(wr->qs);
            const __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(combined_d), s2o_mul_sum_i8_pairs_float(qw, qa), acc);
        }
        dst[j] = s2o_hsum_float_8(acc);
    }
}

// ============================================================================
// Q4_0 GEMM — AVX-512, packed layout (L2 cache-aware tiling)
// ============================================================================

static void s2o_lut_gemm_q4_0_packed_avx512(
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
    const int64_t bytes_per_col = (K / QK4_0) * sizeof(block_q4_0);
    int64_t tile_n = S2O_LUT_DEFAULT_L2_TILE_BYTES / bytes_per_col;
    if (tile_n < 4) tile_n = 4;
    tile_n = (tile_n / 4) * 4;

    const int64_t N = j_end - j_start;

    if (M <= 1 || N <= tile_n) {
        for (int64_t i = 0; i < M; i++) {
            s2o_lut_gemv_q4_0_packed_avx512(
                dst + i * dst_stride,
                src_act + i * act_stride,
                src_wt,
                K, j_start, j_end, nb
            );
        }
        return;
    }

    for (int64_t j_tile = j_start; j_tile < j_end; j_tile += tile_n) {
        const int64_t j_tile_end = std::min(j_tile + tile_n, j_end);

        for (int64_t i = 0; i < M; i++) {
            s2o_lut_gemv_q4_0_packed_avx512(
                dst + i * dst_stride,
                src_act + i * act_stride,
                src_wt,
                K, j_tile, j_tile_end, nb
            );
        }
    }
}

// ============================================================================
// Kernel dispatch table
// ============================================================================

const s2o_lut_kernels s2o_lut_kernels_avx512 = {
    /* .name              = */ "avx512_4wide",
    /* .gemv_q4_0         = */ s2o_lut_gemv_q4_0_avx512,
    /* .gemm_q4_0         = */ s2o_lut_gemm_q4_0_avx512,
    /* .gemv_q4_0_packed  = */ s2o_lut_gemv_q4_0_packed_avx512,
    /* .gemm_q4_0_packed  = */ s2o_lut_gemm_q4_0_packed_avx512,
};

#endif // defined(__AVX512F__) && defined(__AVX512BW__)
