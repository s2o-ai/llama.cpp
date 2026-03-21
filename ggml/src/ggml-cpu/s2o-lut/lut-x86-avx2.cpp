// S2O LUT-based INT4 GEMV/GEMM — AVX2 implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// High-performance Q4_0 × FP32 matmul for AVX2 with:
//   - VPSHUFB dequantization (16-entry LUT replaces nibble-unpack arithmetic)
//   - 4-wide GEMV (4 output columns per iteration, 4x bandwidth reduction)
//   - L2 cache-aware tiling for GEMM (batch/prompt processing)
//   - Software prefetch to hide memory latency
//
// Strategy for each Q4_0 block of 32 weights:
//   1. VPSHUFB: 16-byte nibble data → 32 signed INT8 via constant LUT
//   2. Pre-quantized INT8 activations (done once, reused across columns)
//   3. VPMADDUBSW + VPMADDWD for integer dot product → 8 INT32
//   4. VCVTDQ2PS → 8 FP32, scale by (d_weight × d_activation), VFMADD

#if defined(__AVX2__)

#include "lut-common.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// VPSHUFB dequantization: nibble → signed INT8 via constant LUT
// ============================================================================
// Replaces: bytes_from_nibbles_32() + _mm256_sub_epi8(x, 8)
// Uses the 16-entry table [-8,-7,...,+7] from lut-common.h

static inline __m256i s2o_vpshufb_dequant_q4_0(const uint8_t * qs) {
    const __m128i lut  = _mm_load_si128((const __m128i *)s2o_q4_dequant_table);
    const __m128i m4b  = _mm_set1_epi8(0x0F);
    const __m128i raw  = _mm_loadu_si128((const __m128i *)qs);

    // Low nibbles (elements 0-15): mask to [0..15], VPSHUFB → [-8..+7]
    const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m4b));
    // High nibbles (elements 16-31): shift right 4, mask, VPSHUFB
    const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m4b));

    // Combine: low 128 = elems 0-15, high 128 = elems 16-31
    return _mm256_set_m128i(hi, lo);
}

// ============================================================================
// INT8 × INT8 → FP32 dot product (unchanged from before)
// ============================================================================

static inline __m256 s2o_mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
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
// On-the-fly FP32 → INT8 quantization (32 activations per block)
// ============================================================================

static inline float s2o_quantize_block_f32_to_i8(const float * src, int8_t * dst) {
    __m256 v0 = _mm256_loadu_ps(src);
    __m256 v1 = _mm256_loadu_ps(src + 8);
    __m256 v2 = _mm256_loadu_ps(src + 16);
    __m256 v3 = _mm256_loadu_ps(src + 24);

    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 a0 = _mm256_andnot_ps(sign_mask, v0);
    __m256 a1 = _mm256_andnot_ps(sign_mask, v1);
    __m256 a2 = _mm256_andnot_ps(sign_mask, v2);
    __m256 a3 = _mm256_andnot_ps(sign_mask, v3);

    __m256 mx = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
    __m128 hi = _mm256_extractf128_ps(mx, 1);
    __m128 lo = _mm256_castps256_ps128(mx);
    __m128 m128 = _mm_max_ps(lo, hi);
    m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
    float amax = _mm_cvtss_f32(m128);

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

    __m256 vid = _mm256_set1_ps(id);
    __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
    __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
    __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
    __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));

    __m256i q01 = _mm256_packs_epi32(q0, q1);
    __m256i q23 = _mm256_packs_epi32(q2, q3);
    __m256i q_i8 = _mm256_packs_epi16(q01, q23);

    q_i8 = _mm256_permutevar8x32_epi32(q_i8,
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

    _mm256_storeu_si256((__m256i *)dst, q_i8);
    return d;
}

// ============================================================================
// Q4_0 GEMV — AVX2, 4-wide column processing
// ============================================================================
// Processes 4 output columns per outer iteration. Each weight block is loaded
// once and used for 4 dot products, reducing memory bandwidth by 4x.
// Falls back to 1-wide for trailing columns (remainder after 4-wide).

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

    // Pre-quantize all activation blocks to INT8 (done once, reused across all columns)
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    // ---- 4-wide main loop ----
    int64_t j = j_start;
    for (; j + 3 < j_end; j += 4) {
        const block_q4_0 * wr0 = (const block_q4_0 *)((const char *)src_wt + (j + 0) * nb);
        const block_q4_0 * wr1 = (const block_q4_0 *)((const char *)src_wt + (j + 1) * nb);
        const block_q4_0 * wr2 = (const block_q4_0 *)((const char *)src_wt + (j + 2) * nb);
        const block_q4_0 * wr3 = (const block_q4_0 *)((const char *)src_wt + (j + 3) * nb);

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            // Software prefetch: next blocks for all 4 columns
            if (b + S2O_LUT_PREFETCH_DIST < nb_k) {
                _mm_prefetch((const char *)&wr0[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char *)&wr1[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char *)&wr2[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char *)&wr3[b + S2O_LUT_PREFETCH_DIST], _MM_HINT_T0);
            }

            // Combined scales
            const float d0 = GGML_FP16_TO_FP32(wr0[b].d) * act_d[b];
            const float d1 = GGML_FP16_TO_FP32(wr1[b].d) * act_d[b];
            const float d2 = GGML_FP16_TO_FP32(wr2[b].d) * act_d[b];
            const float d3 = GGML_FP16_TO_FP32(wr3[b].d) * act_d[b];

            // VPSHUFB dequantize 4 weight blocks → 4 × 32 signed INT8
            const __m256i qw0 = s2o_vpshufb_dequant_q4_0(wr0[b].qs);
            const __m256i qw1 = s2o_vpshufb_dequant_q4_0(wr1[b].qs);
            const __m256i qw2 = s2o_vpshufb_dequant_q4_0(wr2[b].qs);
            const __m256i qw3 = s2o_vpshufb_dequant_q4_0(wr3[b].qs);

            // Load pre-quantized INT8 activations (same for all 4 columns)
            const __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            // 4 × integer dot products
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), s2o_mul_sum_i8_pairs_float(qw0, qa), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s2o_mul_sum_i8_pairs_float(qw1, qa), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), s2o_mul_sum_i8_pairs_float(qw2, qa), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), s2o_mul_sum_i8_pairs_float(qw3, qa), acc3);
        }

        dst[j + 0] = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
        dst[j + 2] = s2o_hsum_float_8(acc2);
        dst[j + 3] = s2o_hsum_float_8(acc3);
    }

    // ---- 1-wide remainder loop ----
    for (; j < j_end; j++) {
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
// Q4_0 GEMM — AVX2 (L2 cache-aware tiling for prompt processing)
// ============================================================================
// Tiles the output columns in groups that fit in L2 cache.
// For each tile: all M activation rows process the same weight tile,
// keeping the weight data hot in L2 instead of re-fetching from RAM.

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
    // Compute tile size: how many output columns fit a weight tile in L2
    // Each output column has K/32 Q4_0 blocks × 18 bytes each = (K/32)*18 bytes
    const int64_t bytes_per_col = (K / QK4_0) * sizeof(block_q4_0);
    int64_t tile_n = S2O_LUT_DEFAULT_L2_TILE_BYTES / bytes_per_col;
    if (tile_n < 4) tile_n = 4;  // minimum tile width
    // Align tile to 4-wide column processing
    tile_n = (tile_n / 4) * 4;

    const int64_t N = j_end - j_start;

    if (M <= 1 || N <= tile_n) {
        // Small enough to skip tiling overhead
        for (int64_t i = 0; i < M; i++) {
            s2o_lut_gemv_q4_0_avx2(
                dst + i * dst_stride,
                src_act + i * act_stride,
                src_wt,
                K, j_start, j_end, nb
            );
        }
        return;
    }

    // Tiled: iterate over column tiles, then over activation rows
    for (int64_t j_tile = j_start; j_tile < j_end; j_tile += tile_n) {
        const int64_t j_tile_end = std::min(j_tile + tile_n, j_end);

        for (int64_t i = 0; i < M; i++) {
            s2o_lut_gemv_q4_0_avx2(
                dst + i * dst_stride,
                src_act + i * act_stride,
                src_wt,
                K, j_tile, j_tile_end, nb
            );
        }
    }
}

// ============================================================================
// Q4_0 GEMV — AVX2, packed (column-interleaved) layout
// ============================================================================
// Same 4-wide algorithm, but weight blocks for each 4-column group are
// contiguous in memory: group_base[b*4 + c] for block b, column c in [0,3].
// This reduces cache line fetches from 4 scattered to ~2 adjacent per iteration.

static void s2o_lut_gemv_q4_0_packed_avx2(
    float       * dst,
    const float * src_act,
    const void  * src_wt,
    int64_t       K,
    int64_t       j_start,
    int64_t       j_end,
    size_t        nb
) {
    (void)nb;  // unused for packed layout
    const int64_t nb_k = K / QK4_0;
    const block_q4_0 * wt = (const block_q4_0 *)src_wt;

    // Pre-quantize activations to INT8 (same as standard path)
    int8_t * act_q8 = (int8_t *)alloca(K * sizeof(int8_t));
    float  * act_d  = (float *)alloca(nb_k * sizeof(float));

    for (int64_t b = 0; b < nb_k; b++) {
        act_d[b] = s2o_quantize_block_f32_to_i8(src_act + b * QK4_0, act_q8 + b * QK4_0);
    }

    int64_t j = j_start;

    // Handle unaligned prefix with 1-wide (if j_start not multiple of 4)
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

    // ---- 4-wide packed main loop ----
    for (; j + 3 < j_end; j += 4) {
        const int64_t group = j / 4;
        const block_q4_0 * group_base = wt + group * nb_k * 4;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            // All 4 blocks are contiguous: 4 * 18 = 72 bytes ≈ 2 cache lines
            const block_q4_0 * wr0 = &group_base[b * 4 + 0];
            const block_q4_0 * wr1 = &group_base[b * 4 + 1];
            const block_q4_0 * wr2 = &group_base[b * 4 + 2];
            const block_q4_0 * wr3 = &group_base[b * 4 + 3];

            // Single prefetch covers all 4 next blocks (contiguous)
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

            const __m256i qa = _mm256_loadu_si256((const __m256i *)(act_q8 + b * QK4_0));

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), s2o_mul_sum_i8_pairs_float(qw0, qa), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s2o_mul_sum_i8_pairs_float(qw1, qa), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), s2o_mul_sum_i8_pairs_float(qw2, qa), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), s2o_mul_sum_i8_pairs_float(qw3, qa), acc3);
        }

        dst[j + 0] = s2o_hsum_float_8(acc0);
        dst[j + 1] = s2o_hsum_float_8(acc1);
        dst[j + 2] = s2o_hsum_float_8(acc2);
        dst[j + 3] = s2o_hsum_float_8(acc3);
    }

    // ---- 1-wide remainder ----
    for (; j < j_end; j++) {
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
// Q4_0 GEMM — AVX2, packed layout (L2 cache-aware tiling)
// ============================================================================

static void s2o_lut_gemm_q4_0_packed_avx2(
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
            s2o_lut_gemv_q4_0_packed_avx2(
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
            s2o_lut_gemv_q4_0_packed_avx2(
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

const s2o_lut_kernels s2o_lut_kernels_avx2 = {
    /* .name              = */ "avx2_4wide",
    /* .gemv_q4_0         = */ s2o_lut_gemv_q4_0_avx2,
    /* .gemm_q4_0         = */ s2o_lut_gemm_q4_0_avx2,
    /* .gemv_q4_0_packed  = */ s2o_lut_gemv_q4_0_packed_avx2,
    /* .gemm_q4_0_packed  = */ s2o_lut_gemm_q4_0_packed_avx2,
};

#endif // defined(__AVX2__)
