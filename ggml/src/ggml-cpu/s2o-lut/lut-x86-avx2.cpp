// S2O LUT-based INT4 GEMV/GEMM — AVX2 implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Strategy for Q4_0 dot product using AVX2:
//
// Q4_0 stores 32 INT4 weights per block as 16 bytes (nibble pairs) + fp16 scale.
// Dequantized value: w = d * (q - 8), so:
//
//   dot = d * [ sum_i(q_i * x_i) - 8 * sum_i(x_i) ]
//
// We convert activations to INT8, then use VPMADDUBSW (unsigned q * signed x_int8)
// to compute q_i * x_i in INT16, followed by VPMADDWD to accumulate to INT32.
//
// Alternatively, we use the split-accumulate approach:
//   1. Load 16 bytes of nibble pairs
//   2. Unpack to 32 bytes of INT8 (values 0-15)
//   3. Subtract 8 to center → signed INT8 (-8..+7)
//   4. Multiply with quantized activations via VPMADDUBSW/VPMADDWD
//
// For simplicity and correctness, we use FP32 SIMD for the activation side:
//   - For each pair of elements, load the nibble, dequant to float, multiply, accumulate.
//   - This uses _mm256_fmadd_ps for fused multiply-add.

#if defined(__AVX2__)

#include "lut-common.h"
#include <immintrin.h>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// Q4_0 GEMV — AVX2 (fully vectorized)
// ============================================================================
//
// Process 8 floats at a time using AVX2 FMA.
// For each Q4_0 block of 32 elements:
//   - Extract 32 nibbles into four groups of 8 INT32 values
//   - Subtract 8, convert to FP32
//   - FMA with activation vector
//   - Scale by d and accumulate

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
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    const __m256i bias_8  = _mm256_set1_epi8(8);
    const __m256i ones_16 = _mm256_set1_epi16(1);

    for (int64_t j = j_start; j < j_end; j++) {
        const block_q4_0 * wt_row = (const block_q4_0 *)((const char *)src_wt + j * nb);

        // Accumulate dot product across all blocks for this output column
        __m256 sum_f32 = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const block_q4_0 * blk = &wt_row[b];
            const float d = GGML_FP16_TO_FP32(blk->d);
            const float * x = src_act + b * QK4_0;

            // ---- Integer dot product approach ----
            // Load 16 bytes of packed nibbles into low 128 bits
            __m128i qbytes128 = _mm_loadu_si128((const __m128i *)blk->qs);

            // Expand to 256 bits: [qbytes | qbytes]
            // Low 128 bits: original bytes (we extract low nibbles)
            // High 128 bits: same bytes shifted right (we extract high nibbles)
            __m256i qbytes = _mm256_set_m128i(qbytes128, qbytes128);

            // Extract low and high nibbles
            __m256i q_lo = _mm256_and_si256(qbytes, lo_mask);                                // low nibbles [0..15] in low 128
            __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qbytes, 4), lo_mask);          // high nibbles in high 128

            // Interleave: want sequential q[0],q[1],q[2],... from [lo_0,lo_1,...,lo_15 | hi_0,hi_1,...,hi_15]
            // lo has: q[0], q[2], q[4], ..., q[30] in bytes 0-15 (repeated in 16-31)
            // hi has: q[1], q[3], q[5], ..., q[31] in bytes 16-31
            // Use unpacklo/unpackhi to interleave them
            __m256i q_interleaved_lo = _mm256_unpacklo_epi8(q_lo, q_hi); // q[0],q[1],q[2],q[3],...
            __m256i q_interleaved_hi = _mm256_unpackhi_epi8(q_lo, q_hi);

            // Now q_interleaved_lo has elements 0-15 (in 128-bit lanes, interleaved),
            // q_interleaved_hi has elements 16-31
            // Due to AVX2 lane semantics, we need to be more careful.
            // Let's use the simpler FP32 approach that's still vectorized:

            // Load 32 activations as 4 groups of 8 FP32
            __m256 vx0 = _mm256_loadu_ps(x);       // x[0..7]
            __m256 vx1 = _mm256_loadu_ps(x + 8);   // x[8..15]
            __m256 vx2 = _mm256_loadu_ps(x + 16);  // x[16..23]
            __m256 vx3 = _mm256_loadu_ps(x + 24);  // x[24..31]

            // Extract nibbles to INT32 and convert to FP32, 8 at a time
            // Group 0: elements 0-7 (bytes 0-3, low and high nibbles interleaved)
            const uint8_t * qs = blk->qs;

            // Build 8 dequantized weights for elements 0-7
            __m256 vw0 = _mm256_set_ps(
                (float)((int)(qs[3] >> 4) - 8),   // q[7]
                (float)((int)(qs[3] & 0xF) - 8),  // q[6]
                (float)((int)(qs[2] >> 4) - 8),   // q[5]
                (float)((int)(qs[2] & 0xF) - 8),  // q[4]
                (float)((int)(qs[1] >> 4) - 8),   // q[3]
                (float)((int)(qs[1] & 0xF) - 8),  // q[2]
                (float)((int)(qs[0] >> 4) - 8),   // q[1]
                (float)((int)(qs[0] & 0xF) - 8)   // q[0]
            );

            // Group 1: elements 8-15 (bytes 4-7)
            __m256 vw1 = _mm256_set_ps(
                (float)((int)(qs[7] >> 4) - 8),
                (float)((int)(qs[7] & 0xF) - 8),
                (float)((int)(qs[6] >> 4) - 8),
                (float)((int)(qs[6] & 0xF) - 8),
                (float)((int)(qs[5] >> 4) - 8),
                (float)((int)(qs[5] & 0xF) - 8),
                (float)((int)(qs[4] >> 4) - 8),
                (float)((int)(qs[4] & 0xF) - 8)
            );

            // Group 2: elements 16-23 (bytes 8-11)
            __m256 vw2 = _mm256_set_ps(
                (float)((int)(qs[11] >> 4) - 8),
                (float)((int)(qs[11] & 0xF) - 8),
                (float)((int)(qs[10] >> 4) - 8),
                (float)((int)(qs[10] & 0xF) - 8),
                (float)((int)(qs[9] >> 4) - 8),
                (float)((int)(qs[9] & 0xF) - 8),
                (float)((int)(qs[8] >> 4) - 8),
                (float)((int)(qs[8] & 0xF) - 8)
            );

            // Group 3: elements 24-31 (bytes 12-15)
            __m256 vw3 = _mm256_set_ps(
                (float)((int)(qs[15] >> 4) - 8),
                (float)((int)(qs[15] & 0xF) - 8),
                (float)((int)(qs[14] >> 4) - 8),
                (float)((int)(qs[14] & 0xF) - 8),
                (float)((int)(qs[13] >> 4) - 8),
                (float)((int)(qs[13] & 0xF) - 8),
                (float)((int)(qs[12] >> 4) - 8),
                (float)((int)(qs[12] & 0xF) - 8)
            );

            // FMA: accumulate w * x for all 32 elements
            __m256 dot = _mm256_mul_ps(vw0, vx0);
            dot = _mm256_fmadd_ps(vw1, vx1, dot);
            dot = _mm256_fmadd_ps(vw2, vx2, dot);
            dot = _mm256_fmadd_ps(vw3, vx3, dot);

            // Horizontal sum of dot (8 floats → 1 float)
            // hadd pairs: [a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7]
            __m128 hi128 = _mm256_extractf128_ps(dot, 1);
            __m128 lo128 = _mm256_castps256_ps128(dot);
            __m128 sum128 = _mm_add_ps(lo128, hi128);           // 4 floats
            __m128 shuf = _mm_movehdup_ps(sum128);               // [1,1,3,3]
            __m128 sums = _mm_add_ps(sum128, shuf);              // [0+1, -, 2+3, -]
            shuf = _mm_movehl_ps(shuf, sums);                    // [2+3, -, -, -]
            sums = _mm_add_ss(sums, shuf);                       // [0+1+2+3]

            float block_dot = _mm_cvtss_f32(sums) * d;

            // Accumulate across blocks
            sum_f32 = _mm256_add_ps(sum_f32, _mm256_set1_ps(block_dot));
        }

        dst[j] = _mm256_cvtss_f32(sum_f32);
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
