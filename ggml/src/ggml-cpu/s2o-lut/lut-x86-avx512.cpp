// S2O LUT-based INT4 GEMV/GEMM — AVX-512 implementation
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 enables processing 16 FP32 elements per instruction.
// For Q4_0 blocks of 32 elements, we need only 2 FMA operations per block
// (compared to 4 in AVX2). This doubles throughput for the compute-bound path.
//
// Additionally, AVX-512 VPSHUFB operates on 64 bytes, enabling
// efficient nibble extraction and dequantization.

#if defined(__AVX512F__) && defined(__AVX512BW__)

#include "lut-common.h"
#include <immintrin.h>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// ============================================================================
// Q4_0 GEMV — AVX-512 (16 floats per FMA)
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

    for (int64_t j = j_start; j < j_end; j++) {
        const block_q4_0 * wt_row = (const block_q4_0 *)((const char *)src_wt + j * nb);

        __m512 sum_f32 = _mm512_setzero_ps();

        for (int64_t b = 0; b < nb_k; b++) {
            const block_q4_0 * blk = &wt_row[b];
            const float d = GGML_FP16_TO_FP32(blk->d);
            const float * x = src_act + b * QK4_0;
            const uint8_t * qs = blk->qs;

            // Load 32 activations as 2 groups of 16 FP32
            __m512 vx0 = _mm512_loadu_ps(x);       // x[0..15]
            __m512 vx1 = _mm512_loadu_ps(x + 16);  // x[16..31]

            // Build 16 dequantized weights for elements 0-15
            // Using _mm512_set_ps (fills from high to low index)
            __m512 vw0 = _mm512_set_ps(
                (float)((int)(qs[7] >> 4) - 8),    // q[15]
                (float)((int)(qs[7] & 0xF) - 8),   // q[14]
                (float)((int)(qs[6] >> 4) - 8),    // q[13]
                (float)((int)(qs[6] & 0xF) - 8),   // q[12]
                (float)((int)(qs[5] >> 4) - 8),    // q[11]
                (float)((int)(qs[5] & 0xF) - 8),   // q[10]
                (float)((int)(qs[4] >> 4) - 8),    // q[9]
                (float)((int)(qs[4] & 0xF) - 8),   // q[8]
                (float)((int)(qs[3] >> 4) - 8),    // q[7]
                (float)((int)(qs[3] & 0xF) - 8),   // q[6]
                (float)((int)(qs[2] >> 4) - 8),    // q[5]
                (float)((int)(qs[2] & 0xF) - 8),   // q[4]
                (float)((int)(qs[1] >> 4) - 8),    // q[3]
                (float)((int)(qs[1] & 0xF) - 8),   // q[2]
                (float)((int)(qs[0] >> 4) - 8),    // q[1]
                (float)((int)(qs[0] & 0xF) - 8)    // q[0]
            );

            // Build 16 dequantized weights for elements 16-31
            __m512 vw1 = _mm512_set_ps(
                (float)((int)(qs[15] >> 4) - 8),   // q[31]
                (float)((int)(qs[15] & 0xF) - 8),  // q[30]
                (float)((int)(qs[14] >> 4) - 8),   // q[29]
                (float)((int)(qs[14] & 0xF) - 8),  // q[28]
                (float)((int)(qs[13] >> 4) - 8),   // q[27]
                (float)((int)(qs[13] & 0xF) - 8),  // q[26]
                (float)((int)(qs[12] >> 4) - 8),   // q[25]
                (float)((int)(qs[12] & 0xF) - 8),  // q[24]
                (float)((int)(qs[11] >> 4) - 8),   // q[23]
                (float)((int)(qs[11] & 0xF) - 8),  // q[22]
                (float)((int)(qs[10] >> 4) - 8),   // q[21]
                (float)((int)(qs[10] & 0xF) - 8),  // q[20]
                (float)((int)(qs[9] >> 4) - 8),    // q[19]
                (float)((int)(qs[9] & 0xF) - 8),   // q[18]
                (float)((int)(qs[8] >> 4) - 8),    // q[17]
                (float)((int)(qs[8] & 0xF) - 8)    // q[16]
            );

            // FMA: 2 instructions for 32 elements (vs 4 in AVX2)
            __m512 dot = _mm512_mul_ps(vw0, vx0);
            dot = _mm512_fmadd_ps(vw1, vx1, dot);

            // Horizontal reduce: _mm512_reduce_add_ps
            float block_dot = _mm512_reduce_add_ps(dot) * d;

            sum_f32 = _mm512_add_ps(sum_f32, _mm512_set1_ps(block_dot));
        }

        // Extract scalar from lane 0
        dst[j] = _mm512_cvtss_f32(sum_f32);
    }
}

// ============================================================================
// Q4_0 GEMM — AVX-512 (batched)
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
