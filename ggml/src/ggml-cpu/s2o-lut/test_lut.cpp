// S2O LUT kernel correctness test
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Generates random Q4_0 quantized weights and FP32 activations,
// computes the dot product using both the reference (scalar) path
// and the LUT kernel, and asserts the results match within tolerance.
//
// The LUT kernels quantize activations to INT8 on the fly, so there is
// an expected small error vs the FP32 reference. Tolerances are set to
// accommodate this INT8 quantization error.
//
// Build: g++ -std=c++17 -O2 -mavx2 -mfma -I../../include -I.. test_lut.cpp -o test_lut
//        (add -mavx512f -mavx512bw for AVX-512 testing)
// Run:   ./test_lut

#include "lut-common.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

// ============================================================================
// Reference implementation: scalar Q4_0 dot product
// ============================================================================
// Uses ggml's Q4_0 nibble layout:
//   qs[j] low nibble  → element j       (j = 0..15)
//   qs[j] high nibble → element j + 16

static float ref_dot_q4_0(
    const float * src_act,
    const block_q4_0 * wt_row,
    int64_t K
) {
    const int64_t nb_k = K / QK4_0;
    float sum = 0.0f;

    for (int64_t b = 0; b < nb_k; b++) {
        const block_q4_0 * blk = &wt_row[b];
        const float d = GGML_FP16_TO_FP32(blk->d);
        const float * x = src_act + b * QK4_0;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            int q_lo = (blk->qs[j] & 0x0F);
            int q_hi = ((blk->qs[j] >> 4) & 0x0F);
            block_sum += (float)(q_lo - 8) * x[j];
            block_sum += (float)(q_hi - 8) * x[j + 16];
        }
        sum += d * block_sum;
    }
    return sum;
}

// ============================================================================
// Test harness
// ============================================================================

// Quantize a row of floats to Q4_0 using ggml's nibble layout:
//   qs[j] = quant(x[j]) | (quant(x[j+16]) << 4)
static void quantize_row_q4_0(const float * src, block_q4_0 * dst, int64_t K) {
    const int64_t nb = K / QK4_0;
    for (int64_t b = 0; b < nb; b++) {
        const float * x = src + b * QK4_0;

        // Find abs max
        float amax = 0.0f;
        for (int i = 0; i < QK4_0; i++) {
            float av = fabsf(x[i]);
            if (av > amax) amax = av;
        }

        // Scale: map [-amax, amax] to [-7, 7] centered at 8
        float d = amax / 7.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        dst[b].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 16; j++) {
            int q_lo = (int)(x[j] * id + 8.5f);
            int q_hi = (int)(x[j + 16] * id + 8.5f);
            q_lo = std::max(0, std::min(15, q_lo));
            q_hi = std::max(0, std::min(15, q_hi));
            dst[b].qs[j] = (uint8_t)(q_lo | (q_hi << 4));
        }
    }
}

static int run_test(const char * kernel_name, const s2o_lut_kernels * kernels,
                    int K, int N, unsigned seed) {
    if (!kernels || !kernels->gemv_q4_0) {
        printf("  [SKIP] %s: kernel not available\n", kernel_name);
        return 0;
    }

    printf("  Testing %s: K=%d, N=%d, seed=%u\n", kernel_name, K, N, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random weights (as floats, then quantize)
    std::vector<float> wt_float(K * N);
    for (auto & v : wt_float) v = dist(rng);

    // Quantize weights to Q4_0
    const int64_t blocks_per_row = K / QK4_0;
    std::vector<block_q4_0> wt_q4(blocks_per_row * N);
    for (int j = 0; j < N; j++) {
        quantize_row_q4_0(wt_float.data() + j * K, wt_q4.data() + j * blocks_per_row, K);
    }

    // Generate random activations
    std::vector<float> act(K);
    for (auto & v : act) v = dist(rng);

    // Reference output (FP32 activations)
    std::vector<float> ref_out(N);
    for (int j = 0; j < N; j++) {
        ref_out[j] = ref_dot_q4_0(act.data(), wt_q4.data() + j * blocks_per_row, K);
    }

    // LUT kernel output (quantizes activations to INT8 internally)
    std::vector<float> lut_out(N, 0.0f);
    const size_t nb = blocks_per_row * sizeof(block_q4_0);
    kernels->gemv_q4_0(lut_out.data(), act.data(), wt_q4.data(), K, 0, N, nb);

    // Compare with tolerance for INT8 activation quantization error.
    // The kernels quantize activations to INT8 on the fly, introducing
    // ~1/127 relative error per element. Accumulated over a block, this
    // typically results in <5% relative error for the full dot product.
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int errors = 0;

    for (int j = 0; j < N; j++) {
        float abs_err = fabsf(lut_out[j] - ref_out[j]);
        float rel_err = (fabsf(ref_out[j]) > 1e-6f) ? abs_err / fabsf(ref_out[j]) : abs_err;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        // Error if BOTH absolute and relative tolerance exceeded
        if (abs_err > 2.0f && rel_err > 0.10f) {
            if (errors < 5) {
                printf("    MISMATCH j=%d: ref=%.6f lut=%.6f abs_err=%.2e rel_err=%.2e\n",
                       j, ref_out[j], lut_out[j], abs_err, rel_err);
            }
            errors++;
        }
    }

    printf("    max_abs_err=%.2e  max_rel_err=%.2e  errors=%d/%d\n",
           max_abs_err, max_rel_err, errors, N);

    if (errors > 0) {
        printf("  [FAIL] %s\n", kernel_name);
        return 1;
    }
    printf("  [PASS] %s\n", kernel_name);
    return 0;
}

static int run_gemm_test(const char * kernel_name, const s2o_lut_kernels * kernels,
                         int M, int K, int N, unsigned seed) {
    if (!kernels || !kernels->gemm_q4_0) {
        printf("  [SKIP] %s GEMM: kernel not available\n", kernel_name);
        return 0;
    }

    printf("  Testing %s GEMM: M=%d, K=%d, N=%d, seed=%u\n", kernel_name, M, K, N, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int64_t blocks_per_row = K / QK4_0;
    std::vector<float> wt_float(K * N);
    for (auto & v : wt_float) v = dist(rng);

    std::vector<block_q4_0> wt_q4(blocks_per_row * N);
    for (int j = 0; j < N; j++) {
        quantize_row_q4_0(wt_float.data() + j * K, wt_q4.data() + j * blocks_per_row, K);
    }

    std::vector<float> act(M * K);
    for (auto & v : act) v = dist(rng);

    // Reference: per-row GEMV
    std::vector<float> ref_out(M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            ref_out[i * N + j] = ref_dot_q4_0(act.data() + i * K,
                                                wt_q4.data() + j * blocks_per_row, K);
        }
    }

    // LUT GEMM
    std::vector<float> lut_out(M * N, 0.0f);
    const size_t nb = blocks_per_row * sizeof(block_q4_0);
    kernels->gemm_q4_0(lut_out.data(), act.data(), wt_q4.data(),
                        M, K, 0, N, nb, N, K);

    // Compare with INT8 quantization tolerance
    float max_abs_err = 0.0f;
    int errors = 0;
    for (int idx = 0; idx < M * N; idx++) {
        float abs_err = fabsf(lut_out[idx] - ref_out[idx]);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        float rel_err = (fabsf(ref_out[idx]) > 1e-6f) ? abs_err / fabsf(ref_out[idx]) : abs_err;
        if (abs_err > 2.0f && rel_err > 0.10f) errors++;
    }

    printf("    max_abs_err=%.2e  errors=%d/%d\n", max_abs_err, errors, M * N);

    if (errors > 0) {
        printf("  [FAIL] %s GEMM\n", kernel_name);
        return 1;
    }
    printf("  [PASS] %s GEMM\n", kernel_name);
    return 0;
}

// ============================================================================
// Repack roundtrip test: pack → unpack → compare
// ============================================================================

static int run_repack_roundtrip_test(int K, int N, unsigned seed) {
    printf("  Testing repack roundtrip: K=%d, N=%d, seed=%u\n", K, N, seed);

    std::mt19937 rng(seed);

    const int64_t blocks_per_row = K / QK4_0;
    std::vector<block_q4_0> original(blocks_per_row * N);

    // Fill with random block data
    for (auto & blk : original) {
        blk.d = GGML_FP32_TO_FP16(0.1f * (float)(rng() % 100) / 100.0f);
        for (int i = 0; i < 16; i++) {
            blk.qs[i] = (uint8_t)(rng() & 0xFF);
        }
    }

    // Pack
    std::vector<block_q4_0> packed(blocks_per_row * N);
    s2o_repack_q4_0(packed.data(), original.data(), N, K);

    // Unpack
    std::vector<block_q4_0> unpacked(blocks_per_row * N);
    s2o_unpack_q4_0(unpacked.data(), packed.data(), N, K);

    // Compare element-by-element
    int errors = 0;
    for (int64_t i = 0; i < blocks_per_row * N; i++) {
        if (original[i].d != unpacked[i].d ||
            memcmp(original[i].qs, unpacked[i].qs, 16) != 0) {
            if (errors < 3) {
                printf("    MISMATCH at block %d\n", (int)i);
            }
            errors++;
        }
    }

    if (errors > 0) {
        printf("  [FAIL] repack roundtrip: %d mismatches\n", errors);
        return 1;
    }
    printf("  [PASS] repack roundtrip\n");
    return 0;
}

// ============================================================================
// Packed GEMV correctness test
// ============================================================================

static int run_packed_test(const char * kernel_name, const s2o_lut_kernels * kernels,
                           int K, int N, unsigned seed) {
    if (!kernels || !kernels->gemv_q4_0_packed) {
        printf("  [SKIP] %s packed: kernel not available\n", kernel_name);
        return 0;
    }

    if (N % 4 != 0) {
        printf("  [SKIP] %s packed: N=%d not multiple of 4\n", kernel_name, N);
        return 0;
    }

    printf("  Testing %s packed GEMV: K=%d, N=%d, seed=%u\n", kernel_name, K, N, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> wt_float(K * N);
    for (auto & v : wt_float) v = dist(rng);

    const int64_t blocks_per_row = K / QK4_0;
    std::vector<block_q4_0> wt_q4(blocks_per_row * N);
    for (int j = 0; j < N; j++) {
        quantize_row_q4_0(wt_float.data() + j * K, wt_q4.data() + j * blocks_per_row, K);
    }

    // Repack to column-interleaved layout
    std::vector<block_q4_0> wt_packed(blocks_per_row * N);
    s2o_repack_q4_0(wt_packed.data(), wt_q4.data(), N, K);

    std::vector<float> act(K);
    for (auto & v : act) v = dist(rng);

    // Reference output (standard layout, FP32 activations)
    std::vector<float> ref_out(N);
    for (int j = 0; j < N; j++) {
        ref_out[j] = ref_dot_q4_0(act.data(), wt_q4.data() + j * blocks_per_row, K);
    }

    // Packed kernel output
    std::vector<float> lut_out(N, 0.0f);
    kernels->gemv_q4_0_packed(lut_out.data(), act.data(), wt_packed.data(), K, 0, N, 0);

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int errors = 0;

    for (int j = 0; j < N; j++) {
        float abs_err = fabsf(lut_out[j] - ref_out[j]);
        float rel_err = (fabsf(ref_out[j]) > 1e-6f) ? abs_err / fabsf(ref_out[j]) : abs_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > 2.0f && rel_err > 0.10f) {
            if (errors < 5) {
                printf("    MISMATCH j=%d: ref=%.6f lut=%.6f abs_err=%.2e rel_err=%.2e\n",
                       j, ref_out[j], lut_out[j], abs_err, rel_err);
            }
            errors++;
        }
    }

    printf("    max_abs_err=%.2e  max_rel_err=%.2e  errors=%d/%d\n",
           max_abs_err, max_rel_err, errors, N);

    if (errors > 0) {
        printf("  [FAIL] %s packed GEMV\n", kernel_name);
        return 1;
    }
    printf("  [PASS] %s packed GEMV\n", kernel_name);
    return 0;
}

int main() {
    printf("S2O LUT Kernel Correctness Tests\n");
    printf("================================\n\n");

    int failures = 0;

    // Test configurations: (K, N)
    struct { int K; int N; } configs[] = {
        {  32,   1 },   // minimal: single block, single column
        {  32,  16 },   // single block, multiple columns
        { 128,  32 },   // 4 blocks
        { 256,  64 },   // typical small layer
        { 512, 128 },   // medium
        {1024, 256 },   // large
        {4096, 512 },   // realistic model dimension
    };

    // Packed configs (N must be multiple of 4)
    struct { int K; int N; } packed_configs[] = {
        {  32,  16 },
        { 256,  64 },
        {1024, 256 },
        {4096, 512 },
    };

#if defined(__AVX2__)
    printf("AVX2 GEMV tests:\n");
    for (const auto & cfg : configs) {
        failures += run_test("avx2", &s2o_lut_kernels_avx2, cfg.K, cfg.N, 42);
    }

    printf("\nAVX2 GEMM tests:\n");
    failures += run_gemm_test("avx2", &s2o_lut_kernels_avx2, 1, 256, 64, 42);
    failures += run_gemm_test("avx2", &s2o_lut_kernels_avx2, 4, 512, 128, 42);
    failures += run_gemm_test("avx2", &s2o_lut_kernels_avx2, 16, 1024, 256, 42);

    printf("\nRepack roundtrip tests:\n");
    failures += run_repack_roundtrip_test(256, 64, 42);
    failures += run_repack_roundtrip_test(4096, 512, 99);

    printf("\nAVX2 packed GEMV tests:\n");
    for (const auto & cfg : packed_configs) {
        failures += run_packed_test("avx2", &s2o_lut_kernels_avx2, cfg.K, cfg.N, 42);
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    printf("\nAVX-512 GEMV tests:\n");
    for (const auto & cfg : configs) {
        failures += run_test("avx512", &s2o_lut_kernels_avx512, cfg.K, cfg.N, 42);
    }

    printf("\nAVX-512 GEMM tests:\n");
    failures += run_gemm_test("avx512", &s2o_lut_kernels_avx512, 1, 256, 64, 42);
    failures += run_gemm_test("avx512", &s2o_lut_kernels_avx512, 4, 512, 128, 42);
    failures += run_gemm_test("avx512", &s2o_lut_kernels_avx512, 16, 1024, 256, 42);

    printf("\nAVX-512 packed GEMV tests:\n");
    for (const auto & cfg : packed_configs) {
        failures += run_packed_test("avx512", &s2o_lut_kernels_avx512, cfg.K, cfg.N, 42);
    }
#endif

    printf("\n================================\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
