// S2O LUT-based INT4 inference backend — ggml integration
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Follows the AMX/KleidiAI pattern:
//   1. Registers a custom buffer type via ggml_backend_cpu_s2o_lut_buffer_type()
//   2. On tensor init, sets tensor->extra to our tensor_traits
//   3. On set_tensor, repacks weights into LUT-friendly layout (currently passthrough)
//   4. On compute_forward, dispatches MUL_MAT to our LUT kernels
//
// The buffer type is returned in ggml_backend_cpu_get_extra_buffer_types() so
// the CPU backend automatically considers it for ops it supports.

#include "s2o-lut.h"
#include "lut-common.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "traits.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

// Compile when we have AVX2 (x86) or NEON (ARM)
#if defined(__AVX2__) || defined(__ARM_NEON)

// ============================================================================
// Context: selected kernel set
// ============================================================================

static const s2o_lut_kernels * s2o_ctx_kernels = nullptr;
static bool s2o_initialized = false;

static void s2o_lut_init(void) {
    if (s2o_initialized) return;
    s2o_initialized = true;
    s2o_ctx_kernels = s2o_lut_select_kernels();
    if (s2o_ctx_kernels) {
        fprintf(stderr, "s2o-lut: using %s kernels\n", s2o_ctx_kernels->name);
    } else {
        fprintf(stderr, "s2o-lut: no compatible kernels found\n");
    }
}

// ============================================================================
// tensor_traits: compute_forward dispatch
// ============================================================================

namespace ggml::cpu::s2o_lut {

class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        // No extra workspace needed for our dot-product kernels
        GGML_UNUSED(op);
        size = 0;
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        if (!s2o_ctx_kernels || !s2o_ctx_kernels->gemv_q4_0) {
            return false;
        }

        const struct ggml_tensor * src0 = op->src[0]; // weights (quantized)
        const struct ggml_tensor * src1 = op->src[1]; // activations (FP32)

        // src0: [K, N] quantized weights (ne[0]=K, ne[1]=N)
        // src1: [K, M] activations       (ne[0]=K, ne[1]=M)
        // dst:  [N, M] output            (ne[0]=N, ne[1]=M)
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = src1->ne[1];

        // Partition output columns across threads
        const int64_t n_per_thread = (N + params->nth - 1) / params->nth;
        const int64_t j_start = params->ith * n_per_thread;
        const int64_t j_end   = std::min(j_start + n_per_thread, N);

        if (j_start >= j_end) {
            return true;  // nothing to do for this thread
        }

        float       * dst_data = (float *)op->data;
        const float * act_data = (const float *)src1->data;
        const void  * wt_data  = src0->data;
        const size_t  nb       = src0->nb[1];  // stride between weight rows

        if (M == 1) {
            // GEMV path (single token generation)
            s2o_ctx_kernels->gemv_q4_0(
                dst_data,
                act_data,
                wt_data,
                K, j_start, j_end, nb
            );
        } else {
            // GEMM path (prompt processing / batched)
            const int64_t dst_stride = op->ne[0];      // N
            const int64_t act_stride = src1->ne[0];     // K

            s2o_ctx_kernels->gemm_q4_0(
                dst_data,
                act_data,
                wt_data,
                M, K, j_start, j_end, nb,
                dst_stride, act_stride
            );
        }

        return true;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}

}  // namespace ggml::cpu::s2o_lut

// ============================================================================
// Buffer interface
// ============================================================================

static void ggml_backend_s2o_lut_buffer_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_s2o_lut_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)(buffer->context);
}

static enum ggml_status ggml_backend_s2o_lut_buffer_init_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor
) {
    // Set our tensor_traits so compute_forward dispatches to LUT kernels
    tensor->extra = (void *)ggml::cpu::s2o_lut::get_tensor_traits(buffer, tensor);
    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_s2o_lut_buffer_memset_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
    uint8_t value, size_t offset, size_t size
) {
    memset((char *)tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_s2o_lut_buffer_set_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
    const void * data, size_t offset, size_t size
) {
    // Currently passthrough — weights are stored in standard Q4_0 layout.
    // Future: repack into LUT-friendly interleaved nibble layout for
    //         direct VPSHUFB indexing without per-block split.
    memcpy((char *)tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_s2o_lut_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_s2o_lut_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_s2o_lut_buffer_free,
    /* .get_base        = */ ggml_backend_s2o_lut_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_s2o_lut_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_s2o_lut_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_s2o_lut_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_s2o_lut_buffer_clear,
    /* .reset           = */ nullptr,
};

// ============================================================================
// Buffer type interface
// ============================================================================

static const char * ggml_backend_s2o_lut_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "S2O_LUT";
}

static ggml_backend_buffer_t ggml_backend_s2o_lut_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size
) {
    void * data = ggml_aligned_malloc(size);
    if (!data) {
        fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, size);
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_s2o_lut_buffer_interface, data, size);
}

static size_t ggml_backend_s2o_lut_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

// ============================================================================
// extra_buffer_type: op support check + traits dispatch
// ============================================================================

namespace ggml::cpu::s2o_lut {

class extra_buffer_type : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        const auto * src0 = op->src[0];
        const auto * src1 = op->src[1];

        // Weights must be in our buffer
        if (!src0->buffer || src0->buffer->buft != ggml_backend_cpu_s2o_lut_buffer_type()) {
            return false;
        }

        // Only support Q4_0 weights for now
        if (!s2o_lut_supports_type(src0->type)) {
            return false;
        }

        // Activations must be FP32
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }

        // Both must be contiguous
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            return false;
        }

        // K dimension must be a multiple of QK4_0
        if (src0->ne[0] % QK4_0 != 0) {
            return false;
        }

        return true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_cpu_s2o_lut_buffer_type()) {
            return (ggml::cpu::tensor_traits *)op->src[0]->extra;
        }
        return nullptr;
    }
};

}  // namespace ggml::cpu::s2o_lut

// ============================================================================
// Public API
// ============================================================================

ggml_backend_buffer_type_t ggml_backend_cpu_s2o_lut_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_s2o_lut = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_s2o_lut_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_s2o_lut_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_s2o_lut_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ nullptr,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::s2o_lut::extra_buffer_type(),
    };

    s2o_lut_init();

    if (!s2o_ctx_kernels) {
        return nullptr;
    }

    return &ggml_backend_buffer_type_s2o_lut;
}

#endif // defined(__AVX2__) || defined(__ARM_NEON)
