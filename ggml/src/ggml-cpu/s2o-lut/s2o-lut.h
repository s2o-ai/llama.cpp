// S2O LUT-based INT4 inference backend
// Copyright 2025-2026 S2O AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_s2o_lut_buffer_type(void);

#ifdef __cplusplus
}
#endif
