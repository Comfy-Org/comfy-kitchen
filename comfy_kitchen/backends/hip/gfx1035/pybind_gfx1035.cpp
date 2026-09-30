// SPDX-FileCopyrightText: Copyright (c) 2024 SageAttention team.
// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Module init and op registration for the gfx1035 (RDNA2) SageAttention port.
// Kernel implementations live in attn_gfx103x.hip, ported from
// SageAttention's csrc/qattn/attn_gfx103x.cu.
//
// The ops sit in their own namespace rather than sageattention_qattn_gfx103x so
// that importing this module cannot be confused with (or shadowed by) an
// independently installed SageAttention build.

#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "attn_gfx103x.h"

PyMODINIT_FUNC PyInit__qattn_gfx1035(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_gfx1035",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

STABLE_TORCH_LIBRARY(comfy_kitchen_qattn_gfx1035, m) {
    m.def("qk_int8_sv_bf16_attn_t("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, "
            "int is_causal, float sm_scale, Tensor q_fp, int mask_mode, int mask_dtype, "
            "Tensor mask"
          ") -> Tensor");
    m.def("quant_qk_int8("
            "Tensor query, Tensor key, Tensor key_mean, int tensor_layout, "
            "float sm_scale, int skip_q"
          ") -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("mean_seq(Tensor input, int tensor_layout) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(comfy_kitchen_qattn_gfx1035, CUDA, m) {
    m.impl("qk_int8_sv_bf16_attn_t", TORCH_BOX(qk_int8_sv_bf16_attn_gfx103x_t));
    m.impl("quant_qk_int8", TORCH_BOX(quant_qk_int8_gfx103x));
    m.impl("mean_seq", TORCH_BOX(mean_seq_gfx103x));
}
