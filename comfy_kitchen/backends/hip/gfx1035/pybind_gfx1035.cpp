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
#include <torch/csrc/stable/tensor.h>

#include <tuple>

using torch::stable::Tensor;

// The three host entry points, declared here rather than in a header of their own.
// attn_gfx103x.h existed only to satisfy this one #include, and the mainline HIP
// backend declares its entry points the same way -- in dlpack_bindings.cpp, next to
// the code they wrap, with no separate .h in between. There is no second consumer
// of these declarations to share them with: the kernels are reachable only through
// the ops registered below.

Tensor qk_int8_sv_bf16_attn_gfx103x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor q_scale,
    Tensor k_scale,
    Tensor v_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    Tensor q_fp,
    int64_t mask_mode,
    int64_t mask_dtype,
    Tensor mask);

std::tuple<Tensor, Tensor, Tensor, Tensor> quant_qk_int8_gfx103x(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale,
    int64_t skip_q);


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
}

STABLE_TORCH_LIBRARY_IMPL(comfy_kitchen_qattn_gfx1035, CUDA, m) {
    m.impl("qk_int8_sv_bf16_attn_t", TORCH_BOX(qk_int8_sv_bf16_attn_gfx103x_t));
    m.impl("quant_qk_int8", TORCH_BOX(quant_qk_int8_gfx103x));
}
