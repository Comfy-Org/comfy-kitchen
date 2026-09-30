#pragma once

#include <torch/csrc/stable/tensor.h>

#include <tuple>
#include <vector>

using torch::stable::Tensor;

// gfx103x (RDNA2) int8 attention kernels 的 host 分发入口 (由 pybind 映射到 op)
//
// mask_mode / mask_dtype / mask select the attention mask representation; see
// sageattn_gfx10::Gfx10Mask in mma_gfx10.h. mask_mode 0 means "no mask" and mask
// is then ignored. The prepared modes read the same buffers the main HIP
// backend's sage_prepare_key_mask / sage_prepare_dense_mask produce, so a mask
// snapshot is portable between the two backends.
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

// A std::tuple, not a std::vector: this is declared as `-> (Tensor, Tensor,
// Tensor, Tensor)` rather than `-> Tensor[]` because the stable ABI's TensorList
// return is broken in the torch build this port targets -- reading a returned
// list back raises "vector too long", even for an empty one, so the failure is
// independent of this code (the list is built and read entirely inside
// torch_cpu.dll, through torch_new_list_reserve_size/torch_list_push_back).
// A fixed-arity tuple boxes through the same per-element conversion and works.
// Python-side unpacking is identical, so callers are unaffected.
std::tuple<Tensor, Tensor, Tensor, Tensor> quant_qk_int8_gfx103x(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale,
    int64_t skip_q);

Tensor mean_seq_gfx103x(Tensor input, int64_t tensor_layout);
