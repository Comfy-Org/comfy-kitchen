# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activation + concat + per-tensor FP8 quantization: eager reference.

The sequence a single-stream (parallel attention / MLP) DiT block runs in front
of its output projection when that projection is an FP8 linear with a static
input scale, e.g. Flux's ``SingleStreamBlock``::

    h = act(mlp)                                   # mlp: the MLP half of linear1
    q = quantize_per_tensor_fp8(cat((attn, h), -1), input_scale)
"""

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry

from .quantization import DTYPE_CODE_TO_DTYPE, DTYPE_TO_CODE

# act -> number of h columns per activated column
ACT_CAT_ACTS: dict[str, int] = {
    "swiglu": 2,  # h = [gate | up]: silu(gate) * up
    "swiglu_hidden_gate": 2,  # h = [hidden | gate]: silu(gate) * hidden
    "gelu_tanh": 1,  # gelu(h, approximate="tanh")
}


def _unsupported(act) -> str:
    return f"unsupported act: {act!r} (expected one of {sorted(ACT_CAT_ACTS)})"


def act_cat_width(act: str, h_cols: int) -> int:
    """Width of act(h) for an h row of h_cols columns."""
    if act not in ACT_CAT_ACTS:
        raise ValueError(_unsupported(act))
    return h_cols // ACT_CAT_ACTS[act]


def apply_act(h: Tensor, act: str) -> Tensor:
    """act(h) with the same ops (and roundings) as the unfused model code."""
    if act == "swiglu":
        gate, up = h.chunk(2, dim=-1)
        return functional.silu(gate) * up
    if act == "swiglu_hidden_gate":
        hidden, gate = h.chunk(2, dim=-1)
        return functional.silu(gate) * hidden
    if act == "gelu_tanh":
        return functional.gelu(h, approximate="tanh")
    raise ValueError(_unsupported(act))


def act_cat_quantize_fp8(
    x: Tensor,
    h: Tensor,
    scale: Tensor,
    act: str = "swiglu",
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> Tensor:
    """Reference: ``quantize_per_tensor_fp8(cat((x, act(h)), -1), scale, output_type)``.

    The quantization goes through the registry (``comfy_kitchen::quantize_fp8``), so
    this fallback returns exactly what the unfused call sequence returns on the
    same machine and backend selection.
    """
    if h.shape[-1] % ACT_CAT_ACTS.get(act, 1):
        raise ValueError(f"act {act!r} needs an even number of h columns, got {h.shape[-1]}")
    cat = torch.cat((x, apply_act(h, act)), dim=-1)
    return torch.ops.comfy_kitchen.quantize_fp8(cat, scale, DTYPE_TO_CODE[output_type])


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::act_cat_quantize_fp8", mutates_args=())
def _op_act_cat_quantize_fp8(
    x: torch.Tensor,
    h: torch.Tensor,
    scale: torch.Tensor,
    act: str,
    output_dtype_code: int,
) -> torch.Tensor:
    output_type = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {"x": x, "h": h, "scale": scale, "act": act, "output_type": output_type}
    impl = registry.get_implementation("act_cat_quantize_fp8", kwargs=kwargs)
    return impl(**kwargs)


@_op_act_cat_quantize_fp8.register_fake
def _op_act_cat_quantize_fp8_fake(x, h, scale, act, output_dtype_code):
    width = x.shape[-1] + act_cat_width(act, h.shape[-1])
    return x.new_empty((*x.shape[:-1], width), dtype=DTYPE_CODE_TO_DTYPE[output_dtype_code])
