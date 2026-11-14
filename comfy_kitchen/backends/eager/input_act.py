# SPDX-License-Identifier: Apache-2.0
"""The pre-quantization input activations as a standalone, dispatched op.

Quantized linears fuse these into their quantizer. Unquantized linears still
need ``act(x)`` on its own, and on some devices a backend can do that faster
than the eager chain of ops, so it is dispatched like any other op.
"""
import torch

from comfy_kitchen.backends._activations import apply_input_act, input_act_width
from comfy_kitchen.registry import registry


def input_act(
    x: torch.Tensor,
    input_act: str,
    act_weight: torch.Tensor | None = None,
    act_eps: float = 0.0,
) -> torch.Tensor:
    return apply_input_act(x, input_act, act_weight, act_eps)


@torch.library.custom_op("comfy_kitchen::input_act", mutates_args=())
def _op_input_act(
    x: torch.Tensor,
    input_act: str,
    act_weight: torch.Tensor | None,
    act_eps: float,
) -> torch.Tensor:
    kwargs = {"x": x, "input_act": input_act, "act_weight": act_weight, "act_eps": act_eps}
    impl = registry.get_implementation("input_act", kwargs=kwargs)
    return impl(**kwargs)


@_op_input_act.register_fake
def _op_input_act_fake(x, input_act, act_weight, act_eps):
    return x.new_empty(*x.shape[:-1], x.shape[-1] // input_act_width(input_act))
