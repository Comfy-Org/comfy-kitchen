"""Arm builders and tensor helpers shared by the family modules."""

import torch
import torch._dynamo
from torch._dynamo.utils import counters

import comfy_kitchen as ck

from ..core.case import Arm
from ..core.device import KIND

_GEN = torch.Generator().manual_seed(0)


def rnd(*shape, dtype, scale: float = 1.0) -> torch.Tensor:
    """Drawn on a CPU generator so every device sees the same values."""
    return (torch.randn(*shape, generator=_GEN) * scale).to(KIND, dtype)


def kitchen(name: str, adapt=None) -> Arm:
    """The public API as shipped; adapt maps the case's inputs to its arguments."""
    fn = getattr(ck.adapters, name)
    return Arm(lambda **kw: fn(*adapt(**kw)), note="comfy_kitchen.adapters")


def eager(name: str, adapt=None) -> Arm:
    """Kitchen's torch backend through the same API."""
    fn = getattr(ck.adapters, name)

    def run(**kw):
        with ck.use_backend("eager"):
            return fn(*adapt(**kw))

    return Arm(run, note="comfy_kitchen eager backend")


def comfy(fn, backward: bool = True, note: str = "ComfyUI code") -> Arm:
    return Arm(lambda **kw: fn(**kw), backward=backward, note=note)


def compile_arm(fn, inputs: dict, backward: bool = True) -> Arm:
    """torch.compile of the ComfyUI arm, compiled for this case; refuses a fallback."""
    torch._dynamo.reset()
    cfn = torch.compile(fn, dynamic=None)
    try:
        counters.clear()
        with torch.no_grad():
            cfn(**inputs)
        if counters["stats"]["unique_graphs"] == 0:
            raise RuntimeError(
                "torch.compile produced no graph: it would measure eager"
            )
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"

        def failed(**_):
            raise RuntimeError(err)

        return Arm(failed, backward=False, note="torch.compile(ComfyUI code)")
    return Arm(
        lambda **kw: cfn(**kw), backward=backward, note="torch.compile(ComfyUI code)"
    )
