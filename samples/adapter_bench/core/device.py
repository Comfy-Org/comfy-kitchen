"""The accelerator the benchmark runs on: CUDA when present, else Intel XPU."""

import re

import torch

KIND = "cuda" if torch.cuda.is_available() else "xpu"
# torch.cuda or torch.xpu: synchronize, Event, Stream, memory stats, device properties.
API = getattr(torch, KIND)


def properties():
    return API.get_device_properties(API.current_device())


def arch() -> str:
    """sm_XY on NVIDIA, xpu-<device name> on Intel."""
    p = properties()
    if KIND == "cuda":
        return f"sm_{p.major}{p.minor}"
    return "xpu-" + re.sub(r"[^a-z0-9]+", "-", p.name.lower()).strip("-")


def l2_bytes() -> int:
    p = properties()
    return (
        getattr(p, "L2_cache_size", 0)
        or getattr(p, "last_level_cache_size", 0)
        or (64 << 20)
    )


def units() -> int:
    """SMs or Xe-cores."""
    p = properties()
    return getattr(p, "multi_processor_count", 0) or getattr(p, "gpu_subslice_count", 0)


def capability() -> tuple[int, int] | None:
    return API.get_device_capability() if KIND == "cuda" else None
