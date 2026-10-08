from __future__ import annotations

import torch


def _cuda():
    # imported lazily: the backend package imports this module
    from .backends import cuda as _cuda_backend

    return _cuda_backend


def is_available() -> bool:
    """Also allocates the current device's ring state: call it before capturing CUDA graphs of
    the consumer kernels (int8 GEMVs take the ring pointer as a launch argument)."""
    cuda = _cuda()
    ext = cuda._C if cuda._EXT_AVAILABLE else None
    return bool(ext is not None and hasattr(ext, "prefetch_ring_available") and ext.prefetch_ring_available())


# `credits` bits: which non-weight consumers credit the ring for the bytes they read
CREDIT_KV = 1      # flash_attention_decode_gqa credits the K/V rows it attends
CREDIT_DELTA = 2   # gated_delta_decode_deferred credits its recurrent state

# region flag: nothing credits the region's bytes (read by a plain torch kernel, say); the issuer
# credits them itself as it passes the region
SELF_CREDIT = 1


def configure(regions: torch.Tensor, count: int, lookahead_bytes: int, min_lead_bytes: int = 0, chunk_bytes: int = 96 * 1024, credits: int = 0) -> None:
    """regions: [capacity, 3] uint64 (base, bytes, flags) in read order; the byte counts may be
    rewritten on the same stream between steps (start() re-sums them). Chunks within
    min_lead_bytes of the consumed position are left to demand instead of requested late."""
    cuda = _cuda()
    stream = torch.cuda.current_stream(regions.device).cuda_stream
    cuda._C.prefetch_ring_configure(cuda._wrap_for_dlpack(regions), count, lookahead_bytes, min_lead_bytes, chunk_bytes, credits, stream)


def disable(device: torch.device | int | None = None) -> None:
    cuda = _cuda()
    if not cuda._EXT_AVAILABLE or not hasattr(cuda._C, "prefetch_ring_disable"):
        return
    stream = torch.cuda.current_stream(device).cuda_stream
    cuda._C.prefetch_ring_disable(stream)


def start(device: torch.device | int | None = None) -> None:
    """Launch the issuer for one decode step; it runs alongside the step's kernels on a side stream."""
    stream = torch.cuda.current_stream(device).cuda_stream
    _cuda()._C.prefetch_ring_start(stream)


def counters() -> tuple[int, int, int]:
    """(total, consumed, stalled) of the current device's ring: the last step's region bytes,
    the bytes credited so far, and the issuer CTAs that gave up waiting for credit since
    configure(). Synchronizes the device."""
    torch.cuda.synchronize()
    return _cuda()._C.prefetch_ring_counters()
