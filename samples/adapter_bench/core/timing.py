"""Timing primitives: wall, host issue and device time, with an L2 flush.

wall: median of event-timed single calls, each after an L2 flush and a sync,
so it is device time plus host dispatch, from cold L2.
host: time to issue the call with no sync inside the loop.
device: on CUDA the call captured as a graph and replayed behind an L2 flush,
so it is device time from cold L2 with no host issue inside the window; on XPU
the call itself behind the flush.
"""

import statistics
import time

import torch

from .device import API, KIND, l2_bytes

_FLUSH: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def flush_l2() -> None:
    """Write random data over a buffer twice the card's L2, evicting the previous call's operands."""
    index = API.current_device()
    bufs = _FLUSH.get(index)
    if bufs is None:
        src = torch.randint(-(2**31), 2**31 - 1, (2 * l2_bytes() // 4,), device=KIND)
        bufs = (src.to(torch.int32), torch.empty_like(src, dtype=torch.int32))
        _FLUSH[index] = bufs
    torch.bitwise_not(bufs[0], out=bufs[1])


def wall_ms(fn, warmup: int = 3, iters: int = 8) -> float:
    """Median event-timed call, L2 flushed before each: device plus host issue."""
    for _ in range(warmup):
        fn()
    API.synchronize()
    beg = [API.Event(enable_timing=True) for _ in range(iters)]
    end = [API.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush_l2()
        API.synchronize()
        beg[i].record()
        fn()
        end[i].record()
    API.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(beg, end, strict=True))


def _capture(fn):
    """fn captured as one CUDA graph, or None where it cannot be (XPU, a host sync inside)."""
    if KIND != "cuda":
        return None
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph
    except Exception:
        torch.cuda.synchronize()
        return None


def device_ms(fn, warmup: int = 3, iters: int = 10) -> tuple[float, bool]:
    """(median device ms, whether it was graph-replayed), L2 flushed before every call.

    The call is captured once and replayed: the flush is enqueued first and is
    still running when the start event and the replay are enqueued behind it,
    so no host issue time lands between the events. An arm that cannot be
    captured is timed the same way unreplayed; its host issue can then fall
    inside the window, and the row says so.
    """
    for _ in range(warmup):
        fn()
    API.synchronize()
    graph = _capture(fn)
    run = graph.replay if graph is not None else fn
    run()
    API.synchronize()
    beg = [API.Event(enable_timing=True) for _ in range(iters)]
    end = [API.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush_l2()
        beg[i].record()
        run()
        end[i].record()
    API.synchronize()
    ms = statistics.median(s.elapsed_time(e) for s, e in zip(beg, end, strict=True))
    del graph
    return ms, run is not fn


def host_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    """Host time to issue one call, no device sync inside the loop."""
    for _ in range(warmup):
        fn()
    API.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) * 1e3 / iters
    API.synchronize()
    return elapsed


def interleave(arms: dict, rounds: int = 2, warmup: int = 3, iters: int = 8) -> dict:
    """{arm: (wall, host, device, graphed)}; arms run round-robin, each keeps its
    best round for wall and device, a round being a median."""
    wall = {name: float("inf") for name in arms}
    dev = {name: float("inf") for name in arms}
    graphed = {}
    for _ in range(rounds):
        for name, fn in arms.items():
            wall[name] = min(wall[name], wall_ms(fn, warmup, iters))
            ms, graphed[name] = device_ms(fn, warmup=1, iters=iters)
            dev[name] = min(dev[name], ms)
    return {
        name: (wall[name], host_ms(fn), dev[name], graphed[name])
        for name, fn in arms.items()
    }


def peak_extra_bytes(fn) -> int:
    """Peak memory one call allocates above what was resident before it."""
    flush_l2()
    API.synchronize()
    before = API.memory_allocated()
    API.reset_peak_memory_stats()
    fn()
    API.synchronize()
    return API.max_memory_allocated() - before
