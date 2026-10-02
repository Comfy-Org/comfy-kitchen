# Explicit Ascend W4A4 weight reuse

The Ascend-specific `convrot_w4a4_linear` accepts an optional caller-owned cache:

```python
from comfy_kitchen.backends import ascend

# Reserve this memory in addition to the model and activation working set.
with ascend.W4A4WeightCache(max_bytes=4 * 1024**3) as cache:
    for x in inputs:
        output = ascend.convrot_w4a4_linear(
            x, packed_weight, weight_scales, bias, weight_cache=cache
        )
```

The cache is opt-in. The public
`comfy_kitchen.tensor.convrot_w4a4.convrot_w4a4_linear` also accepts
`weight_cache=cache` and preserves normal backend selection; non-Ascend
implementations ignore this optional optimization. No path enables caching
automatically. A framework adapter must explicitly own the scope and pass the
cache. Do not install a process-global cache or retain one across requests.

The budget counts both a packed snapshot and the unpacked INT8 representation:
approximately three times the packed weight size in additional storage. A
budget counts logical tensor bytes, not allocator padding/reserved blocks. A
content comparison validates every hit, including inference tensors and writes
through aliases or `.data`. This introduces an NPU synchronization per hit;
measure end-to-end performance, not just the avoided unpack kernels.

Entries weakly reference source weights. Collected sources release their cached
tensors. Scope exit and exceptions clear all entries. New weights that do not
fit the budget use the original unpack path without evicting live cached
weights. No implicit free-memory heuristic or allocation-failure recovery is
provided: the caller owns the memory reservation.

Cross-thread calls, cross-stream hits and graph capture use uncached unpacking.
The caller still owns normal stream dependencies for source writes. Mutating a
weight concurrently with its use is unsupported, as in the uncached operation.
This cache does not alter checkpoint storage or serialization, and does not
cache activation-dependent scales or outputs.

For framework validation, include weight mutation, unload/reload, memory limits,
model changes and exception cleanup. A successful fixed-model benchmark alone
does not validate the framework integration.
