#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// L2 prefetch ring for decode weight sweeps.
//
// The host records the byte ranges one decode step reads, in the order the
// consumer kernels read them (a scatter-gather work list; for the streamed W4A8
// GEMM that is chunk order, not address order). Consumers report `consumed`,
// the bytes of that stream whose loads have completed. An issuer kernel on a
// side stream walks the list and requests [consumed, consumed + lookahead)
// into L2 with cp.async.bulk.prefetch.L2 (no data return), so the L2 footprint
// of prefetched-but-unread weights never exceeds `lookahead` and DRAM stays
// busy while the consumer stream runs kernels that do not read weights. It
// exits when the step is consumed. Ranges must be 16-byte aligned in base and
// size. Besides weights the list may carry per-step state the attention
// kernels read (KV cache rows, DeltaNet recurrent state); `credits` says which
// of those kernels credit their reads, so the host only lists what is credited. Counters are updated without synchronization by design: the consumer
// credits with fire-and-forget atomics and the issuer reads snapshots; the
// only consequence of a stale read is a chunk prefetched late or twice.
#ifndef PREFETCH_RING_ISSUERS
#define PREFETCH_RING_ISSUERS 10
#endif

enum : uint32_t {
    PREFETCH_RING_CREDIT_KV = 1u,      // flash decode credits the K/V rows it attends
    PREFETCH_RING_CREDIT_DELTA = 2u,   // gated delta decode credits its recurrent state tile
};

struct PrefetchRegion {
    const unsigned char* base;
    uint64_t bytes;
};

struct PrefetchRingState {
    const PrefetchRegion* regions;
    int count;
    int enabled;
    uint64_t total;       // sum of region bytes (one step)
    uint64_t lookahead;   // bytes kept in flight ahead of `consumed`
    uint32_t chunk;       // bytes per prefetch request
    uint32_t credits;     // PREFETCH_RING_CREDIT_* mask: non-weight consumers that credit their reads
    uint32_t stalled;     // issuer CTAs that gave up waiting for consumption (diagnostic)
    uint64_t consumed;    // bytes whose demand loads completed this step
    // diagnostics, cumulative over steps (read with prefetch_ring_read_stats)
    uint64_t touched;     // bytes the issuer requested
    uint64_t skipped;     // bytes the issuer skipped because demand got there first
    uint64_t waited_ns;   // CTA-nanoseconds spent throttled at consumed + lookahead
    uint32_t smid[PREFETCH_RING_ISSUERS];    // SM each issuer CTA ran on in the last step (placement diagnostic)
    uint32_t arrived;     // issuer CTAs started this step
    uint32_t distinct_hist[PREFETCH_RING_ISSUERS + 1];   // steps by number of distinct SMs the issuer CTAs landed on
    // Lead trace (diagnostic, set with prefetch_ring_set_trace): one record per
    // issue batch / wait exit / skip, restarted every step, so after a step the
    // buffer holds that step's (globaltimer ns, consumed snapshot, issuer cursor,
    // blockIdx | event << 8) history: the ring's lead over demand as a function
    // of stream position and time.
    uint64_t* trace;      // [trace_cap][4], nullptr when off
    uint32_t trace_cap;
    uint32_t trace_n;     // records written this step (may exceed trace_cap: dropped)
};

enum : uint32_t {
    PREFETCH_RING_TRACE_ISSUE = 0,
    PREFETCH_RING_TRACE_WAIT = 1,   // recorded when the issuer leaves the consumed + lookahead throttle
    PREFETCH_RING_TRACE_SKIP = 2,   // recorded after jumping past chunks demand had already read
};

#ifdef __CUDACC__
static __device__ __forceinline__ void prefetch_ring_consume_device(
    PrefetchRingState* ring, uint64_t bytes) {
    if (ring != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(&ring->consumed),
                  static_cast<unsigned long long>(bytes));
}
#endif

bool prefetch_ring_is_available();

extern "C" void launch_prefetch_ring_configure(
    const uint64_t* regions, int count, uint64_t lookahead, uint32_t chunk, uint32_t credits,
    cudaStream_t stream);
extern "C" void launch_prefetch_ring_disable(cudaStream_t stream);
extern "C" void launch_prefetch_ring_start(cudaStream_t stream);
extern "C" void launch_prefetch_ring_set_trace(uint64_t* trace, uint32_t cap, cudaStream_t stream);
extern "C" void set_w4a8_prefetch_ring_state(PrefetchRingState* state);
extern "C" void set_int8_prefetch_ring_state(PrefetchRingState* state);
extern "C" void set_flash_prefetch_ring_state(PrefetchRingState* state);
extern "C" void set_gated_delta_prefetch_ring_state(PrefetchRingState* state);
extern "C" void prefetch_ring_read_stats(PrefetchRingState* host);
