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
// of those kernels credit their reads, so the host only lists what is credited.
// Small tensors read by kernels that never credit (norm scales, conv taps, gate
// projections) carry PREFETCH_REGION_SELF_CREDIT: the issuer credits them
// itself as its walk passes them, so they ride in the stream without a
// consumer. Counters are updated without synchronization by design: the consumer
// credits with fire-and-forget atomics and the issuer reads snapshots; the
// only consequence of a stale read is a chunk prefetched late or twice.

enum : uint32_t {
    PREFETCH_RING_CREDIT_KV = 1u,      // flash decode credits the K/V rows it attends
    PREFETCH_RING_CREDIT_DELTA = 2u,   // gated delta decode credits its recurrent state tile
};

enum : uint64_t {
    PREFETCH_REGION_SELF_CREDIT = 1ull,   // no consumer credits it: the issuer does when its walk passes the region
};

struct PrefetchRegion {
    const unsigned char* base;
    uint64_t bytes;
    uint64_t flags;   // PREFETCH_REGION_* mask
};

struct PrefetchRingState {
    const PrefetchRegion* regions;
    int count;
    int enabled;
    uint64_t total;       // sum of region bytes (one step)
    uint64_t lookahead;   // bytes kept in flight ahead of `consumed`
    uint64_t min_lead;    // chunks closer than this to `consumed` are left to demand (see the issuer)
    uint32_t chunk;       // bytes per prefetch request
    uint32_t credits;     // PREFETCH_RING_CREDIT_* mask: non-weight consumers that credit their reads
    uint32_t stalled;     // issuer CTAs that gave up waiting for consumption
    uint64_t consumed;    // bytes whose demand loads completed this step
};

#ifdef __CUDACC__
static __device__ __forceinline__ void prefetch_ring_consume_device(
    PrefetchRingState* ring, uint64_t bytes) {
    if (ring != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(&ring->consumed),
                  static_cast<unsigned long long>(bytes));
}
#endif

// Allocates the current device's ring state on first call (false before sm_90).
bool prefetch_ring_is_available();
// The current device's ring state for kernel-argument consumers; nullptr until
// prefetch_ring_is_available has run on this device.
extern "C" PrefetchRingState* prefetch_ring_consumer_state();

extern "C" void launch_prefetch_ring_configure(
    const uint64_t* regions, int count, uint64_t lookahead, uint64_t min_lead, uint32_t chunk, uint32_t credits,
    cudaStream_t stream);
extern "C" void launch_prefetch_ring_disable(cudaStream_t stream);
extern "C" void launch_prefetch_ring_start(cudaStream_t stream);
extern "C" void set_w4a8_prefetch_ring_state(PrefetchRingState* state);
extern "C" void set_flash_prefetch_ring_state(PrefetchRingState* state);
extern "C" void set_gated_delta_prefetch_ring_state(PrefetchRingState* state);
extern "C" void prefetch_ring_read_counters(PrefetchRingState* host);
