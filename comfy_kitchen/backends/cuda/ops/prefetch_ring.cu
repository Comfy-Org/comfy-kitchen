#include <cuda_runtime.h>

#include <cstdint>

#include "../prefetch_ring.h"

namespace {

// Issuer geometry, measured on RTX 5090 / CUDA 13.0 (pfbench/pf_rate.cu,
// contend4_bench.py): one cp.async.bulk.prefetch.L2 issuer thread sustains
// ~170 GB/s and self-throttles, 10 of them reach the DRAM ceiling (1.64-1.71
// TB/s) with every line landing; from 14 on, requests are dropped. Prefetch
// requests return no data to the SM, which is what makes them cheap for a
// concurrently running L2-hitting GEMM: an ld-based issuer at the same rate
// slows such a GEMM about twice as much (contend3_bench.py).
constexpr int kIssuers = PREFETCH_RING_ISSUERS;
#ifndef PREFETCH_RING_ISSUER_CARVEOUT
#define PREFETCH_RING_ISSUER_CARVEOUT cudaSharedmemCarveoutMaxShared
#endif
constexpr int kIssuerThreads = 32;
constexpr int kIssueBatch = 8;   // chunks issued per consumed-snapshot
PrefetchRingState* g_states[16] = {};
bool g_unsupported[16] = {};   // pre-sm_90: no bulk prefetch, consumers keep ring == nullptr
cudaStream_t g_issue_streams[16] = {};
cudaEvent_t g_start_events[16] = {};

__device__ uint64_t region_total(const PrefetchRegion* regions, int count) {
    uint64_t total = 0;
    for (int i = 0; i < count; ++i) total += regions[i].bytes;
    return total;
}

__global__ void configure_prefetch_ring_kernel(
    PrefetchRingState* ring, const PrefetchRegion* regions, int count,
    uint64_t lookahead, uint32_t chunk, uint32_t credits) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    ring->regions = regions;
    ring->count = count;
    ring->total = region_total(regions, count);
    ring->lookahead = lookahead;
    ring->chunk = chunk;
    ring->credits = credits;
    ring->stalled = 0;
    ring->consumed = 0;
    ring->touched = ring->skipped = ring->waited_ns = 0;
    ring->arrived = 0;
    for (int i = 0; i <= kIssuers; ++i) ring->distinct_hist[i] = 0;
    ring->enabled = count > 0 && ring->total > 0;
}

__global__ void disable_prefetch_ring_kernel(PrefetchRingState* ring) {
    if (blockIdx.x == 0 && threadIdx.x == 0) ring->enabled = 0;
}

// Runs on the consumer stream at step start: every consumer of the previous
// step (including W4A8 kernels outside the ring scope) has finished, so the
// step's byte count restarts at zero. Re-arms a ring that was disabled
// mid-step (a host sync must stop the issuer first; see prefetch_ring_disable).
// Region byte counts may change between steps (KV rows grow), so the step
// total is summed here from the descriptors the host updated on `stream`.
__global__ void reset_prefetch_ring_kernel(PrefetchRingState* ring) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ring->total = region_total(ring->regions, ring->count);
        ring->consumed = 0;
        ring->arrived = 0;
        ring->trace_n = 0;
        ring->enabled = ring->count > 0 && ring->total > 0;
    }
}

__global__ void set_prefetch_ring_trace_kernel(PrefetchRingState* ring, uint64_t* trace, uint32_t cap) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ring->trace = trace;
        ring->trace_cap = cap;
        ring->trace_n = 0;
    }
}

__device__ __forceinline__ uint64_t global_timer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void trace_record(PrefetchRingState* ring, uint64_t* trace, uint32_t cap,
                                             uint64_t consumed, uint64_t cursor, uint32_t event) {
    if (trace == nullptr) return;
    const uint32_t i = atomicAdd(&ring->trace_n, 1u);
    if (i >= cap) return;
    uint64_t* r = trace + static_cast<size_t>(i) * 4;
    r[0] = global_timer();
    r[1] = consumed;
    r[2] = cursor;
    r[3] = blockIdx.x | (static_cast<uint64_t>(event) << 8);
}

// Cursor into the concatenated region list; advancing by n bytes walks regions.
// The current region is cached in registers: the issue loop must not take a
// dependent global load per chunk (see the issuer geometry note).
//
// Every CTA walks the whole stream (it prefetches only its own chunks), so the
// CTA given `consumed` credits each self-crediting region once, as its walk
// leaves the region on the first pass; the wrap-around past `total` is not
// a read of this step.
struct RingCursor {
    const PrefetchRegion* regions;
    int count;
    int index;
    uint64_t offset;
    uint64_t position;   // walked bytes of the step's stream
    uint64_t total;
    uint64_t* consumed;  // nullptr: this CTA does not credit
    const unsigned char* base;
    uint64_t bytes;
    uint64_t flags;

    __device__ void load() {
        const PrefetchRegion r = regions[index];
        base = r.base;
        bytes = r.bytes;
        flags = r.flags;
    }
    __device__ void next_region() {
        if (consumed != nullptr && (flags & PREFETCH_REGION_SELF_CREDIT) && position <= total)
            atomicAdd(reinterpret_cast<unsigned long long*>(consumed), static_cast<unsigned long long>(bytes));
        offset = 0;
        if (++index == count) index = 0;
        load();
    }
    __device__ void advance(uint64_t n) {
        while (n != 0) {
            const uint64_t avail = bytes - offset;
            const uint64_t step = n < avail ? n : avail;
            n -= step;
            offset += step;
            position += step;
            if (offset == bytes) next_region();
        }
    }
    // Request [cursor, cursor + n) into L2, one bulk prefetch per region piece.
    // Bulk prefetch is sm_90+; prefetch_ring_is_available keeps older devices off the ring.
    __device__ void prefetch(uint64_t n) {
        while (n != 0) {
            const uint64_t avail = bytes - offset;
            const uint64_t step = n < avail ? n : avail;
#if __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                         :: "l"(base + offset), "r"(static_cast<unsigned>(step)) : "memory");
#endif
            n -= step;
            offset += step;
            position += step;
            if (offset == bytes) next_region();
        }
    }
};

// One thread per CTA; CTA i owns chunks i, i + kIssuers, ... of the step's byte
// stream. A chunk is requested once it lies within `lookahead` of the consumed
// position; chunks demand has already consumed are skipped. The kernel ends
// after requesting `lookahead` bytes past the end of the step (the wrap-around
// start of the next one). CTA 0 credits the self-crediting regions as it
// passes them (RingCursor); that credit lands when the region is issued, a
// little ahead of its read, which widens the window by at most the
// self-crediting bytes within one lookahead.
//
// The loop decides from a snapshot of the racy counters that was loaded while
// the previous chunk issued, so the fast path has no load latency in it; a
// stale snapshot only delays a skip or a window advance by one chunk. The
// counters are polled synchronously only while the window is exhausted.

__global__ void __launch_bounds__(kIssuerThreads) prefetch_ring_issuer_kernel(PrefetchRingState* ring) {
    if (threadIdx.x != 0) return;
    {
        unsigned sm;
        asm("mov.u32 %0, %%smid;" : "=r"(sm));
        ring->smid[blockIdx.x] = sm;
        __threadfence();
        if (atomicAdd(&ring->arrived, 1u) == gridDim.x - 1) {   // last CTA to start: count distinct SMs
            int distinct = 0;
            for (int i = 0; i < static_cast<int>(gridDim.x); ++i) {
                const unsigned si = *reinterpret_cast<volatile uint32_t*>(&ring->smid[i]);
                bool seen = false;
                for (int j = 0; j < i; ++j) seen |= *reinterpret_cast<volatile uint32_t*>(&ring->smid[j]) == si;
                distinct += !seen;
            }
            atomicAdd(&ring->distinct_hist[distinct], 1u);
        }
    }
    const uint64_t end = ring->total;
    const uint64_t lookahead = ring->lookahead;
    const uint64_t chunk = ring->chunk;
    const uint64_t stride = chunk * gridDim.x;
    volatile uint64_t* consumed_p = &ring->consumed;
    volatile int* enabled_p = &ring->enabled;
    uint64_t* const trace = ring->trace;
    const uint32_t trace_cap = ring->trace_cap;

    uint64_t cursor = chunk * blockIdx.x;
    RingCursor pos{ring->regions, ring->count, 0, 0, 0, end, blockIdx.x == 0 ? &ring->consumed : nullptr, nullptr, 0, 0};
    pos.load();
    pos.advance(cursor);

    uint64_t touched = 0, skipped = 0, waited = 0;
    uint64_t consumed = *consumed_p;
    int enabled = *enabled_p;
    while (cursor < end + lookahead && enabled) {
        if (cursor + chunk <= consumed) {
            // demand already read this: jump to our first chunk at/after the consumed position
            const uint64_t skip = ((consumed / chunk) * chunk - cursor + stride - 1) / stride * stride;
            cursor += skip;
            pos.advance(skip);
            skipped += skip / gridDim.x;
            trace_record(ring, trace, trace_cap, consumed, cursor, PREFETCH_RING_TRACE_SKIP);
            continue;
        }
        if (cursor >= consumed + lookahead) {
            const uint64_t t0 = global_timer();
            do {
                __nanosleep(256);
                waited += 256;
                consumed = *consumed_p;
                enabled = *enabled_p;
                if (!enabled) break;
                if (global_timer() - t0 > 100000000ull) {   // 100 ms without consumption
                    atomicAdd(&ring->stalled, 1u);
                    enabled = 0;
                    break;
                }
            } while (cursor >= consumed + lookahead);
            trace_record(ring, trace, trace_cap, consumed, cursor, PREFETCH_RING_TRACE_WAIT);
            continue;   // re-evaluate against the fresh snapshot
        }
        // Issue a batch of chunks against the current snapshot. The refreshed
        // snapshot is loaded first and only read after the batch, so its L2
        // round trip (which grows under GEMM hit traffic) overlaps the issues
        // instead of gating each one: one dependent load per chunk capped the
        // issuer at ~1.2 TB/s during GEMMs.
        const uint64_t next_consumed = *consumed_p;
        const int next_enabled = *enabled_p;
        const uint64_t cap = consumed + lookahead < end + lookahead ? consumed + lookahead : end + lookahead;
        trace_record(ring, trace, trace_cap, consumed, cursor, PREFETCH_RING_TRACE_ISSUE);
        for (int k = 0; k < kIssueBatch && cursor < cap; ++k) {
            pos.prefetch(chunk);
            touched += chunk;
            pos.advance(stride - chunk);
            cursor += stride;
        }
        consumed = next_consumed;
        enabled = next_enabled;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&ring->touched), touched);
    atomicAdd(reinterpret_cast<unsigned long long*>(&ring->skipped), skipped);
    atomicAdd(reinterpret_cast<unsigned long long*>(&ring->waited_ns), waited);
}


PrefetchRingState* current_state(int* device_out) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16 || g_unsupported[device])
        return nullptr;
    if (g_states[device] == nullptr) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess || major < 9) {
            g_unsupported[device] = true;
            return nullptr;
        }
        if (cudaMalloc(&g_states[device], sizeof(PrefetchRingState)) != cudaSuccess)
            return nullptr;
        cudaMemset(g_states[device], 0, sizeof(PrefetchRingState));
        cudaStreamCreateWithFlags(&g_issue_streams[device], cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&g_start_events[device], cudaEventDisableTiming);
        // The issuer CTAs are resident for the whole step. An SM only hosts CTAs
        // that agree on its L1/shared carveout, and a kernel with no shared
        // memory would be configured for maximum L1, keeping the streamed GEMM
        // (~46 KB dynamic shared per CTA) off those SMs for the whole step.
        // Ask for the GEMM's configuration so the SMs stay shared.
        cudaFuncSetAttribute(prefetch_ring_issuer_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             PREFETCH_RING_ISSUER_CARVEOUT);
        set_w4a8_prefetch_ring_state(g_states[device]);
        set_flash_prefetch_ring_state(g_states[device]);
        set_gated_delta_prefetch_ring_state(g_states[device]);
    }
    if (device_out) *device_out = device;
    return g_states[device];
}

} // namespace

// Also allocates the device's state, so consumer kernels captured into a CUDA graph
// after this call bake the live pointer even when the ring is configured later.
bool prefetch_ring_is_available() {
    return current_state(nullptr) != nullptr;
}

// Lookup only: consumer launches may be under stream capture, where allocation is illegal.
extern "C" PrefetchRingState* prefetch_ring_consumer_state() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) return nullptr;
    return g_states[device];
}

extern "C" void launch_prefetch_ring_configure(
    const uint64_t* regions, int count, uint64_t lookahead, uint32_t chunk, uint32_t credits,
    cudaStream_t stream) {
    PrefetchRingState* state = current_state(nullptr);
    if (state == nullptr) return;
    configure_prefetch_ring_kernel<<<1, 1, 0, stream>>>(
        state, reinterpret_cast<const PrefetchRegion*>(regions), count, lookahead, chunk, credits);
}

extern "C" void launch_prefetch_ring_disable(cudaStream_t stream) {
    PrefetchRingState* state = current_state(nullptr);
    if (state != nullptr) disable_prefetch_ring_kernel<<<1, 1, 0, stream>>>(state);
}

// Start the issuer for one step on the ring's side stream, ordered after the
// work already enqueued on `stream` (configure / previous step) and the
// counter reset.
extern "C" void launch_prefetch_ring_start(cudaStream_t stream) {
    int device = 0;
    PrefetchRingState* state = current_state(&device);
    if (state == nullptr) return;
    reset_prefetch_ring_kernel<<<1, 1, 0, stream>>>(state);
    cudaEventRecord(g_start_events[device], stream);
    cudaStreamWaitEvent(g_issue_streams[device], g_start_events[device], 0);
    prefetch_ring_issuer_kernel<<<kIssuers, kIssuerThreads, 0, g_issue_streams[device]>>>(state);
}

extern "C" void launch_prefetch_ring_set_trace(uint64_t* trace, uint32_t cap, cudaStream_t stream) {
    PrefetchRingState* state = current_state(nullptr);
    if (state != nullptr) set_prefetch_ring_trace_kernel<<<1, 1, 0, stream>>>(state, trace, cap);
}

// Diagnostic: synchronous copy of the state counters.
extern "C" void prefetch_ring_read_stats(PrefetchRingState* host) {
    PrefetchRingState* state = current_state(nullptr);
    *host = PrefetchRingState{};
    if (state != nullptr) cudaMemcpy(host, state, sizeof(*host), cudaMemcpyDeviceToHost);
}
