// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <cutlass/arch/barrier.h>

#include "launch.h"
#include "layout.h"
#include "math.cuh"
#include "softmax.cuh"

namespace comfy_deco::detail {
// Per-thread operands mirror the warp-level MMA register fragments. All array
// accesses in the hot loop must be statically resolved to avoid local memory.
template <int Sequence, int Heads>
struct Registers {
    uint32_t qa[3][4], qsf[3], Pst[12], Mlo[2];
    float O[64], anchor[2], lsum[2], thr[2];
    float c, M, vscale;
    int safe, t, par;
    int ldsm_off, ksf_lane, vsf_lane, vm_lane, vm_mh, vm_pb;
    static constexpr int S = Sequence, H = Heads;
};

constexpr int smem_mean_offset = 4 * layout::mean_offset;
constexpr int full_offset = smem_mean_offset + 2 * layout::mean_bytes;
constexpr int empty_offset = full_offset + 32;
constexpr int count_offset = empty_offset + 32;
constexpr int shared_bytes = full_offset + 128;
static_assert(shared_bytes == 100480);

struct Ring {
    unsigned char* smem;
    const unsigned char* image;
    int end, wait_id, post_id;
    __device__ __forceinline__ uint32_t addr(int offset) const {
        return static_cast<uint32_t>(__cvta_generic_to_shared(smem + offset));
    }
    __device__ __forceinline__ uint64_t* barrier(int off, int stage) const {
        return reinterpret_cast<uint64_t*>(smem + off + 8 * stage);
    }
};

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, int bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}
template <int Stage, bool Initial>
__device__ __forceinline__ void fill(const Ring& ring, int tile) {
    const auto* src = ring.image + static_cast<size_t>(tile) * layout::tile_bytes;
    const bool has_means = (Stage & 1) || tile == ring.end - 1;
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        ring.barrier(full_offset, Stage), layout::mean_offset + (has_means ? layout::mean_bytes : 0));
    const uint32_t bar = ring.addr(full_offset + 8 * Stage);
    bulk_load(ring.addr(Stage * layout::mean_offset), src, layout::mean_offset, bar);
    if ((Stage & 1) || (Initial && has_means)) {
        bulk_load(ring.addr(smem_mean_offset + (Stage / 2) * layout::mean_bytes),
                  src + layout::mean_offset, layout::mean_bytes, bar);
    }
}
template <int Stage>
__device__ __forceinline__ void release(const Ring& ring, int previous_tile, int parity) {
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::ClusterBarrier::arrive(ring.barrier(empty_offset, Stage));
    __syncwarp();
    uint32_t previous;
    asm volatile("{ .reg .pred elected; .reg .b32 old; elect.sync _|elected, 0xffffffff; "
                 "mov.u32 old,0; @elected atom.relaxed.cta.shared::cta.add.u32 old,[%1],1; "
                 "mov.b32 %0,old; }"
                 : "=r"(previous)
                 : "r"(ring.addr(count_offset + 4 * Stage))
                 : "memory");
    if ((previous & 7) == 7) {
        if (previous_tile + 4 < ring.end) {
            cutlass::arch::ClusterBarrier::wait(ring.barrier(empty_offset, Stage), parity);
            cutlass::arch::fence_view_async_shared();
            fill<Stage, false>(ring, previous_tile + 4);
        } else if constexpr (Stage & 1) {
            // An odd tile and the unpaired last tile share this means buffer.
            // Delay the final DMA until every warp consumed the older means.
            if ((ring.end & 1) && previous_tile == ring.end - 4) {
                cutlass::arch::ClusterBarrier::wait(ring.barrier(empty_offset, Stage), parity);
                cutlass::arch::fence_view_async_shared();
                constexpr int last_stage = (Stage + 3) & 3;
                bulk_load(ring.addr(smem_mean_offset + (last_stage / 2) * layout::mean_bytes),
                          ring.image + size_t(ring.end - 1) * layout::tile_bytes + layout::mean_offset,
                          layout::mean_bytes, ring.addr(full_offset + 8 * last_stage));
            }
        }
    }
}
template <int Stage, bool Masked, class State>
__device__ __forceinline__ void step(State& st, const Ring& ring, int tile, int phase) {
    constexpr int Prev = (Stage + 3) & 3;
    const int valid = Masked ? 1 + (State::S - 1) % 128 : 128;
    cutlass::arch::ClusterBarrier::wait(ring.barrier(full_offset, Stage), phase);
    Fragment early_q[4], early_v[4];
    const uint32_t qs = ring.addr(Stage * layout::mean_offset);
    const uint32_t vs = ring.addr(Prev * layout::mean_offset);
    preload(st, qs, early_q);
    preload(st, vs + layout::value_offset, early_v);
    cutlass::arch::NamedBarrier::sync(64, ring.wait_id);
    float scores[16][4];
    qk<Masked>(st, qs, scores, valid, early_q);
    pv(st, vs, early_v);
    if constexpr (Prev & 1) {
        means(st, ring.addr(smem_mean_offset + (Prev / 2) * layout::mean_bytes), st.Pst[10], st.Pst[11]);
    }
    cutlass::arch::NamedBarrier::arrive(64, ring.post_id);
    release<Prev>(ring, tile - 1, phase ^ (Stage == 0));
    softmax<Masked, (Stage & 1) != 0>(st, scores, valid);
}
template <int Stage, class State>
__device__ __forceinline__ void finish(State& st, const Ring& ring) {
    Fragment early_v[4];
    const uint32_t vs = ring.addr(Stage * layout::mean_offset);
    preload(st, vs + layout::value_offset, early_v);
    cutlass::arch::NamedBarrier::sync(64, ring.wait_id);
    pv(st, vs, early_v);
    means(st, ring.addr(smem_mean_offset + (Stage / 2) * layout::mean_bytes),
          (Stage & 1) ? st.Pst[10] : 0, (Stage & 1) ? st.Pst[11] : 0);
    if (threadIdx.x < 128) cutlass::arch::NamedBarrier::arrive(64, ring.post_id);
}

template <int Sequence, int Heads, bool SingleWriter>
__global__ __launch_bounds__(256, 1) void deco_fp4_core(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ sf,
    const float* __restrict__ score_scale, const float* __restrict__ vscale,
    const unsigned char* __restrict__ image, uint16_t* __restrict__ out,
    unsigned char* __restrict__ scratch, int full, int split) {
    constexpr int tile_count = (Sequence + 127) / 128;
    constexpr int key_midpoint = 4 * ((tile_count + 4) / 8);
    static_assert(tile_count >= 24 && Sequence <= INT_MAX - 127 && Heads > 0);
    static_assert(int64_t(2) * Heads * tile_count <= INT_MAX);
    extern __shared__ __align__(1024) unsigned char smem[];
    const int block = blockIdx.x;
    const int part = block < full ? 0 : 1 + ((block - full) & 1);
    const int unit = block < full ? block : full + ((block - full) >> 1);
    const int qt = unit % tile_count;
    const int head = unit / tile_count;
    const int first = part == 2 ? key_midpoint : 0;
    const int end = part == 1 ? key_midpoint : tile_count;
    const int warp = threadIdx.x >> 5;
    const int pair = warp & 3;
    const int phase_b = warp >> 2;
    const Ring ring{smem, image + size_t(head) * tile_count * layout::tile_bytes, end,
                    pair + 4 * phase_b, 4 + pair - 4 * phase_b};
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < 4; s++) {
            cutlass::arch::ClusterBarrier::init(ring.barrier(full_offset, s), 1);
            cutlass::arch::ClusterBarrier::init(ring.barrier(empty_offset, s), 256);
            reinterpret_cast<uint32_t*>(smem + count_offset)[s] = 0;
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        fill<0, true>(ring, first);
        fill<1, true>(ring, first + 1);
        fill<2, true>(ring, first + 2);
        fill<3, true>(ring, first + 3);
    }
    Registers<Sequence, Heads> st;
    load_query(st, q, sf, Sequence, Heads, qt, head);
    st.par = (threadIdx.x & 1);
    st.c = *score_scale;
    st.M = make_subnormal_scale(st.c);
    st.vscale = *vscale;
    st.safe = 0;
#pragma unroll
    for (int i = 0; i < 64; ++i)
        st.O[i] = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        st.anchor[i] = st.thr[i] = __int_as_float(0xff800000);
        st.lsum[i] = 0;
    }
    {
        cutlass::arch::ClusterBarrier::wait(ring.barrier(full_offset, 0), 0);
        Fragment early_q[4];
        const auto qs = ring.addr(0);
        preload(st, qs, early_q);
        if (phase_b) cutlass::arch::NamedBarrier::sync(64, ring.wait_id);
        float scores[16][4];
        qk<false>(st, qs, scores, 128, early_q);
        cutlass::arch::NamedBarrier::arrive(64, ring.post_id);
        softmax<false, false>(st, scores, 128);
    }
    if constexpr (Sequence == 16481 && Heads == 56) {
        int tile = first + 1, ph = 0;
// All plan lengths leave three or four iterations after the last
// complete four-stage cycle. Reuse the same three instruction blocks.
#pragma unroll 1
        for (;; tile += 4, ph ^= 1) {
            step<1, false>(st, ring, tile, ph);
            step<2, false>(st, ring, tile + 1, ph);
            step<3, false>(st, ring, tile + 2, ph);
            if (tile + 4 >= end) break;
            step<0, false>(st, ring, tile + 3, ph ^ 1);
        }
        if (end & 1) {
            step<0, true>(st, ring, tile + 3, ph ^ 1);
            finish<0>(st, ring);
        } else {
            finish<3>(st, ring);
        }
    } else {
        constexpr bool partial_tail = (Sequence % 128) != 0;
// The full sequence and the right split share their final tile. The
// left split always ends on stage 3 and uses all 128 keys in that tile.
#pragma unroll 1
        for (int tile = first + 1, phase = 0;; tile += 4, phase ^= 1) {
            if constexpr (tile_count % 4 == 2) {
                if (tile + 1 == end) {
                    step<1, partial_tail>(st, ring, tile, phase);
                    finish<1>(st, ring);
                    break;
                }
            }
            step<1, false>(st, ring, tile, phase);
            if constexpr (tile_count % 4 == 3) {
                if (tile + 2 == end) {
                    step<2, partial_tail>(st, ring, tile + 1, phase);
                    finish<2>(st, ring);
                    break;
                }
            }
            step<2, false>(st, ring, tile + 1, phase);
            if constexpr (tile_count % 4 == 0 && partial_tail) {
                if (tile + 3 == tile_count) {
                    step<3, true>(st, ring, tile + 2, phase);
                    finish<3>(st, ring);
                    break;
                }
            }
            step<3, false>(st, ring, tile + 2, phase);
            if (tile + 3 == end) {
                finish<3>(st, ring);
                break;
            }
            if constexpr (tile_count % 4 == 1) {
                if (tile + 4 == end) {
                    step<0, partial_tail>(st, ring, tile + 3, phase ^ 1);
                    finish<0>(st, ring);
                    break;
                }
            }
            step<0, false>(st, ring, tile + 3, phase ^ 1);
        }
    }
    const int row_g = qt * 128 + warp * 16 + ((threadIdx.x & 31) >> 2);
    if (part == 0) {
        store_final<SingleWriter>(st, out, row_g, head);
    } else {
        store_partial<SingleWriter>(st, out, row_g, head, scratch, split, unit - full, part,
                                    reinterpret_cast<int*>(smem + count_offset + 16));
    }
}

template <int Sequence, int Heads, bool SingleWriter>
cudaError_t launch_policy(const CoreArguments& args, int sms, cudaStream_t stream) {
    constexpr int units = ((Sequence + 127) / 128) * Heads;
    const int split = units % sms;
    const int full = units - split;

    auto err = cudaFuncSetAttribute(deco_fp4_core<Sequence, Heads, SingleWriter>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (err != cudaSuccess) return err;
    deco_fp4_core<Sequence, Heads, SingleWriter>
        <<<full + 2 * split, layout::threads, shared_bytes, stream>>>(
            static_cast<const unsigned char*>(args.q4), static_cast<const unsigned char*>(args.q_sf),
            static_cast<const float*>(args.score_scale), static_cast<const float*>(args.v4_scale),
            static_cast<const unsigned char*>(args.image), static_cast<uint16_t*>(args.out),
            static_cast<unsigned char*>(args.scratch), full, split);
    return cudaGetLastError();
}

template <int Sequence, int Heads>
cudaError_t launch_shape(const CoreArguments& args, int sms, bool single_writer, cudaStream_t stream) {
    return single_writer ? launch_policy<Sequence, Heads, true>(args, sms, stream)
                         : launch_policy<Sequence, Heads, false>(args, sms, stream);
}

} // namespace comfy_deco::detail
