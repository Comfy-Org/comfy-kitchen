// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SM12x Q/K packing and final scale encoding.
#include <cuda_runtime.h>
#include <cstdint>
#include "prep_v.cuh"
#include "../sage_attention/mma.cuh"

namespace comfy_deco {
namespace {

constexpr int threads = 256;
constexpr int stripes_per_head = 8;
constexpr float query_prescale = static_cast<float>(1.4426950408889634 / 11.31370849898476039);

using U32 = uint32_t;
using U16 = uint16_t;
using U8 = uint8_t;

// Four consecutive BF16 scales per group. Quantization sets the low
// four mantissa bits to zero. Middle holds the next eight bits. Each warp
// covers 16 rows and shares one flag for atoms 0 and 1, with a second for atom 2.
struct alignas(64) ScaleSidePlane {
    U16 upper_nibbles[layout::query_scale_bytes / 4];
    U32 exceptional[8][2];
};
static_assert(sizeof(ScaleSidePlane) == layout::scale_side_bytes);


struct Block {
    float scale;
    U32 codes;
};

__device__ __forceinline__ Block quantize(float a, float b, float c, float d) {
    float peak = fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d)));
    peak = fmaxf(0.0f, peak);
    peak = fmaxf(peak, __shfl_xor_sync(~0u, peak, 1));
    peak = fmaxf(peak, __shfl_xor_sync(~0u, peak, 2));
    const U32 bits = __float_as_uint(__fmul_rn(peak, 1.0f / 6.0f));
    const float scale = __uint_as_float((bits + 0x7ffffu + ((bits >> 20) & 1u)) & 0xfff00000u);
    float inv;
    asm("rcp.approx.f32 %0,%1;" : "=f"(inv) : "f"(scale));
    if (scale <= 1e-30f) inv = 0;
    return {scale, comfy::pack_fp4x4<false>(a * inv, b * inv, c * inv, d * inv)};
}

struct alignas(16) WarpStage {
    U8 codes[16][144];  // Another 16 bytes per row avoid shared bank aliasing.
    U16 scales[16][12];
};

__device__ __forceinline__ void stage_block(WarpStage &stage, int block, int row, int quad,
                                            Block values) {
    stage.codes[row][8 * block + quad] = static_cast<U8>(values.codes);
    stage.codes[row][8 * block + 4 + quad] = static_cast<U8>(values.codes >> 8);
    if (quad == (block & 3))
        stage.scales[row][block] = static_cast<U16>(__float_as_uint(values.scale) >> 16);
}

__device__ __forceinline__ int ntile_row(int r) {
    return 8 * (8 * (r >> 6) + 4 * ((r >> 5) & 1) + ((r & 7) >> 1)) +
           2 * ((r & 31) >> 3) + (r & 1);
}

__device__ __forceinline__ void store_stream(uint4 *dest, uint4 data) {
    asm volatile("st.global.cs.v4.b32 [%0],{%1,%2,%3,%4};"
                 :: "l"(dest), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w) : "memory");
}

__device__ __forceinline__ void store_staged(PrepGeometry shape, WarpStage &stage, int head,
                                             int tile, bool key, U8 *q4, U8 *img, U16 *prov, U8* qsf) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int tok0 = tile * 128 + warp * 16;
    const int tile_id = head * shape.tiles + tile;
    __syncwarp();
    if (!key) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int i = 32*n + lane, row = i / 8, chunk = i % 8;
            if (tok0 + row < shape.sequence && chunk < 6) {
                uint4 data = *reinterpret_cast<const uint4*>(&stage.codes[row][16*chunk]);
                U8* dst = q4 + (size_t(tok0 + row)*shape.heads + head) * layout::query_row_bytes;
                store_stream(reinterpret_cast<uint4*>(dst) + chunk, data);
            }
        }
    } else {
        int row = lane & 15, key_row = ntile_row(warp * 16 + row);
        U8 *base = img + size_t(tile_id) * layout::tile_bytes;
        #pragma unroll
        for (int n = 0; n < 3; ++n) {
            int chunk = (lane >> 4) + 2 * n;
            uint4 data = tok0 + row < shape.sequence
                ? *reinterpret_cast<uint4*>(&stage.codes[row][16 * chunk]) : make_uint4(0, 0, 0, 0);
            store_stream(reinterpret_cast<uint4*>(base) + 128 * chunk + key_row, data);
        }
    }
    // For BF16 bit patterns [0x3800, 0x4800), the upper nibble is 3
    // when bit 11 is set and 4 otherwise. If any scale in this warp
    // segment is outside the interval, retain every upper nibble.
    auto& packed = reinterpret_cast<ScaleSidePlane*>(prov)[(key ? shape.heads * shape.tiles : 0) + tile_id];
    auto* final_scales = key ? img+size_t(tile_id)*layout::tile_bytes+layout::key_scale_offset : qsf+tile_id*layout::query_scale_bytes;
    #pragma unroll
    for (int segment = 0; segment < 2; ++segment) {
        const int i = lane + 32 * segment;
        if (i < 48) {
            const int row = i & 15, atom = i >> 4;
            const int group = atom * 128 + warp * 16 + row;
            const uint2 values = *reinterpret_cast<const uint2*>(&stage.scales[row][4 * atom]);
            // Adding 0x0800 to each halfword moves the common interval
            // into [0x4000, 0x5000), allowing a single upper-nibble test.
            const U32 out_x = __vadd2(values.x, 0x08000800u) ^ 0x40004000u;
            const U32 out_y = __vadd2(values.y, 0x08000800u) ^ 0x40004000u;
            const U32 mask = segment == 0 ? ~0u : 0xffffu;
            const bool exceptional = __any_sync(mask, ((out_x | out_y) & 0xf000f000u) != 0);
            if (lane == 0) packed.exceptional[warp][segment] = exceptional;
            const U32 middle = __byte_perm(values.x >> 4, values.y >> 4, 0x6420);
            const int r = group & 127;
            const int index = atom*128 + (r%32)*4 + r/32;
            reinterpret_cast<U32*>(final_scales)[index] = middle;
            if (exceptional) {
                const U32 upper = __byte_perm(values.x, values.y, 0x7531) & 0xf0f0f0f0u;
                const U32 pairs = (upper >> 4) | (upper >> 8);
                packed.upper_nibbles[group] = static_cast<U16>(__byte_perm(pairs, 0u, 0x4420));
            }
        }
    }
    __syncwarp();
}


__device__ __forceinline__ void load_rows(InputView src, int sequence, int tok0, int head,
                                          uint2 (&rows)[8][2]) {
    const int lane = threadIdx.x & 31;
    const int row = tok0 + (lane >> 2), t = lane & 3;
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int offset = 32 * a + t * 8;
        rows[a][0] = row < sequence
            ? *reinterpret_cast<const uint2*>(src.row(row,head)+offset) : make_uint2(0,0);
        rows[a][1] = row+8 < sequence
            ? *reinterpret_cast<const uint2*>(src.row(row+8,head)+offset) : make_uint2(0,0);
    }
}
__device__ __forceinline__ void transform(const uint2 (&rows)[8][2],
                                          const uint2 (&rotation)[8][2], float (&y)[8][2][4]) {
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const uint2 lo = rows[a][0], hi = rows[a][1];
        const unsigned operand[4] = {lo.x, hi.x, lo.y, hi.y};
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) y[a][n][e] = 0;
            const uint32_t b[] = {rotation[a][n].x, rotation[a][n].y};
            mma::mma_bf16(y[a][n], operand, b);
        }
    }
    // The canonical DeCo rotation is diag(signs) times H8 (x) H16.
    #pragma unroll
    for (int step = 1; step <= 4; step *= 2) {
        #pragma unroll
        for (int a = 0; a < 8; ++a) if (!(a & step)) {
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float lo = y[a][n][e], hi = y[a+step][n][e];
                    y[a][n][e] = lo + hi;
                    y[a+step][n][e] = lo - hi;
                }
            }
        }
    }
}

__device__ __forceinline__ void encode_item(float (&y)[8][2][4], bool key,
                                            const U16 (&basis)[32][136],
                                            WarpStage &stage, float (&groups)[3]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    float body[4][4] = {}, residual[4][4] = {};
    const float factor = key ? 1.f : query_prescale;
    #pragma unroll
    for (int bo = 0; bo < 8; ++bo) {
        unsigned aq[4], ad[4];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float a = __fmul_rn(y[bo][0][2*r], factor);
            float b = __fmul_rn(y[bo][0][2*r+1], factor);
            float c = __fmul_rn(y[bo][1][2*r], factor);
            float d = __fmul_rn(y[bo][1][2*r+1], factor);
            Block x = quantize(a, b, c, d);
            stage_block(stage, bo, g + 8*r, t, x);
            groups[0] = fmaxf(groups[0], x.scale);
            const auto ab = __half22float2(__nv_cvt_fp4x2_to_halfraw2(uint8_t(x.codes), __NV_E2M1));
            const auto cd = __half22float2(__nv_cvt_fp4x2_to_halfraw2(uint8_t(x.codes >> 8), __NV_E2M1));
            const float da = ab.x * x.scale, db = ab.y * x.scale;
            const float dc = cd.x * x.scale, dd = cd.y * x.scale;
            aq[r] = comfy::pack_bf2(da, db); aq[2+r] = comfy::pack_bf2(dc, dd);
            ad[r] = comfy::pack_bf2(a-da, b-db); ad[2+r] = comfy::pack_bf2(c-dc, d-dd);
        }
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const U16 *ub = &basis[8*n+g][16*bo+2*t];
            const U32 frag[] = {*reinterpret_cast<const U32*>(ub), *reinterpret_cast<const U32*>(ub+8)};
            mma::mma_bf16(body[n], aq, frag);
            mma::mma_bf16(residual[n], ad, frag);
        }
    }
    #pragma unroll
    for (int ci = 0; ci < 2; ++ci) {
        bool take_body = key == (ci == 0);
        #pragma unroll
        for (int bp = 0; bp < 2; ++bp) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int n = 2*bp, e = 2*r;
                Block x = take_body
                    ? quantize(body[n][e], body[n][e+1], body[n+1][e], body[n+1][e+1])
                    : quantize(residual[n][e], residual[n][e+1], residual[n+1][e], residual[n+1][e+1]);
                groups[1+ci] = fmaxf(groups[1+ci], x.scale);
                stage_block(stage, 8+2*ci+bp, g+8*r, t, x);
            }
        }
    }
}

// Each block owns one head and side. Adjacent blocks touch adjacent 256-byte
// heads in the source projection; several blocks interleave that head's tiles.
// Three BF16 maxima suffice because each provisional scale is rounded to M3.
template<bool Key>
__device__ __forceinline__ void process_head(PrepGeometry shape, InputView src, const U16* had,
        const U16* ut, U8* q4, U8* img, U16* prov, U32* maxima, U8* qsf,
        U16 (&basis)[32][136], WarpStage (&stage)[8], float (&red)[8][4]) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int head=blockIdx.x%shape.heads, stripe=blockIdx.x/(2*shape.heads);
    uint2 rotation[8][2];
    #pragma unroll
    for (int a=0; a<8; ++a) {
        #pragma unroll
        for (int n=0; n<2; ++n) {
            const U16* h = had + (16*a + 4*t)*128 + 8*n + g;
            rotation[a][n] = make_uint2(h[0] | (U32(h[128]) << 16),
                                        h[256] | (U32(h[384]) << 16));
        }
    }

    #pragma unroll
    for (int idx = tid; idx < 512; idx += threads) {
        int row=idx>>4, col=(idx&15)*8;
        *reinterpret_cast<uint4*>(&basis[row][col]) =
            *reinterpret_cast<const uint4*>(ut + head*4096 + row*128 + col);
    }
    __syncthreads();
    float groups[3] = {};
    // Each stripe visits four neighboring tiles before advancing to its
    // next group. This keeps adjacent heads on matching sequence ranges.
    constexpr int group_tiles = 4;
    const auto tile_for = [=](int i) {
        return stripe * group_tiles + (i / group_tiles) * stripes_per_head * group_tiles + i % group_tiles;
    };
    uint2 next_rows[8][2];
    load_rows(src,shape.sequence,tile_for(0)*128+warp*16,head,next_rows);
    for (int item = 0; tile_for(item) < shape.tiles; ++item) {
        const int tile = tile_for(item);
        uint2 rows[8][2];
        #pragma unroll
        for (int i=0;i<8;++i) {
            rows[i][0]=next_rows[i][0];
            rows[i][1]=next_rows[i][1];
        }
        if (tile_for(item + 1) < shape.tiles)
            load_rows(src,shape.sequence,tile_for(item + 1)*128+warp*16,head,next_rows);
        float y[8][2][4];
        transform(rows,rotation,y);
        encode_item(y,Key,basis,stage[warp],groups);
        store_staged(shape,stage[warp],head,tile,Key,q4,img,prov,qsf);
    }
    #pragma unroll
    for (int j=0;j<3;++j) {
        const float v = comfy::warp_reduce_fmax(groups[j]);
        if (lane==0) red[warp][j]=v;
    }
    __syncthreads();
    if (tid<3) {
        float v=0;
        #pragma unroll
        for (int w=0;w<8;++w) v=fmaxf(v,red[w][tid]);
        const unsigned rounded = __float_as_uint(v) >> 16;
        atomicMax(maxima + (Key ? 3 : 0) + tid, rounded);
    }
}

__global__ __launch_bounds__(threads) void repair_v_and_pack_qk(
        PrepGeometry shape, InputView q, InputView k, InputView v, const U16* had, PrepBuffers out) {
    // Eight warps own the eight V groups in a tile. Q/K and V populate
    // disjoint bytes in the same output image.
    for (int unit = blockIdx.x; unit < shape.heads * shape.tiles; unit += gridDim.x)
        v_pack::finish_grouped_tile(shape, v, out.image, out.chosen_scales, out.v4_scale, unit);

    __shared__ __align__(16) U16 basis[32][136];
    __shared__ WarpStage stage[8];
    __shared__ float red[8][4];
    if ((blockIdx.x / shape.heads) % 2 == 0)
        process_head<false>(shape,q,had,out.ut,out.q4,out.image,out.side,out.maxima,out.q_sf,basis,stage,red);
    else
        process_head<true>(shape,k,had,out.ut,out.q4,out.image,out.side,out.maxima,out.q_sf,basis,stage,red);
}

__device__ __forceinline__ float scaled_pow2(float x) {
    unsigned bits = __float_as_uint(fmaxf(__fmul_rn(x, 1.f / 448.f), 1e-30f));
    return __uint_as_float((bits + 0x7fffffu) & 0x7f800000u);
}

__device__ __forceinline__ U32 pack_e4m3(U32 a, U32 b, float db, float dc) {
    float x0 = __uint_as_float(a << 16) * db;
    float x1 = __uint_as_float(a & 0xffff0000u) * db;
    float x2 = __uint_as_float(b << 16) * dc;
    float x3 = __uint_as_float(b & 0xffff0000u) * dc;
    return __nv_fp8x4_e4m3(make_float4(x0, x1, x2, x3)).__x;
}

__device__ __forceinline__ U32 common_upper_bytes(U32 middle) {
    return 0x40404040u - ((middle >> 3) & 0x10101010u);
}

__device__ __forceinline__ U32 upper_from_nibbles(U32 nibbles) {
    const U32 repeated = __byte_perm(nibbles, 0u, 0x1100);
    return (repeated & 0xf000f000u) | ((repeated << 4) & 0x00f000f0u);
}

__device__ __forceinline__ uint2 expand_bytes(U32 middle, U32 upper) {
    const U32 high = upper | ((middle >> 4) & 0x0f0f0f0fu);
    const U32 low = (middle << 4) & 0xf0f0f0f0u;
    return make_uint2(__byte_perm(high, low, 0x1504), __byte_perm(high, low, 0x3726));
}

// Reconstruct one group of four BF16 provisional scales using the exact
// exception certificate from the producer. The final MMA layout puts rows
// r and r+32 in adjacent four-byte words of the same eight-byte slot.
__device__ __forceinline__ uint2 decode_inplace_group(U32 middle, int group,
                                                       const ScaleSidePlane& side) {
    U32 upper = common_upper_bytes(middle);
    const int warp = (group & 127) >> 4;
    if (side.exceptional[warp][(group >> 7) == 2])
        upper = upper_from_nibbles(side.upper_nibbles[group]);
    return expand_bytes(middle,upper);
}

__device__ __forceinline__ uint2 encode_inplace(const U8* dst, const ScaleSidePlane& side,
                                                int chunk, float db, float dc) {
    // Output chunk j consists of four scales from row r and four from r+32.
    const int g = (chunk / 64) * 128 + (chunk % 64) / 2 + 64*(chunk & 1);
    // Each lane reads and updates just its own eight bytes.
    const uint2 bytes = reinterpret_cast<const uint2*>(dst)[chunk];
    const uint2 a = decode_inplace_group(bytes.x,g,side);
    const uint2 b = decode_inplace_group(bytes.y,g+32,side);
    return make_uint2(pack_e4m3(a.x,a.y,db,dc),pack_e4m3(b.x,b.y,db,dc));
}

// Every producer CTA publishes three nonnegative BF16 maxima into six words.
// The kernel boundary makes the completed maxima visible to all encoder CTAs.
// Lanes 0..2 derive each query scale and lanes 4..6 derive its key partner.
__device__ __forceinline__ float encoder_factor(const U32* maxima, float* score_scale, bool key_side) {
    const int lane = threadIdx.x & 31;
    const int axis = min(lane & 3, 2);
    const float scale = scaled_pow2(__uint_as_float(maxima[axis + ((lane & 4) ? 3 : 0)] << 16));
    const float query = __shfl_sync(~0u,scale,axis);
    const float key = __shfl_sync(~0u,scale,4+axis);
    float product = query * key;
    product = fmaxf(product,__shfl_xor_sync(~0u,product,1));
    product = fmaxf(product,__shfl_xor_sync(~0u,product,2));
    const int gap = 128 - int(__float_as_uint(product) >> 23);
    const float up = product < 2.f ? __uint_as_float(U32(127+((gap+1)>>1))<<23) : 1.f;
    product = fmaxf(product,2.f);
    if (threadIdx.x == 0 && !blockIdx.x && !blockIdx.y) *score_scale = product * (1.f/128.f);
    const float gq = query * up;
    return key_side ? gq * __frcp_rn(product) : __frcp_rn(gq);
}

__global__ void encode_scales(PrepGeometry shape, PrepBuffers out) {
    const bool key = blockIdx.y != 0;
    const float f = encoder_factor(out.maxima,out.score_scale,key);
    float side[3];
    #pragma unroll
    for (int j=0;j<3;++j) side[j] = __shfl_sync(~0u,f,j);
    const int units = shape.heads*shape.tiles;
    for (int tile=blockIdx.x; tile<units; tile+=gridDim.x) {
        U8* dst=key ? out.image+size_t(tile)*layout::tile_bytes+layout::key_scale_offset : out.q_sf+tile*layout::query_scale_bytes;
        const int chunk=threadIdx.x;
        const float db=chunk<128 ? side[0] : side[1];
        const float dc=chunk<128 ? side[0] : side[2];
        const auto& metadata = reinterpret_cast<const ScaleSidePlane*>(out.side)[(key ? units : 0) + tile];
        reinterpret_cast<uint2*>(dst)[chunk] = encode_inplace(dst, metadata, chunk, db, dc);
    }
}
} // namespace

PrepLayout prep_layout(int sequence, int heads, int sms) {
    const size_t units = size_t(heads) * ((sequence + 127) / 128);
    size_t end = 0;
    auto reserve = [&](size_t bytes) {
        const size_t start = (end + 255) & ~size_t(255);
        end = start + bytes;
        return start;
    };
    PrepLayout w{};
    w.q4 = reserve(size_t(sequence) * heads * layout::query_row_bytes);
    w.q_sf = reserve(units * layout::query_scale_bytes);
    w.img = reserve(units * layout::tile_bytes);
    w.score_scale = reserve(sizeof(float));
    w.ut = reserve(size_t(heads) * 32 * 128 * sizeof(uint16_t));
    w.side = reserve(2 * units * layout::scale_side_bytes);
    w.chosen_scales = reserve(units * 8 * sizeof(float));
    w.split = reserve((units % sms) * (layout::split_pair_bytes + sizeof(uint32_t)));
    // The core's split counters follow its partial records. Group all reductions
    // at the end so one stream-ordered memset initializes each invocation.
    w.v4_scale = reserve(sizeof(float));
    w.maxima = reserve(6 * sizeof(uint32_t));
    w.bytes = reserve(0);
    return w;
}

cudaError_t prepare(const PrepLayout& w, const void* q, const void* k, const void* v, const int32_t* rows,
                    const void* rotation, void* workspace, int sequence, int heads, int sms,
                    int64_t qt, int64_t qh, int64_t kt, int64_t kh, int64_t vt, int64_t vh,
                    cudaStream_t stream) {
    auto* p = static_cast<unsigned char*>(workspace);
    PrepBuffers out{p + w.q4, p + w.q_sf, p + w.img,
                    reinterpret_cast<uint16_t*>(p + w.ut), reinterpret_cast<uint16_t*>(p + w.side),
                    reinterpret_cast<float*>(p + w.score_scale), reinterpret_cast<float*>(p + w.v4_scale),
                    reinterpret_cast<float*>(p + w.chosen_scales), reinterpret_cast<uint32_t*>(p + w.maxima)};
    const PrepGeometry shape{sequence, heads, (sequence + 127) / 128};
    const InputView query{static_cast<const char*>(q), qt, qh};
    const InputView key{static_cast<const char*>(k), kt, kh};
    const InputView value{static_cast<const char*>(v), vt, vh};
    const int splits = heads * shape.tiles % sms;
    const size_t counters = w.split + splits * layout::split_pair_bytes;
    auto err = cudaMemsetAsync(p + counters, 0, w.bytes - counters, stream);
    if (err != cudaSuccess) return err;
    err = basis_and_v(shape, query, key, value, rows, rotation, out, sms, stream);
    if (err != cudaSuccess) return err;
    repair_v_and_pack_qk<<<2 * stripes_per_head * shape.heads, threads, 0, stream>>>(
        shape, query, key, value, static_cast<const U16*>(rotation), out);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    encode_scales<<<dim3(4 * sms, 2), 192, 0, stream>>>(shape, out);
    return cudaGetLastError();
}
} // namespace comfy_deco
