// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <stdint.h>

#pragma once
#include "prep_internal.cuh"
#include "../float_utils.cuh"
#include "../sage_attention/math.cuh"

namespace comfy_deco::v_pack {

__device__ __forceinline__ float low_bf16(uint32_t word) { return __uint_as_float(word << 16); }
__device__ __forceinline__ float high_bf16(uint32_t word) { return __uint_as_float(word & 0xffff0000u); }
__device__ __forceinline__ void accumulate(float &lo, float &hi, uint32_t x) {
    const float a = low_bf16(x), b = high_bf16(x);
    asm("{ .reg .b64 v,w; mov.b64 v,{%0,%1}; mov.b64 w,{%2,%3}; "
        "add.rn.f32x2 v,v,w; mov.b64 {%0,%1},v; }"
        : "+f"(lo), "+f"(hi) : "f"(a), "f"(b));
}
__device__ __forceinline__ void extrema(uint32_t &mn, uint32_t &mx, uint32_t x) {
    asm("min.bf16x2 %0,%1,%2;" : "=r"(mn) : "r"(mn), "r"(x));
    asm("max.bf16x2 %0,%1,%2;" : "=r"(mx) : "r"(mx), "r"(x));
}
__device__ __forceinline__ unsigned warp_max(float value) {
    return __reduce_max_sync(0xffffffffu, __float_as_uint(value));
}
constexpr int slots_per_sm = 4;

__device__ __forceinline__ float tensor_scale(float maximum) {
    const float required=fmaxf(1.e-30f,__fdiv_rn(maximum,2688.f));
    return __uint_as_float((__float_as_uint(required)+0x7fffffu)&0x7f800000u);
}

__device__ __forceinline__ float quantization_multiplier(float dev, float inv, unsigned char& code) {
    const __nv_fp8_e4m3 encoded(__fdiv_rn(dev*inv,6.0f));
    code = encoded.__x;
    const float sf = float(encoded);
    return (sf>0.f ? math::ptx_rcp(sf) : 1.0f)*inv;
}
__device__ __forceinline__ unsigned char* mean_address(unsigned char* dest, int tile, int tiles, int group, int ch) {
    // Adjacent even/odd tiles share the next odd tile's 4 KiB mean region.
    // The unpaired final tile owns its mean region and explicitly zeros its pad.
    return dest+layout::mean_offset+group*256+((ch/8)^group)*16+(ch%8)*2+
           ((tile&1) ? 2048 : (tile<tiles-1 ? layout::tile_bytes : 0));
}
__device__ __forceinline__ uint2* code_address(unsigned char* dest,int group,int ch) {
    return reinterpret_cast<uint2*>(dest+layout::value_offset)+2*((group/2)*128+ch)+(group%2);
}
__device__ __forceinline__ int scale_offset(int ch,int group) {
    const int u=ch/32,z=ch%32;
    return (u/2)*512+(4*(z%8)+z/8)*16+(u%2)*8+group;
}
template<bool High,bool Tail>
__device__ __forceinline__ uint2 pack16(const uint32_t(&vals)[16], float m,float mul,int count) {
    uint2 packed;
    #pragma unroll
    for (int side=0;side<2;++side) {
        float x[8];
        #pragma unroll
        for(int k=0;k<8;++k) {
            int row=side*8+k;
            float v=High ? high_bf16(vals[row]) : low_bf16(vals[row]);
            x[k]=(!Tail || row<count) ? (v-m)*mul : 0.f;
        }
        const uint32_t lo = comfy::pack_fp4x4<false>(x[0], x[1], x[2], x[3]);
        const uint32_t hi = comfy::pack_fp4x4<false>(x[4], x[5], x[6], x[7]);
        (side == 0 ? packed.x : packed.y) = lo | (hi << 16);
    }
    return packed;
}

// Each warp owns four adjacent channels per lane and all sixteen key rows.
// Its local power-of-two bound is a valid tensor-scale candidate; an exact
// global comparison decides whether this group's already packed bytes survive.
template<bool Tail, bool FirstPass>
__device__ __forceinline__ float pack_warp_group(PrepGeometry shape, InputView v, unsigned char* img,
        float* chosen, int unit, float global_inv, unsigned char* staged_scales) {
    const int lane = threadIdx.x & 31;
    const int group = (threadIdx.x >> 5) & 7;
    const int ch = 4 * lane;
    const int head = unit / shape.tiles, tile = unit % shape.tiles;
    const int row = tile * 128 + group * 16;
    const int count = Tail ? min(max(shape.sequence-row,0),16) : 16;
    const char* src = v.row(min(shape.sequence-1,row), head) + ch*2;
    uint32_t values[2][16], mn[2]={0x7f807f80u,0x7f807f80u}, mx[2]={0xff80ff80u,0xff80ff80u};
    float sums[4]={};
    #pragma unroll
    for (int k=0;k<16;++k) {
        uint2 w=make_uint2(0,0);
        if (!Tail || k<count) {
            w=*reinterpret_cast<const uint2*>(src+k * v.token_bytes);
            extrema(mn[0],mx[0],w.x);
            extrema(mn[1],mx[1],w.y);
        }
        accumulate(sums[0],sums[1],w.x);
        accumulate(sums[2],sums[3],w.y);
        values[0][k]=w.x;
        values[1][k]=w.y;
    }
    float mean[4], dev[4];
    float bound=0.f;
    const float divisor = float(max(count, 1));
    #pragma unroll
    for(int p=0;p<2;++p) {
        uint32_t bf=comfy::pack_bf2(__fdiv_rn(sums[2*p], divisor), __fdiv_rn(sums[2*p+1], divisor));
        float m0=low_bf16(bf), m1=high_bf16(bf);
        mean[2*p]=m0;
        mean[2*p+1]=m1;
        float d0=fmaxf(0.f,fmaxf(low_bf16(mx[p])-m0,m0-low_bf16(mn[p])));
        float d1=fmaxf(0.f,fmaxf(high_bf16(mx[p])-m1,m1-high_bf16(mn[p])));
        dev[2*p]=d0;
        dev[2*p+1]=d1;
        if constexpr (FirstPass)
            bound=fmaxf(bound,fmaxf(fmaxf(d0,d1),fmaxf(fabsf(m0),fabsf(m1))*0x1p-90f));
    }
    float inv=global_inv;
    float local_max=0.f;
    if constexpr (FirstPass) {
        local_max=__uint_as_float(warp_max(bound));
        const float minimum = tensor_scale(0.f);
        const float exact = tensor_scale(local_max);
        // A group usually has a smaller peak than the full tensor. Allow one
        // exponent of headroom while retaining the unmodified bound for the
        // global atomic reduction. A negative final shift takes the BF16 path.
        const bool use_headroom = exact > minimum && isfinite(exact) && exact <= 0x1p126f;
        const float scale = use_headroom ? 2.f * exact : exact;
        if (lane==0) chosen[unit*8+group]=scale;
        inv=math::ptx_rcp(scale);
    }
    auto* dst=img+size_t(unit)*layout::tile_bytes;
    uint2 scaled;
    #pragma unroll
    for (int p=0;p<2;++p) {
        const float m0=mean[2*p],m1=mean[2*p+1];
        (p ? scaled.y : scaled.x)=comfy::pack_bf2(m0*inv,m1*inv);
        unsigned char sf0,sf1;
        float a=quantization_multiplier(dev[2*p],inv,sf0);
        float b=quantization_multiplier(dev[2*p+1],inv,sf1);
        int channel=ch+2*p;
        *code_address(dst,group,channel)=pack16<false,Tail>(values[p],m0,a,count);
        *code_address(dst,group,channel+1)=pack16<true,Tail>(values[p],m1,b,count);
        if constexpr (FirstPass) {
            staged_scales[channel*8+group]=sf0;
            staged_scales[(channel+1)*8+group]=sf1;
        } else {
            dst[layout::value_scale_offset+scale_offset(channel,group)]=sf0;
            dst[layout::value_scale_offset+scale_offset(channel+1,group)]=sf1;
        }
    }
    auto* mean_dst=mean_address(dst,tile,shape.tiles,group,ch);
    *reinterpret_cast<uint2*>(mean_dst)=scaled;
    if constexpr (Tail && FirstPass) {
        if ((tile & 1) == 0) *reinterpret_cast<uint2*>(mean_dst + 2048) = make_uint2(0,0);
    }
    return local_max;
}


__device__ __forceinline__ void prepack_cta(PrepGeometry shape, InputView v, PrepBuffers out,
                                           int slot, int slots, unsigned* sm) {
    const int half = threadIdx.x >> 8, t = threadIdx.x & 255;
    auto* sfs = reinterpret_cast<unsigned char*>(sm) + half * 1024;
    const int units = shape.heads * shape.tiles;
    float mx = 0.f;
    // Keep both half-blocks at every barrier, including an unpaired final tile.
    for (int pair = 2 * slot; pair < units; pair += 2 * slots) {
        const int i = pair + half;
        const int unit = (i % shape.heads) * shape.tiles + i / shape.heads;
        if (i < units) {
            float here = unit % shape.tiles == shape.tiles - 1
                ? pack_warp_group<true,true>(shape,v,out.image,out.chosen_scales,unit,0.f,sfs)
                : pack_warp_group<false,true>(shape,v,out.image,out.chosen_scales,unit,0.f,sfs);
            mx = fmaxf(mx, here);
        }
        __syncthreads();
        if (t < 128 && i < units) {
            auto* dst = out.image + size_t(unit)*layout::tile_bytes + layout::value_scale_offset + scale_offset(t,0);
            *reinterpret_cast<uint2*>(dst) = reinterpret_cast<const uint2*>(sfs)[t];
        }
        __syncthreads();
    }
    unsigned r = warp_max(mx);
    if (!(threadIdx.x & 31)) sm[threadIdx.x/32] = r;
    __syncthreads();
    if (threadIdx.x < 32) {
        r = __reduce_max_sync(0xffffffffu, threadIdx.x < 16 ? sm[threadIdx.x] : 0u);
        if (!threadIdx.x) atomicMax(reinterpret_cast<unsigned*>(out.v4_scale),
                                   __float_as_uint(tensor_scale(__uint_as_float(r))));
    }
}
__device__ __forceinline__ float decode_positive_e4m3(unsigned code) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<uint8_t>(code);
    return float(value);
}

// All accepted paths derive the final packed fields from the first-pass
// representation without touching the original keys. Ambiguous rounding
// boundaries return false before any store and use the reference packer.
__device__ __forceinline__ bool renormalize_lane(unsigned char* img,int unit,int tiles,int group,
                                                 float local,float global) {
    const int ch=(threadIdx.x&31)*4;
    unsigned char* tile=img+size_t(unit)*layout::tile_bytes;
    float local_inv=math::ptx_rcp(local), global_inv=math::ptx_rcp(global);
    if (!(local_inv>0.f && isfinite(local_inv) && global_inv>0.f && isfinite(global_inv))) return false;
    const int shift=int(__float_as_uint(global)>>23)-int(__float_as_uint(local)>>23);
    if (shift<=0) return false;
    auto* means=reinterpret_cast<uint2*>(mean_address(tile,unit%tiles,tiles,group,ch));
    uint2 old_mean=*means;
    float result_mean[4];
    unsigned char sf[4];
    bool zero[4];
    const float ratio=global_inv*local;
    #pragma unroll
    for(int j=0;j<4;++j) {
        unsigned m=(j<2 ? old_mean.x : old_mean.y) >> (16*(j&1));
        unsigned mag=m & 0x7fffu;
        // A normal BF16 mean scaled by a power of two has no rounding loss.
        // A stored signed zero remains signed zero as the scale decreases.
        // First-pass subnormals can double-round and require original input.
        if ((mag>0 && mag<128) || mag>=0x7f80u) return false;
        result_mean[j]=(low_bf16(m)*local)*global_inv;
        const unsigned code=tile[layout::value_scale_offset+scale_offset(ch+j,group)];
        if (code>=126) return false;
        const int old_exp=code>>3;
        if(old_exp-shift>=2) {
            // Both RNE cells and their neighbors stay in the E4M3 normal range.
            sf[j]=static_cast<unsigned char>(code-8*shift);
            float old_a=math::ptx_rcp(decode_positive_e4m3(code))*local_inv;
            float new_a=math::ptx_rcp(decode_positive_e4m3(sf[j]))*global_inv;
            if (!(isfinite(old_a) && __float_as_uint(old_a)==__float_as_uint(new_a))) return false;
            zero[j]=false;
        } else {
            // The next E4M3 value bounds the pre-rounding value from above.
            // Use margin below the subnormal midpoint. The centered values
            // also fall well below 0.25, the first FP4 rounding midpoint.
            if (!(decode_positive_e4m3(code+1)*ratio < 0x1p-12f)) return false;
            float old_a=code ? math::ptx_rcp(decode_positive_e4m3(code))*local_inv : local_inv;
            if (!(isfinite(old_a) && old_a>0.f)) return false;
            sf[j]=0;
            zero[j]=code!=0;
        }
    }
    *means=make_uint2(comfy::pack_bf2(result_mean[0],result_mean[1]),
                      comfy::pack_bf2(result_mean[2],result_mean[3]));
    #pragma unroll
    for(int j=0;j<4;++j) {
        tile[layout::value_scale_offset+scale_offset(ch+j,group)]=sf[j];
        if(zero[j]) {
            auto* ptr=code_address(tile,group,ch+j);
            auto bits=*ptr;
            *ptr=make_uint2(bits.x & 0x88888888u,bits.y & 0x88888888u);
        }
    }
    return true;
}

__device__ __forceinline__ void finish_grouped_tile(PrepGeometry shape, InputView v, unsigned char* img,
                                                      const float* chosen, const float* scale, int unit) {
    int group=threadIdx.x>>5;
    float global=*scale;
    float local=chosen[unit*8+group];
    if (local==global || renormalize_lane(img,unit,shape.tiles,group,local,global)) return;
    float inv=math::ptx_rcp(global);
    if(unit%shape.tiles==shape.tiles-1) pack_warp_group<true,false>(shape,v,img,nullptr,unit,inv,nullptr);
    else pack_warp_group<false,false>(shape,v,img,nullptr,unit,inv,nullptr);
}
} // namespace comfy_deco::v_pack
