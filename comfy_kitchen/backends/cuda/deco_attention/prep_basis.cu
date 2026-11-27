// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Native compensation eigensolver for sampled BF16 Q/K rows, D=128 and rank=32.
// One block computes a complete basis for a head, including the final Hadamard rotation.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "prep_v.cuh"
#include "../sage_attention/mma.cuh"

namespace comfy_deco {
namespace {
using bf16 = __nv_bfloat16;
constexpr int shared_bytes = 71424;
constexpr unsigned full_warp = 0xffffffffu;

// Keep the fused multiply-adds and explicit rounding points of the reference solver.
__device__ __forceinline__ float prod(float x, float y) { return __fmul_rn(x,y); }
__device__ __forceinline__ float plus(float x, float y) { return __fadd_rn(x,y); }
__device__ __forceinline__ float minus(float x, float y) { return __fsub_rn(x,y); }
__device__ __forceinline__ float addprod(float sum, float a, float b) { return __fmaf_rn(a,b,sum); }
__device__ __forceinline__ float subprod(float sum, float a, float b) { return __fmaf_rn(-a,b,sum); }
__device__ __forceinline__ float sqrt_approx(float x) {
    float v; asm("sqrt.approx.ftz.f32 %0,%1;":"=f"(v):"f"(x)); return v;
}
__device__ __forceinline__ float xor_sum(float x, int step) {
    return plus(x,__shfl_xor_sync(full_warp,x,step));
}
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int step=16; step; step/=2) x=xor_sum(x,step);
    return x;
}
__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return addprod(addprod(addprod(prod(a.y,b.y),a.x,b.x),a.z,b.z),a.w,b.w);
}
__device__ __forceinline__ float4 load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void store4(float* p, float4 v) {
    *reinterpret_cast<float4*>(p)=v;
}
__device__ __forceinline__ float4 subtract_vector(float4 v,float f,float4 u) {
    return make_float4(subprod(v.x,f,u.x),subprod(v.y,f,u.y),
                       subprod(v.z,f,u.z),subprod(v.w,f,u.w));
}
__device__ __forceinline__ float4 scale4(float4 v,float f) {
    return make_float4(prod(v.x,f),prod(v.y,f),prod(v.z,f),prod(v.w,f));
}

// The Gram fragment per warp is 16x64 from Q, then 16x64 from K.
__device__ __forceinline__ void sampled_gram(PrepGeometry shape, InputView q, InputView k,
        const int* rows, char* sm, float (&a)[32]) {
    int tid=threadIdx.x, lane=tid%32, warp=tid/32, mi=warp%8, nb=(warp/8)*8;
    int group=lane/4, pair=lane%4;
    bf16* stage=reinterpret_cast<bf16*>(sm);
    float acc[64]={};
    const int samples = max(shape.sequence / 256, 64);
    // Accumulate 48-row rounds in reference order. Each side has a separate
    // 128x128 Gram matrix, and only the final round is padded with zeros.
    for (int start = 0; start < samples; start += 48) {
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            int vec=tid+512*it, side=vec/(48*16);
            int row=(vec/16)%48, chunk=vec%16;
            uint4 bits{};
            if (start + row < samples) {
                int tok=min(shape.sequence - 1, max(0,rows[start + row]));
                const char* base = side ? k.row(tok,blockIdx.x) : q.row(tok,blockIdx.x);
                bits = reinterpret_cast<const uint4*>(base)[chunk];
            }
            *reinterpret_cast<uint4*>(stage+(side*48+row)*136+8*chunk)=bits;
        }
        __syncthreads();
        #pragma unroll
        for (int quad=0;quad<3;++quad) {
            #pragma unroll
            for (int side=0;side<2;++side) {
                const bf16* base=stage+(48*side+16*quad)*136;
                int oct=lane/8, r8=lane%8;
                uint32_t af[4];
                mma::ldmatrix_m8n8x4_trans(af, base+((oct/2)*8+r8)*136+16*mi+(oct%2)*8);
                #pragma unroll
                for (int blk=0;blk<4;++blk) {
                    uint32_t bf[4];
                    mma::ldmatrix_m8n8x4_trans(bf, base+((oct%2)*8+r8)*136+8*(nb+2*blk+oct/2));
                    mma::mma_bf16(acc+32*side+8*blk, af, bf);
                    mma::mma_bf16(acc+32*side+8*blk+4, af, bf+2);
                }
            }
        }
        __syncthreads();
    }
    float* red=reinterpret_cast<float*>(sm+70272);
    #pragma unroll
    for (int side=0;side<2;++side) {
        float trace=0;
        #pragma unroll
        for (int n=0;n<8;++n) {
            #pragma unroll
            for (int r=0;r<2;++r) {
                #pragma unroll
                for (int e=0;e<2;++e) {
                    if (16*mi+group+8*r==8*(nb+n)+2*pair+e)
                        trace=plus(trace,acc[side*32+n*4+2*r+e]);
                }
            }
        }
        trace=warp_sum(trace);
        if (lane==0) red[16*side+warp]=trace;
    }
    __syncthreads();
    float tq=0,tk=0;
    #pragma unroll
    for (int w=0;w<16;++w) { tq=plus(tq,red[w]); tk=plus(tk,red[16+w]); }
    float iq=tq>0 ? __frcp_rn(tq) : 0, ik=tk>0 ? __frcp_rn(tk) : 0;
    #pragma unroll
    for (int i=0;i<32;++i) acc[i]=addprod(prod(acc[i+32],ik),acc[i],iq);
    float* tile=reinterpret_cast<float*>(sm+33792);
    #pragma unroll
    for (int half=0;half<2;++half) {
        __syncthreads();
        if (mi/4==half) {
            #pragma unroll
            for (int n=0;n<8;++n) {
                float* dst=tile+128*(16*mi+group-64*half)+8*(nb+n)+2*pair;
                dst[0]=acc[4*n]; dst[1]=acc[4*n+1];
                dst[8*128]=acc[4*n+2]; dst[8*128+1]=acc[4*n+3];
            }
        }
        __syncthreads();
        if ((tid/4)/64==half) {
            #pragma unroll
            for(int m=0;m<8;++m) {
                auto v=load4(tile+((tid/4)%64)*128+16*m+4*pair);
                a[4*m]=v.x; a[4*m+1]=v.y; a[4*m+2]=v.z; a[4*m+3]=v.w;
            }
        }
    }
}
__device__ __forceinline__ int reflector_offset(int k) {
    int a=k/4;
    return 512*a-8*a*(a-1)+(k%4)*(128-4*a);
}
struct Scratch {
    float *hh,*work,*u,*p,*d,*e,*e2,*lam,*column,*red,*scale;
    __device__ explicit Scratch(char* base)
        :hh(reinterpret_cast<float*>(base)),work(hh+33792/4),
         u(hh+66560/4),p(hh+67072/4),d(hh+67584/4),e(hh+68096/4),
         e2(hh+68608/4),lam(hh+69120/4),column(hh+69248/4),red(hh+70272/4),scale(hh+70976/4){}
};
template<int Seg>
__device__ __forceinline__ void tridiagonal_segment(Scratch s,float (&a)[32]) {
    int tid=threadIdx.x, warp=tid/32, rr=tid/4, qq=tid%4;
    #pragma unroll 1
    for (int k=Seg*16;k<min(Seg*16+16,126);++k) {
        if (warp==k/8) {
            float sq=0,x0=0,diag=0;
            #pragma unroll
            for (int m=Seg;m<8;++m) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int j=16*m+4*qq+e;
                    float val=a[4*m+e], x=(m==Seg && j<=k) ? 0 : val;
                    sq=addprod(sq,x,x);
                    if (m==Seg && j==k) diag=plus(diag,val);
                    if (m<=Seg+1 && j==k+1) x0=plus(x0,val);
                }
            }
            sq=xor_sum(xor_sum(sq,1),2);
            x0=xor_sum(xor_sum(x0,1),2);
            diag=xor_sum(xor_sum(diag,1),2);
            float norm=sqrt_approx(sq);
            float alpha=x0>=0 ? -norm:norm;
            bool active=subprod(sq,x0,x0)>prod(sq,1e-30f);
            float scale=active ? math::rsqrt(subprod(sq,x0,alpha)) : 0;
            if (!active) alpha=x0;
            if (rr==k) {
                #pragma unroll
                for (int m=Seg;m<8;++m) {
                    #pragma unroll
                    for (int e=0;e<4;++e) {
                        int j=16*m+4*qq+e;
                        float v=a[4*m+e]-(j==k+1 ? alpha : 0);
                        v=j>k ? prod(v,scale):0;
                        s.u[j]=v;
                        if (j>=k/4*4) s.hh[reflector_offset(k)+j-k/4*4]=v;
                    }
                }
                if (!qq) {s.d[k]=diag; s.e[k]=alpha;}
            }
        }
        __syncthreads();
        float uu[32],p=0,ur=0;
        if (warp>=2*Seg) {
            float l=0,r=0;
            #pragma unroll
            for(int m=Seg;m<8;++m) {
                float4 u=load4(s.u+16*m+4*qq);
                uu[4*m]=u.x; uu[4*m+1]=u.y; uu[4*m+2]=u.z; uu[4*m+3]=u.w;
                l=addprod(addprod(l,a[4*m],u.x),a[4*m+2],u.z);
                r=addprod(addprod(r,a[4*m+1],u.y),a[4*m+3],u.w);
            }
            float part=xor_sum(xor_sum(plus(l,r),1),2);
            p=rr>k ? part:0; ur=s.u[rr];
            float w=qq==0 ? prod(ur,p):0;
            w=xor_sum(xor_sum(xor_sum(w,4),8),16);
            if (!(tid%32)) s.red[warp]=w;
            if (!qq) s.p[rr]=p;
        }
        __syncthreads();
        if (warp>=2*Seg) {
            float dot=0;
            #pragma unroll
            for(int w=2*Seg;w<16;++w) dot=plus(dot,s.red[w]);
            dot=prod(dot,0.5f);
            float wr=subprod(p,dot,ur);
            #pragma unroll
            for (int m=Seg;m<8;++m) {
                #pragma unroll
                for (int e=0;e<4;++e) {
                    float uj=uu[4*m+e];
                    float wj=subprod(s.p[16*m+4*qq+e],dot,uj);
                    a[4*m+e]=subprod(subprod(a[4*m+e],ur,wj),wr,uj);
                }
            }
        }
    }
    if constexpr (Seg<7) tridiagonal_segment<Seg+1>(s,a);
}

__device__ __forceinline__ float init_spectrum(Scratch s,float &lo,float &hi) {
    int tid=threadIdx.x;
    if (tid<128) {
        float prev=tid ? fabsf(s.e[tid-1]):0;
        float next=tid<127 ? fabsf(s.e[tid]):0;
        s.e2[tid]=tid ? fmaxf(prod(prev,prev),1e-30f) : 0;
        float lower=minus(minus(s.d[tid],next),prev);
        float upper=plus(plus(s.d[tid],next),prev);
        #pragma unroll
        for(int n=16;n;n/=2) {
            lower=fminf(lower,__shfl_xor_sync(full_warp,lower,n));
            upper=fmaxf(upper,__shfl_xor_sync(full_warp,upper,n));
        }
        if (!(tid%32)) { s.red[tid/32]=lower; s.red[4+tid/32]=upper; }
    }
    __syncthreads();
    lo=fminf(fminf(s.red[0],s.red[1]),fminf(s.red[2],s.red[3]));
    hi=fmaxf(fmaxf(s.red[4],s.red[5]),fmaxf(s.red[6],s.red[7]));
    return minus(hi,lo);
}
__device__ __forceinline__ int sturm_count(Scratch s,float x) {
    float pc=1,pp=0;
    unsigned count=0;
    #pragma unroll 2
    for (int i=0;i<128;i+=4) {
        float4 dv=load4(s.d+i),ev=load4(s.e2+i);
        const float ds[4]={dv.x,dv.y,dv.z,dv.w};
        const float es[4]={ev.x,ev.y,ev.z,ev.w};
        #pragma unroll
        for (int j=0;j<4;++j) {
            float pn=__fmaf_rn(minus(ds[j],x),pc,-prod(es[j],pp));
            count+=(__float_as_uint(pn)^__float_as_uint(pc))>>31;
            pp=pc;pc=pn;
        }
        unsigned mx=__float_as_uint(fmaxf(fabsf(pc),fabsf(pp)));
        float scale=__uint_as_float(0x7f000000u-(mx&0x7f800000u));
        pc=prod(pc,scale); pp=prod(pp,scale);
    }
    return count;
}
__device__ __forceinline__ void eigenvalues(Scratch s,float lo,float hi,float span) {
    int tid=threadIdx.x, section=tid%16, target=128-tid/16;
    lo=minus(subprod(lo,span,1e-6f),1e-30f);
    hi=plus(addprod(hi,span,1e-6f),1e-30f);
    #pragma unroll 1
    for (int r=0;r<6;++r) {
        float width=prod(minus(hi,lo),1.f/17.f);
        auto place=[&](int n){return addprod(lo,width,float(n));};
        float x=place(section+1);
        unsigned mask=__ballot_sync(full_warp,sturm_count(s,x)>=target);
        mask=(mask>>((tid%32)/16*16))&0xffffu;
        int first=__popc(~mask&(mask-1));
        float lower=mask ? (first>0 ? place(first):lo):place(16);
        float upper=mask ? place(first+1):hi;
        lo=lower; hi=upper;
    }
    if (!section) s.lam[tid/16]=prod(plus(lo,hi),0.5f);
}
__device__ __forceinline__ float hash_start(int i,int key) {
    unsigned h=unsigned(i+1)^(unsigned(key)*0x9e3779b9u);
    h=(h^(h>>16))*0x7feb352du;
    h=(h^(h>>15))*0x846ca68bu;
    h^=h>>16;
    return minus(prod(float(h>>8),0x1p-24f),0.5f);
}
__device__ __forceinline__ void inverse_iteration(Scratch s,float span) {
    int c=threadIdx.x;
    if (c>=32) return;
    float lam=s.lam[c],pfl=addprod(1e-30f,span,1e-9f);
    auto floor_pivot=[&](float x){return fabsf(x)<pfl ? (x<0 ? -pfl:pfl):x;};
    float* z=s.work+c;
    float* pivots=z+128*32;
    #pragma unroll 4
    for (int i=0;i<128;++i) z[32*i]=hash_start(i,c+7);
    float scale=1;
    #pragma unroll 1
    for (int iter=0;iter<2;++iter) {
        float piv=floor_pivot(minus(s.d[0],lam));
        pivots[0]=piv;
        float y=prod(z[0],scale);
        z[0]=y;
        #pragma unroll 4
        for(int i=1;i<128;++i) {
            float e=s.e[i-1];
            float mult=prod(e,math::ptx_rcp(piv));
            piv=floor_pivot(subprod(minus(s.d[i],lam),mult,e));
            y=iter==0 ? subprod(z[32*i],mult,y) : __fmaf_rn(z[32*i],scale,-prod(mult,y));
            pivots[32*i]=piv; z[32*i]=y;
        }
        float v=prod(y,math::ptx_rcp(piv));
        z[127*32]=v;
        float mx=fabsf(v);
        #pragma unroll 4
        for (int i=126;i>=0;--i) {
            v=prod(subprod(z[32*i],s.e[i],v),math::ptx_rcp(pivots[32*i]));
            z[32*i]=v; mx=fmaxf(mx,fabsf(v));
        }
        scale=mx>0 ? math::ptx_rcp(mx):1;
    }
    s.scale[c]=scale;
}
__device__ __forceinline__ void orthogonalize(Scratch s) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    float va[4],vb[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const float* row=s.work+(4*lane+j)*32+2*warp;
        va[j]=prod(row[0],s.scale[2*warp]);
        vb[j]=prod(row[1],s.scale[2*warp+1]);
    }
    float4 a=make_float4(va[0],va[1],va[2],va[3]);
    float4 b=make_float4(vb[0],vb[1],vb[2],vb[3]);
    #pragma unroll 1
    for (int c=0;c<32;++c) {
        float* buf=s.column+(c%2)*128+4*lane;
        if (warp==c/2) {
            float4 v=c%2 ? b:a;
            v=scale4(v,math::rsqrt(fmaxf(warp_sum(dot4(v,v)),1e-30f)));
            if (c%2) b=v; else a=v;
            store4(buf,v);
        }
        __syncthreads();
        float4 v=load4(buf);
        float da=warp_sum(dot4(v,a)),db=warp_sum(dot4(v,b));
        a=subtract_vector(a,2*warp>c ? da:0,v);
        b=subtract_vector(b,2*warp+1>c ? db:0,v);
    }
    #pragma unroll 1
    for (int k=125;k>=0;--k) {
        int off=4*lane-k/4*4;
        float4 u=off>=0 ? load4(s.hh+reflector_offset(k)+off) : make_float4(0,0,0,0);
        a=subtract_vector(a,warp_sum(dot4(u,a)),u);
        b=subtract_vector(b,warp_sum(dot4(u,b)),u);
    }
    __syncthreads();
    float* col=s.work+4*lane*32+2*warp;
    col[0]=a.x;col[1]=b.x;col[32]=a.y;col[33]=b.y;
    col[64]=a.z;col[65]=b.z;col[96]=a.w;col[97]=b.w;
}
__global__ __launch_bounds__(512,1) void basis_kernel(PrepGeometry shape, InputView q, InputView k,
        InputView v, const int* rows, const bf16* had, PrepBuffers out, int slots) {
    extern __shared__ __align__(128) char sm[];
    if (blockIdx.x >= shape.heads) {
        v_pack::prepack_cta(shape,v,out,blockIdx.x - shape.heads,slots,reinterpret_cast<uint32_t*>(sm));
        return;
    }
    Scratch s(sm);
    float a[32];
    sampled_gram(shape,q,k,rows,sm,a);
    tridiagonal_segment<0>(s,a);
    int tid=threadIdx.x,rr=tid/4;
    if ((tid%4)==3 && rr>=126) {
        if (rr==126) {s.d[126]=a[30];s.e[126]=a[31];}
        else s.d[127]=a[31];
    }
    __syncthreads();
    float lo,hi;
    float span=init_spectrum(s,lo,hi);
    eigenvalues(s,lo,hi,span);
    __syncthreads();
    inverse_iteration(s,span);
    __syncthreads();
    orthogonalize(s);
    // Each thread projects one Hadamard column against eight eigenvectors.
    auto* h=reinterpret_cast<bf16*>(sm);
    #pragma unroll
    for (int i=tid;i<2048;i+=512) {
        reinterpret_cast<uint4*>(h)[i]=reinterpret_cast<const uint4*>(had)[i];
    }
    __syncthreads();
    int d=tid%128,rg=tid/128;
    float sums[8]={};
    #pragma unroll 4
    for(int j=0;j<128;++j) {
        float hv=__bfloat162float(h[j*128+d]);
        #pragma unroll
        for (int e=0;e<8;++e) sums[e]=addprod(sums[e],hv,s.work[j*32+8*rg+e]);
    }
    #pragma unroll
    for (int e=0;e<8;++e)
        reinterpret_cast<bf16*>(out.ut)[(blockIdx.x*32+8*rg+e)*128+d]=__float2bfloat16_rn(prod(sums[e],0.08838834764831844055f));
}
} // namespace

cudaError_t basis_and_v(PrepGeometry shape, InputView q, InputView k, InputView v,
                        const int32_t* rows, const void* rotation, PrepBuffers out,
                        int sms, cudaStream_t stream) {
    auto err = cudaFuncSetAttribute(basis_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (err != cudaSuccess) return err;
    const int slots = v_pack::slots_per_sm * sms;
    basis_kernel<<<shape.heads + slots, 512, shared_bytes, stream>>>(
        shape, q, k, v, rows, static_cast<const bf16*>(rotation), out, slots);
    return cudaGetLastError();
}
} // namespace comfy_deco
