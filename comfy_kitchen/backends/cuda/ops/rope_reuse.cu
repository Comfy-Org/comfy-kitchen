/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Apply-RoPE and RMSNorm+RoPE with rotation-matrix reuse across heads.
 */
#include "dtype_dispatch.cuh"
#include "rope_device.cuh"
#include "utils.cuh"

#include <cstdint>
#include <type_traits>

namespace comfy::rope_reuse {
inline int device_sm() {
  int current;
  CUDA_CHECK(cudaGetDevice(&current));
  static thread_local int cached_device = -1, sm = 0;
  if (current != cached_device) {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, current));
    sm = props.major * 10 + props.minor;
    cached_device = current;
  }
  return sm;
}
inline bool overlap(const void *a, const void *b, uint64_t bytes) {
  auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x <= y ? y - x < bytes : x - y < bytes;
}

// Reuse each general FP32 rotation matrix across up to 16 heads.
// Four input buffers hide load latency; FP32 RMS statistics and the
// input-dtype normalization materialization are preserved.
template <typename T,int V> union Packed {
  uint4 v4[(V + 7) / 8];
  uint64_t v2[(V + 3) / 4];
  std::conditional_t<std::is_same_v<T,half>,half2,__nv_bfloat162> h[V / 2];
};
template <int V,typename T>
__device__ __forceinline__ void storep(T *x, const Packed<T,V> &a) {
  if constexpr (V >= 8) {
#pragma unroll
    for (int j = 0; j < V / 8; j++)
      reinterpret_cast<uint4 *>(x)[j] = a.v4[j];
  } else
    *reinterpret_cast<uint64_t *>(x) = a.v2[0];
}
__device__ __forceinline__ float rsqrt_fast(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float2 expand2(__nv_bfloat162 x) {
  union {
    __nv_bfloat162 h;
    uint32_t u;
  } a;
  a.h = x;
  return make_float2(__uint_as_float(a.u << 16),
                     __uint_as_float(a.u & 0xffff0000u));
}
__device__ __forceinline__ float expand1(__nv_bfloat16 x) {
  union {
    __nv_bfloat16 h;
    uint16_t u;
  } a;
  a.h = x;
  return __uint_as_float(uint32_t(a.u) << 16);
}
__device__ __forceinline__ float2 expand2(half2 v){return __half22float2(v);}
__device__ __forceinline__ float expand1(half v){return __half2float(v);}
template<class T> __device__ __forceinline__ auto pack2(float a,float b){
 if constexpr(std::is_same_v<T,half>)return __floats2half2_rn(a,b);
 else return __floats2bfloat162_rn(a,b);
}
template <typename T,bool SPLIT>
__device__ __forceinline__ void readrow(Packed<T,8> &v, const T *x,
                                        uint64_t row, int lane, int D, bool valid) {
  if (!valid) {
    v.v4[0] = make_uint4(0, 0, 0, 0);
    return;
  }
  if constexpr (SPLIT) {
    v.v2[0] =
        *reinterpret_cast<const uint64_t *>(x + (uint64_t)row * D + lane * 4);
    v.v2[1] = *reinterpret_cast<const uint64_t *>(x + (uint64_t)row * D + D/2 +
                                                  lane * 4);
  } else
    v.v4[0] =
        *reinterpret_cast<const uint4 *>(x + (uint64_t)row * D + lane * 8);
}
template <typename T,bool SPLIT, bool WF32,bool RMS>
__global__ void
rope_reuse_kernel(const T *q, const T *k,
                   const float4 *__restrict__ f, const void *__restrict__ wq,
                   const void *__restrict__ wk, T *oq,
                   T *ok, int B, int H, int N, int D, bool bnhd, bool dual,
                   float eps) {
  constexpr int G = 16, STAGES = 4, V = 8, P = 4;
  int lane = threadIdx.x % 32;
  const int width=D/8, groups=32/width;
  const int head_groups=(dual&&width<32)?groups/2:groups;
  const uint64_t warp=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)>>5;
  const int duplicate=dual&&width==32;
  const int lc=lane%width;
  const bool isk=dual && (width==32 ? (warp&1) : ((lane/width)&1));
  const int head_group=dual ? (lane/width)/2 : lane/width;
  const T *x = isk ? k : q;
  T *y = isk ? ok : oq;
  const void *w = isk ? wk : wq;
  const uint64_t head_tiles = (H + G - 1) / G,
                 tiles = (uint64_t)B * (uint64_t)N * head_tiles;
  const bool pow2 = (N & (N - 1)) == 0;
  const int tsh = __ffs(N) - 1;
  for (uint64_t tile = warp >> duplicate;
       tile < tiles; tile += (uint64_t(gridDim.x) * (blockDim.x >> 5)) >> duplicate) {
    uint64_t token, bh, batch, hb;
    if (pow2) {
      token = tile & ((uint64_t)N - 1);
      bh = tile >> tsh;
    } else {
      bh = tile / (uint64_t)N;
      token = tile - bh * (uint64_t)N;
    }
    if (B == 1) {
      batch = 0;
      hb = bh * G;
    } else if (head_tiles == 1) {
      batch = bh;
      hb = 0;
    } else {
      batch = bh / head_tiles;
      hb = (bh - batch * head_tiles) * G;
    }
    const int available=min(G,int(H-hb));
    const int iterations=(available+head_groups-1)/head_groups;
    hb += head_group*iterations;
    float4 ff[P];
    float2 ww[P];
#pragma unroll
    for (int p = 0; p < P; p++) {
      int pair = lc * P + p, c0 = SPLIT ? pair : pair * 2,
          c1 = SPLIT ? pair + D/2 : pair * 2 + 1;
      ff[p] = f[(uint64_t)token * (D/2) + pair];
      ww[p]=make_float2(1.f,1.f);
      if constexpr(RMS) {
      if constexpr (WF32)
        ww[p] = make_float2(((const float *)w)[c0], ((const float *)w)[c1]);
      else
        ww[p] = make_float2(expand1(((const T *)w)[c0]),
                            expand1(((const T *)w)[c1]));
      }
    }
    Packed<T,V> buf[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; s++) {
      int head = hb + s;
      uint64_t row = bnhd ? (batch * (uint64_t)N + token) * (uint64_t)H + head
                          : (batch * (uint64_t)H + head) * (uint64_t)N + token;
      readrow<T,SPLIT>(buf[s], x, row, lc, D, head < H && s < iterations);
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
      if(g>=iterations)break;
      int head = hb + g;
      bool valid = head < H;
      uint64_t row = bnhd ? (batch * (uint64_t)N + token) * (uint64_t)H + head
                          : (batch * (uint64_t)H + head) * (uint64_t)N + token;
      float2 xx[P];
      float sum = 0;
      if constexpr (SPLIT) {
#pragma unroll
        for (int j = 0; j < P / 2; j++) {
          float2 l = expand2(buf[g % STAGES].h[j]),
                 h = expand2(buf[g % STAGES].h[j + P / 2]);
          xx[2 * j] = make_float2(l.x, h.x);
          xx[2 * j + 1] = make_float2(l.y, h.y);
        }
      } else {
#pragma unroll
        for (int p = 0; p < P; p++)
          xx[p] = expand2(buf[g % STAGES].h[p]);
      }
      if constexpr (STAGES < G) {
        if (g + STAGES < iterations) {
          int fh = hb + g + STAGES;
          uint64_t fr = bnhd ? (batch * (uint64_t)N + token) * (uint64_t)H + fh
                             : (batch * (uint64_t)H + fh) * (uint64_t)N + token;
          readrow<T,SPLIT>(buf[g % STAGES], x, fr, lc, D, fh < H);
        }
      }
#pragma unroll
      for (int p = 0; p < P; p++)
        sum += xx[p].x * xx[p].x + xx[p].y * xx[p].y;
#pragma unroll
      for (int delta = width/2; delta; delta /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, delta, width);
      float inv = RMS ? rsqrt_fast(sum * (1.f / D) + eps) : 1.f;
      Packed<T,V> out;
      Packed<T,P> lo, hi;
      if constexpr (SPLIT) {
#pragma unroll
        for (int j = 0; j < P / 2; j++) {
          int p = 2 * j;
          float4 a = ff[p], b = ff[p + 1];
          float2 n = expand2(pack2<T>(xx[p].x * inv * ww[p].x,
                                                   xx[p].y * inv * ww[p].y));
          float2 m =
              expand2(pack2<T>(xx[p + 1].x * inv * ww[p + 1].x,
                                            xx[p + 1].y * inv * ww[p + 1].y));
          float a0 = fmaf(a.y, n.y, a.x * n.x), a1 = fmaf(a.z, n.x, a.w * n.y),
                b0 = fmaf(b.y, m.y, b.x * m.x), b1 = fmaf(b.z, m.x, b.w * m.y);
          lo.h[j] = pack2<T>(a0, b0);
          hi.h[j] = pack2<T>(a1, b1);
        }
      } else {
#pragma unroll
        for (int p = 0; p < P; p++) {
          float4 a = ff[p];
          float2 n = expand2(pack2<T>(xx[p].x * inv * ww[p].x,
                                                   xx[p].y * inv * ww[p].y));
          out.h[p] = pack2<T>(fmaf(a.y, n.y, a.x * n.x),
                                           fmaf(a.z, n.x, a.w * n.y));
        }
      }
      if (valid) {
        if constexpr (SPLIT) {
          storep<P>(y + (uint64_t)row * D + lc * P, lo);
          storep<P>(y + (uint64_t)row * D + D/2 + lc * P, hi);
        } else
          storep<V>(y + (uint64_t)row * D + lc * V, out);
      }
    }
  }
}

template <typename T,bool Split, bool FP32Weights,bool RMS>
inline void
launch_rope_reuse(const void *q, const void *k, const void *freqs,
                          const void *wq, const void *wk, void *oq, void *ok,
                          int batch, int heads, int tokens, int D, bool bnhd, bool dual,
                          float epsilon, cudaStream_t stream) {
  const int threads = Split && !FP32Weights ? 64 : 128;
  const int tiles = batch * tokens * ((heads + 15) / 16);
  const int warps=tiles*((D==256&&dual)?2:1);
  const int blocks = (warps + threads / 32 - 1) / (threads / 32);
  rope_reuse_kernel<T,Split, FP32Weights,RMS><<<blocks, threads, 0, stream>>>(
      static_cast<const T *>(q),
      static_cast<const T *>(k), static_cast<const float4 *>(freqs),
      wq, wk, static_cast<T *>(oq),
      static_cast<T *>(ok), batch, heads, tokens, D, bnhd, dual, epsilon);
}

inline bool try_rope_reuse(
    const void *q, const void *k, const void *f, const void *wq,
    const void *wk, void *oq, void *ok, int64_t B, int64_t d1, int64_t d2,
    int64_t D, int64_t rot, int64_t fb, int64_t f1, int64_t f2,
    const int64_t *qs, const int64_t *ks, const int64_t *os,
    const int64_t *ps, const int64_t *fs, int64_t wqs, int64_t wks,
    float epsilon, int dtype, int ftype, int wtype, bool dual, bool split,
    bool rms, cudaStream_t stream) {
  if ((dtype != 1 && dtype != 2) || ftype != 0 || fb != 1 || rot != D ||
      (D != 64 && D != 128 && D != 256) || B < 1 || B > 2 ||
      d1 < 1 || d2 < 1 || d1 > INT32_MAX || d2 > INT32_MAX ||
      B > INT32_MAX / d1 / d2 || B*d1*d2*D < 1048576 ||
      (rms && ((wtype != 0 && wtype != dtype) || wqs != 1 || (dual && wks != 1))))
    return false;
  bool bnhd;
  int64_t H, N;
  if (f1 == 1 && f2 == d2) { bnhd=false; H=d1; N=d2; }
  else if (f2 == 1 && f1 == d1) { bnhd=true; H=d2; N=d1; }
  else return false;
  if (H > 32 || N < 1024 || N > 262144 || fs[3] != 4 || fs[4] != 2 ||
      fs[5] != 1 || fs[bnhd ? 1 : 2] != D*2)
    return false;
  const int64_t strides[] = {d1*d2*D,d2*D,D,1};
  for (int i=0;i<4;++i)
    if (qs[i]!=strides[i] || os[i]!=strides[i] ||
        (dual && (ks[i]!=strides[i] || ps[i]!=strides[i]))) return false;
  if (!comfy::optimized::aligned_to(q,16) || !comfy::optimized::aligned_to(oq,16) || !comfy::optimized::aligned_to(f,16) ||
      (dual && (!comfy::optimized::aligned_to(k,16) || !comfy::optimized::aligned_to(ok,16)))) return false;
  const uint64_t bytes=uint64_t(B)*d1*d2*D*2;
  if ((q!=oq && overlap(q,oq,bytes)) ||
      (dual && ((k!=ok && overlap(k,ok,bytes)) || overlap(q,ok,bytes) ||
                overlap(k,oq,bytes) || overlap(oq,ok,bytes)))) return false;
  const int sm=device_sm();
  if (sm!=90 && sm!=100 && sm!=103) return false;
  // Preserve the existing BF16 specialization where it already wins.
  if (rms && dtype==2 && dual && D==128 && q!=oq && k!=ok &&
      (H==8 || H==16 || H==32) && N<=32768 && !(N&(N-1)) &&
      B*H*N>=32768 && (sm==90 || (sm==103 && !bnhd))) return false;
#define REUSE(T,S,W,R) launch_rope_reuse<T,S,W,R>(q,k,f,wq,wk,oq,ok,B,H,N,D,bnhd,dual,epsilon,stream)
#define MODE(T,S) if(rms) { if(wtype==0) { REUSE(T,S,true,true); } else { REUSE(T,S,false,true); } } else { REUSE(T,S,true,false); }
  if(dtype==1) { if(split) { MODE(half,true) } else { MODE(half,false) } }
  else { if(split) { MODE(nv_bfloat16,true) } else { MODE(nv_bfloat16,false) } }
#undef MODE
#undef REUSE
  CUDA_CHECK(cudaGetLastError());
  return true;
}

} // namespace comfy::rope_reuse

extern "C" void launch_apply_rope_kernel(
    const void *q, const void *k, const void *freqs, void *q_out, void *k_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2,
    int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0,
    int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1,
    int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2,
    int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3,
    int64_t f_s4, int64_t f_s5, int input_dtype_code, int freqs_dtype_code,
    bool has_k, bool split_half, cudaStream_t stream);
extern "C" void launch_rms_rope_kernel(
    const void *q, const void *k, const void *freqs, const void *q_scale,
    const void *k_scale, void *q_out, void *k_out, int64_t batch, int64_t dim1,
    int64_t dim2, int64_t head_dim, int64_t rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1,
    int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0,
    int64_t qo_s1, int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1,
    int64_t ko_s2, int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2,
    int64_t f_s3, int64_t f_s4, int64_t f_s5, int64_t qs_stride,
    int64_t ks_stride, float epsilon, int input_dtype_code,
    int freqs_dtype_code, int scale_dtype_code, bool has_k, bool split_half,
    cudaStream_t stream);

extern "C" void launch_apply_rope_dispatch(
    const void *q, const void *k, const void *freqs, void *q_out, void *k_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2,
    int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0,
    int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1,
    int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2,
    int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3,
    int64_t f_s4, int64_t f_s5, int input_dtype_code, int freqs_dtype_code,
    bool has_k, bool split_half, cudaStream_t stream) {
  const int64_t qs[] = {q_s0,q_s1,q_s2,q_s3};
  const int64_t ks[] = {k_s0,k_s1,k_s2,k_s3};
  const int64_t os[] = {qo_s0,qo_s1,qo_s2,qo_s3};
  const int64_t ps[] = {ko_s0,ko_s1,ko_s2,ko_s3};
  const int64_t fs[] = {f_s0,f_s1,f_s2,f_s3,f_s4,f_s5};
  if (comfy::rope_reuse::try_rope_reuse(q,k,freqs,nullptr,nullptr,q_out,k_out,
      batch,dim1,dim2,head_dim,head_dim,freqs_batch,freqs_dim1,freqs_dim2,
      qs,ks,os,ps,fs,0,0,0.f,input_dtype_code,freqs_dtype_code,0,
      has_k,split_half,false,stream)) return;
  launch_apply_rope_kernel(
      q, k, freqs, q_out, k_out, batch,
      dim1, dim2, head_dim, freqs_batch, freqs_dim1, freqs_dim2,
      q_s0, q_s1, q_s2, q_s3, k_s0, k_s1,
      k_s2, k_s3, qo_s0, qo_s1, qo_s2, qo_s3,
      ko_s0, ko_s1, ko_s2, ko_s3, f_s0, f_s1,
      f_s2, f_s3, f_s4, f_s5, input_dtype_code, freqs_dtype_code,
      has_k, split_half, stream);
}
extern "C" void launch_rms_rope_dispatch(
    const void *q, const void *k, const void *freqs, const void *q_scale,
    const void *k_scale, void *q_out, void *k_out, int64_t batch, int64_t dim1,
    int64_t dim2, int64_t head_dim, int64_t rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1,
    int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0,
    int64_t qo_s1, int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1,
    int64_t ko_s2, int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2,
    int64_t f_s3, int64_t f_s4, int64_t f_s5, int64_t qs_stride,
    int64_t ks_stride, float epsilon, int input_dtype_code,
    int freqs_dtype_code, int scale_dtype_code, bool has_k, bool split_half,
    cudaStream_t stream) {
  const int64_t qs[] = {q_s0,q_s1,q_s2,q_s3};
  const int64_t ks[] = {k_s0,k_s1,k_s2,k_s3};
  const int64_t os[] = {qo_s0,qo_s1,qo_s2,qo_s3};
  const int64_t ps[] = {ko_s0,ko_s1,ko_s2,ko_s3};
  const int64_t fs[] = {f_s0,f_s1,f_s2,f_s3,f_s4,f_s5};
  if (comfy::rope_reuse::try_rope_reuse(q,k,freqs,q_scale,k_scale,q_out,k_out,
      batch,dim1,dim2,head_dim,rot_dim,freqs_batch,freqs_dim1,freqs_dim2,
      qs,ks,os,ps,fs,qs_stride,ks_stride,epsilon,input_dtype_code,
      freqs_dtype_code,scale_dtype_code,has_k,split_half,true,stream)) return;
  launch_rms_rope_kernel(
      q, k, freqs, q_scale, k_scale, q_out,
      k_out, batch, dim1, dim2, head_dim, rot_dim,
      freqs_batch, freqs_dim1, freqs_dim2, q_s0, q_s1, q_s2,
      q_s3, k_s0, k_s1, k_s2, k_s3, qo_s0,
      qo_s1, qo_s2, qo_s3, ko_s0, ko_s1, ko_s2,
      ko_s3, f_s0, f_s1, f_s2, f_s3, f_s4,
      f_s5, qs_stride, ks_stride, epsilon, input_dtype_code, freqs_dtype_code,
      scale_dtype_code, has_k, split_half, stream);
}
