// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Launchers called from more than one translation unit. They are extern "C", so a
// declaration that drifts from its definition still links and then corrupts the
// call frame; declaring them once where the definition can see it makes the
// compiler check instead.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>

// True on the small RDNA3 iGPU (Radeon 780M, gfx1103) that the kernel-tuning
// changes in this tree were measured on. dGPUs keep the upstream defaults:
// several block-size and dispatch choices tuned against 6 WGPs are a
// regression on 60-96 CU parts. Host-safe (no device intrinsics), so both the
// .hip kernels and the dlpack bindings can consult it.
//
// Resolved for the given device ordinal and cached per ordinal, not once per
// process against device 0 and not against the *current* device: a box with
// both an iGPU and a dGPU sees two architectures, and the kernels run on the
// stream of the tensor's device, so the gate must answer for that tensor's
// device rather than whichever is current. The query stays off the hot path the
// same way device_wgp_count keeps its own.
inline bool comfy_small_igpu(int device) {
    constexpr int kMaxDevices = 16;
    // 0 unknown, 1 yes, 2 no. Racing threads write the same value and nothing
    // else is published through the cache, so relaxed.
    static std::atomic<int> cache[kMaxDevices] = {};
    if (device < 0 || device >= kMaxDevices) return false;
    int v = cache[device].load(std::memory_order_relaxed);
    if (v == 0) {
        hipDeviceProp_t prop{};
        v = (hipGetDeviceProperties(&prop, device) == hipSuccess &&
             std::strstr(prop.gcnArchName, "gfx1103") != nullptr)
                ? 1
                : 2;
        cache[device].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

// Current-device overload for the kernel-TU envelopes (hadamard, GEMM, rope,
// quant), which run on the device a launch stream was set current for. The
// bindings that can name the operand's device call the explicit-ordinal form.
inline bool comfy_small_igpu() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    return comfy_small_igpu(dev);
}

// True on RDNA2 (gfx103x), which has no matrix cores. The GEMM kernels use
// VALU v_dot4_i32_i8 and v_dot2_f32_f16 instead of WMMA on these devices.
// Runtime check: CMAKE_HIP_ARCHITECTURES compiles for all targets, so the
// host code never sees __gfx1035__; we must query the device at runtime.
inline bool comfy_is_gfx10(int device) {
    constexpr int kMaxDevices = 16;
    static std::atomic<int> cache[kMaxDevices] = {};
    if (device < 0 || device >= kMaxDevices) return false;
    int v = cache[device].load(std::memory_order_relaxed);
    if (v == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
            v = 2;
        } else {
            const char* n = prop.gcnArchName;
            v = (std::strstr(n, "gfx1030") || std::strstr(n, "gfx1031") ||
                 std::strstr(n, "gfx1032") || std::strstr(n, "gfx1033") ||
                 std::strstr(n, "gfx1034") || std::strstr(n, "gfx1035") ||
                 std::strstr(n, "gfx1036"))
                    ? 1
                    : 2;
        }
        cache[device].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

inline bool comfy_is_gfx10() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    return comfy_is_gfx10(dev);
}

extern "C" {

// Fused 3D neighborhood attention over contiguous (B, T, H, W, NH, HD) tensors.
// dtype_code is a DTYPE_TO_CODE value: 1 float16, 2 bfloat16. See ops/na3d.hip.
void launch_na3d_kernel(const void* q, const void* k, const void* v, void* out, int batch,
                        int t_size, int h_size, int w_size, int num_heads, int head_dim, int kt,
                        int kh, int kw, int causal_t, int causal_h, int causal_w, float scale,
                        int dtype_code, hipStream_t stream);

// BF16 decode attention over a fixed-capacity KV cache. query_length is the GQA
// group count folded into the query sequence dimension by the Python layer, and
// head_dim is 128 or 256. out_accum and lse_accum are read only when
// num_splits > 1. See ops/flash_decode.hip.
void launch_flash_decode(const void* q, const void* k, const void* v, const int* kv_lengths,
                         void* out, float* softmax_lse, float* out_accum, float* lse_accum,
                         int batch, int query_length, int heads, int head_dim, int kv_capacity, int num_splits,
                         int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
                         int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
                         hipStream_t stream);

// FP16/BF16 WMMA GEMMs over A[M,K] @ B[N,K]^T. Returns false when the shape is
// declined and the caller must serve it (torch fallback). bias/rscale/resid are
// fp16. See ops/gemm_fp16.hip.
bool launch_fp16_gemm_kernel(const void* a, const void* b, void* d, const void* bias,
                             const void* rscale, const void* resid, int M, int N, int K,
                             hipStream_t stream);
bool launch_bf16_gemm_kernel(const void* a, const void* b, void* d, const void* bias,
                             const void* rscale, const void* resid, int M, int N, int K,
                             hipStream_t stream);

// ldc is c's row stride, so a caller writing an N-column slice of a wider output
// passes that output's width; a whole GEMM passes N.
void launch_int8_gemm_kernel(const void* a, const void* b, void* c, const void* scale_a,
                             const void* scale_b, int scale_b_stride, const void* bias,
                             int bias_code, int M, int N, int K, int ldc, int out_code,
                             hipStream_t stream);
// Same shape as launch_int8_gemm_kernel, but a and b are signed int4 packed two
// per byte (low nibble = even k), so a row is K / 2 bytes wide and K must be a
// multiple of 32. gfx10 only: other targets get a trapping stub and the binding
// refuses before reaching it.
void launch_int4_gemm_kernel(const void* a, const void* b, void* c, const void* scale_a,
                             const void* scale_b, int scale_b_stride, const void* bias,
                             int bias_code, int M, int N, int K, int ldc, int out_code,
                             hipStream_t stream);
// scale_code is a DTYPE_TO_CODE value: 0 float32, 5 e4m3 (passed as raw bytes).
// codebook is 16 floats, or null for the uniform levels. bits is 4 or 6.
void launch_dequant_int4_grouped_to_int8_kernel(const void* qw, const void* s_rel, int scale_code,
                                                const void* codebook, void* out, int64_t n,
                                                int64_t k, int group_size, int bits,
                                                hipStream_t stream);

// weight is the raw [N, K] weight, in_dtype_code 1 float16 or 2 bfloat16. bits 4 takes
// the 16-entry codebook at g 16, bits 6 uniform levels at g 16/32/64. s_rel is written
// as raw e4m3 bytes; seed is ignored unless stochastic is set.
void launch_quantize_wxa8_convrot_fused_kernel(const void* weight, const void* codebook,
                                               void* packed, void* s_rel, void* s_channel,
                                               int64_t n, int64_t k, int bits, int g,
                                               int in_dtype_code, bool stochastic, uint64_t seed,
                                               hipStream_t stream);

// Staged 4-bit requantize of an already rotated [N, K] weight, in_dtype_code 0 float32,
// 1 float16 or 2 bfloat16. Throws when the K/16 group scales do not fit in LDS.
void launch_quantize_w4a8_convrot_kernel(const void* rotated, const void* codebook, void* packed,
                                         void* s_rel, void* s_channel, int64_t n, int64_t k,
                                         int in_dtype_code, bool stochastic, uint64_t seed,
                                         hipStream_t stream);

// Widest K the fused requantize can take at this group size on the current device,
// 0 if unknown.
int wxa8_requant_max_k_kernel(int group_size);

void launch_w4a8_int8_gemm_chunked_kernel(const void* xq, const void* qw, const void* s_rel,
                                          int scale_code, const void* codebook,
                                          const void* s_channel, const void* xs, const void* bias,
                                          int bias_code, void* workspace, void* out, int M, int N,
                                          int K, int group_size, int chunk_cols, int bits,
                                          int out_code, hipStream_t stream);

// V unquantized and transposed to the packed [B*H*D, padded_N] layout the direct
// short-key kernel reads. Defined in sage_attention/quant_v_int8.hip.
void launch_sage_transpose_v(const void* v, void* out, int B, int H, int N, int D, int padded_N,
                             int64_t stride_b, int64_t stride_h, int64_t stride_n,
                             int input_dtype_code, hipStream_t stream);

// Direct fp16/bf16 attention (no int8 quantization) for short keys, mirroring
// the reference library's use_direct path. v is the fp16-transposed buffer.
void launch_sage_direct_attn(const void* q, const void* k, const void* v, void* o,
                             int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
                             int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
                             int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d,
                             int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
                             int batch, int qo_len, int kv_len, int num_qo_heads,
                             int num_kv_heads, int head_dim, float sm_scale, int dtype_code,
                             hipStream_t stream);

// Sol-Attn sparse attention -- see sage_attention/sol_attn.hip. The whole pipeline
// runs over one caller-allocated workspace whose carve-up sol_attn_plan reports.
extern const char* const sol_attn_plan_names[];  // null-terminated
int sol_attn_plan(int batch, int seq_len, int num_heads, int n_tok, int64_t* out, int cap);
void sol_producer_begin(void* workspace, int batch, int seq_len, int num_heads, int n_tok,
                        hipStream_t stream);
void sol_producer_chunk(void* workspace, const void* qkv, const void* fab, const void* qw,
                        const void* kw, const void* kmean, const void* vscale, const void* blen,
                        float rope_eps, int rot_dim, int t0, int M, int batch, int seq_len,
                        int num_heads, int n_tok, hipStream_t stream);
void launch_sol_attn_core(void* workspace, void* out, const void* vscale, void* kmean_next,
                          void* vamax_out, const void* blen, int tail, int batch, int seq_len,
                          int num_heads, float tau, float scale, const void* ext_threshold,
                          int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                          int n_tok, hipStream_t stream);
void launch_sol_attn(const void* q, const void* k, const void* v, void* out, void* workspace,
                     int batch, int seq_len, int num_heads, int head_dim, int elem, float tau, float scale,
                     const void* key_bias, const void* ext_threshold, const void* blen, int tail,
                     int sink_start, int sink_end, int sink_q_start, int sink_q_end, int64_t qs_b,
                     int64_t qs_t, int64_t qs_h, int64_t ks_b, int64_t ks_t, int64_t ks_h,
                     int64_t vs_b, int64_t vs_t, int64_t vs_h, int n_tok, hipStream_t stream);

}  // extern "C"
