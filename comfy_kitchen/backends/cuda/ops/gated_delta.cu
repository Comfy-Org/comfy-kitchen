// Fused GatedDeltaNet decode for short sequences (plain decode and the MTP
// speculative verify, S <= 8). The full [DK, DV] fp32 recurrent state lives in
// dynamic shared memory across all S steps — one block per (batch, head), one
// thread per value column, two state passes per step — replacing the ~10 small
// elementwise/matmul launches per step per layer of the eager chain.
// Per-step math (matches the eager stepwise loop):
//   state *= g[s]
//   kv[t]   = sum_k k[s][kk] * state[kk][t]
//   delta[t] = (v[s][t] - kv[t]) * beta[s]
//   state[kk][t] += k[s][kk] * delta[t]
//   out[s][t] = sum_k q[s][kk] * state[kk][t]
// Snapshots of the state after steps 0..S-2 are written for verify rollback.

#include <cuda_runtime.h>
#include <cstdint>

namespace {

template <int DK>
__global__ void gated_delta_decode_kernel(
    const float* __restrict__ q,     // [B, S, H, DK] normalized, pre-scaled
    const float* __restrict__ k,     // [B, S, H, DK] normalized
    const float* __restrict__ v,     // [B, S, H, DV]
    const float* __restrict__ beta,  // [B, S, H]
    const float* __restrict__ g,     // [B, S, H] decay multiplier (already exp)
    float* __restrict__ state,       // [B, H, DK, DV] updated in place
    float* __restrict__ out,         // [B, S, H, DV]
    float* __restrict__ snapshots,   // [S-1, B, H, DK, DV] or nullptr
    int B, int H, int S, int DV)
{
    extern __shared__ float sm[];    // DK * DV state, then DK k-row, DK q-row
    float* sk = sm + DK * DV;
    float* sq = sk + DK;
    const int b = static_cast<int>(blockIdx.x) / H;
    const int h = static_cast<int>(blockIdx.x) % H;
    const int t = threadIdx.x;
    const int64_t state_off = (static_cast<int64_t>(b) * H + h) * DK * DV;

    for (int i = t; i < DK * DV; i += blockDim.x)
        sm[i] = state[state_off + i];
    __syncthreads();

    for (int s = 0; s < S; ++s) {
        const int64_t row = (static_cast<int64_t>(b) * S + s) * H + h;
        const float gs = g[row];
        const float bs = beta[row];
        for (int i = t; i < DK; i += blockDim.x) {
            sk[i] = k[row * DK + i];
            sq[i] = q[row * DK + i];
        }
        __syncthreads();
        if (t < DV) {
            float kv = 0.0f;
            #pragma unroll 4
            for (int kk = 0; kk < DK; ++kk) {
                const float sv = sm[kk * DV + t] * gs;
                sm[kk * DV + t] = sv;
                kv = fmaf(sk[kk], sv, kv);
            }
            const float delta = (v[row * DV + t] - kv) * bs;
            float o = 0.0f;
            #pragma unroll 4
            for (int kk = 0; kk < DK; ++kk) {
                const float sv = fmaf(sk[kk], delta, sm[kk * DV + t]);
                sm[kk * DV + t] = sv;
                o = fmaf(sq[kk], sv, o);
            }
            out[row * DV + t] = o;
        }
        __syncthreads();
        if (snapshots != nullptr && s < S - 1) {
            float* snap = snapshots + ((static_cast<int64_t>(s) * B + b) * H + h) * DK * DV;
            for (int i = t; i < DK * DV; i += blockDim.x)
                snap[i] = sm[i];
        }
        __syncthreads();
    }

    for (int i = t; i < DK * DV; i += blockDim.x)
        state[state_off + i] = sm[i];
}

}  // namespace

extern "C" bool launch_gated_delta_decode(
    const void* q, const void* k, const void* v, const void* beta, const void* g,
    void* state, void* out, void* snapshots,
    int64_t B, int64_t H, int64_t S, int64_t DK, int64_t DV, cudaStream_t stream)
{
    if (DK != 128 || DV <= 0 || DV > 512 || S < 1 || S > 8)
        return false;
    const size_t shmem = (static_cast<size_t>(DK) * DV + 2 * DK) * sizeof(float);
    int dev = 0, max_shmem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    if (cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        return false;
    if (shmem > static_cast<size_t>(max_shmem))
        return false;
    if (cudaFuncSetAttribute(gated_delta_decode_kernel<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem)) != cudaSuccess)
        return false;
    const int threads = static_cast<int>(DV) >= 128 ? static_cast<int>(DV) : 128;
    gated_delta_decode_kernel<128><<<static_cast<unsigned>(B * H), threads, shmem, stream>>>(
        static_cast<const float*>(q), static_cast<const float*>(k), static_cast<const float*>(v),
        static_cast<const float*>(beta), static_cast<const float*>(g),
        static_cast<float*>(state), static_cast<float*>(out), static_cast<float*>(snapshots),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(DV));
    return cudaGetLastError() == cudaSuccess;
}
