// SPDX-License-Identifier: Apache-2.0
//
// Kitchen CUDA AWQ W4A16: int4 weight, fp16/bf16 activation matmul.
//
// Three kernels live here behind one launch entry point
// (`launch_awq_w4a16_kernel`):
//
//   * naive — M ≤ kGemvMThreshold (8): each thread owns one output element,
//             dequant on the fly. Best when M is small enough that the
//             tile-launch overhead of the MMA path swamps the per-output
//             cost of redundant qweight reads.
//   * mma   — M >  kGemvMThreshold: cooperative dequant of the int4 weight
//             tile to bf16 in shmem, then mma.m16n8k16.bf16.bf16.f32 fused
//             over the K dim. No 113 MB temporary W workspace, no Python
//             dequant fallback. fp32 accumulator → downcast on store.
//
// All paths consume the kitchen-native AWQ layout (matches
// comfy_kitchen.backends.eager.awq):
//
//   x        (M, K)      bf16/fp16   row-major activation
//   qweight  (N, K/2)    int8        two uint4 per byte
//                                      bits 0..3  -> column 2k    (range [0, 15])
//                                      bits 4..7  -> column 2k+1
//   wscales  (K/G, N)    bf16/fp16   per-group, per-column scale
//   wzeros   (K/G, N)    bf16/fp16   per-group, per-column zero
//   out      (M, N)      bf16/fp16   = wscales.dtype
//
// Forward math (per-group asymmetric, matches eager / nunchaku-compat):
//   W[n, k] = (qweight[n, k] - 8) * wscales[k/G, n] + wzeros[k/G, n]
//   out     = x @ W.T
//
// bias and any LoRA-up are applied externally in the Python wrapper to keep
// kernel shape minimal (mirrors scaled_mm_svdquant_w4a4 epilogue contract).
//
// fp32 accumulator: per-group scale × dequant nibble products can reach the
// same order-of-magnitude as the W4A4 path; fp16 accumulators silently
// overflow on Qwen-Image-Edit modulation columns. Always accumulate fp32
// and downcast on store.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kGemvMThreshold   = 8;    // M ≤ 8 routes to gemv (naive) kernel

template<typename T>
__device__ __forceinline__ float to_fp32(T v);

template<>
__device__ __forceinline__ float to_fp32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<>
__device__ __forceinline__ float to_fp32<__half>(__half v) { return __half2float(v); }

template<typename T>
__device__ __forceinline__ T from_fp32(float v);

template<>
__device__ __forceinline__ __nv_bfloat16 from_fp32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<>
__device__ __forceinline__ __half from_fp32<__half>(float v) { return __float2half(v); }

// ---------------------------------------------------------------------------
// MMA helpers: ldmatrix + mma.m16n8k16.row.col.f32.bf16.bf16.f32 (or fp16).
// Pattern matches CUTLASS Hopper tutorials and nunchaku/src/kernels/awq.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvta_smem_u32(const void* ptr) {
    uint32_t s;
    asm("{ .reg .u64 ll; cvta.to.shared.u64 ll, %1; cvt.u32.u64 %0, ll; }"
        : "=r"(s) : "l"(ptr));
    return s;
}

// ldmatrix.x4: each lane gets 4 32-bit regs covering 4× (8x8 b16) sub-tiles.
// Used for A operand of mma.m16n8k16 (16M × 16K row-major in shmem).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "r"(addr));
}

template<typename T>
__device__ __forceinline__ void mma_m16n8k16_f32(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]);

template<>
__device__ __forceinline__ void mma_m16n8k16_f32<__nv_bfloat16>(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template<>
__device__ __forceinline__ void mma_m16n8k16_f32<__half>(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---------------------------------------------------------------------------
// Naive correctness kernel (Phase 1 baseline).
//   grid  = (ceil(N / kThreadsN), M)
//   block = kThreadsN threads (one per output column in the CTA's M row)
// Each thread iterates the full K range, dequantizing one nibble at a time.
// fp32 accumulator. No shmem reuse, no MMA — just gets the math right.
// ---------------------------------------------------------------------------
template<typename T>
__global__ void awq_w4a16_naive_kernel(
    const T* __restrict__ x,          // (M, K)
    const int8_t* __restrict__ qweight, // (N, K/2)
    const T* __restrict__ wscales,    // (K/G, N)
    const T* __restrict__ wzeros,     // (K/G, N)
    T* __restrict__ out,              // (M, N)
    int M, int N, int K, int G)
{
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int K_half = K / 2;
    const int n_groups = K / G;

    float acc = 0.0f;
    const T* x_row = x + m * K;
    const int8_t* qw_row = qweight + n * K_half;

    for (int g = 0; g < n_groups; ++g) {
        const float scale = to_fp32(wscales[g * N + n]);
        const float zero  = to_fp32(wzeros[g * N + n]);
        const int k_base  = g * G;
        const int kh_base = k_base / 2;

        // Two K positions per byte; G is even so this loops G/2 times cleanly.
        #pragma unroll 8
        for (int kk = 0; kk < G; kk += 2) {
            const uint8_t byte = static_cast<uint8_t>(qw_row[kh_base + kk / 2]);
            const int lo = byte & 0xF;
            const int hi = (byte >> 4) & 0xF;
            const float w_lo = (static_cast<float>(lo) - 8.0f) * scale + zero;
            const float w_hi = (static_cast<float>(hi) - 8.0f) * scale + zero;
            acc += to_fp32(x_row[k_base + kk])     * w_lo;
            acc += to_fp32(x_row[k_base + kk + 1]) * w_hi;
        }
    }
    out[m * N + n] = from_fp32<T>(acc);
}

template<typename T>
void launch_naive(
    const void* x, const void* qweight,
    const void* wscales, const void* wzeros,
    void* out,
    int M, int N, int K, int G,
    cudaStream_t stream)
{
    constexpr int kThreadsN = 128;
    dim3 block(kThreadsN);
    dim3 grid((N + kThreadsN - 1) / kThreadsN, M);
    awq_w4a16_naive_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(x),
        reinterpret_cast<const int8_t*>(qweight),
        reinterpret_cast<const T*>(wscales),
        reinterpret_cast<const T*>(wzeros),
        reinterpret_cast<T*>(out),
        M, N, K, G);
}

// ---------------------------------------------------------------------------
// Shmem-tiled GEMM kernel (M > 8 path).
//
// CTA covers (BLOCK_M × BLOCK_N) outputs. Each iteration of K consumes one
// quantization group (BLOCK_K = G = 64), reading:
//   * qweight tile (BLOCK_N × BLOCK_K/2) int8 — shared by all M rows in CTA
//   * x       tile (BLOCK_M × BLOCK_K) bf16/fp16 — shared by all N cols
//   * one (scale, zero) pair per N — broadcast across the K window
//
// Threads = BLOCK_N (one per output column); each thread accumulates BLOCK_M
// fp32 partial sums in registers. Group-shared scale/zero stay in scalar
// registers, qweight bytes come from shmem cache, x rows from shmem.
//
// Why this is faster than the naive kernel at large M:
//   - qweight is read once per CTA per K group, not once per (M, N) cell
//   - BLOCK_N=128 threads cooperate to load tiles in coalesced bursts
//   - x row reuse: BLOCK_M rows share the same dequantized weight bytes
// ---------------------------------------------------------------------------
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void awq_w4a16_tiled_kernel(
    const T* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const T* __restrict__ wscales,
    const T* __restrict__ wzeros,
    T* __restrict__ out,
    int M, int N, int K, int G)
{
    static_assert(BLOCK_K % 2 == 0, "BLOCK_K must be even (uint4 pair packing)");
    static_assert(BLOCK_N <= 1024, "BLOCK_N must fit in one CTA");

    constexpr int BLOCK_KH = BLOCK_K / 2;

    __shared__ int8_t qw_sh[BLOCK_N * BLOCK_KH];
    __shared__ T      x_sh [BLOCK_M * BLOCK_K];

    const int cta_n = blockIdx.x * BLOCK_N;
    const int cta_m = blockIdx.y * BLOCK_M;
    const int tid   = threadIdx.x;
    const int n_global = cta_n + tid;            // tid is one-per-N-col within CTA

    const int K_half      = K / 2;
    const int n_groups    = K / G;
    const int groups_per_block = BLOCK_K / G;    // = 1 when BLOCK_K == G
    static_assert(BLOCK_K == 64, "current code assumes one quantization group per K-tile");

    float acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; ++i) acc[i] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        const int k_base  = g * G;
        const int kh_base = k_base / 2;

        // ---- Cooperative loads for this K group ----
        // qweight: (BLOCK_N, BLOCK_KH) — each thread loads one full row segment.
        if (n_global < N) {
            const int8_t* qw_row = qweight + n_global * K_half + kh_base;
            #pragma unroll
            for (int kk = 0; kk < BLOCK_KH; ++kk) {
                qw_sh[tid * BLOCK_KH + kk] = qw_row[kk];
            }
        } else {
            #pragma unroll
            for (int kk = 0; kk < BLOCK_KH; ++kk) {
                qw_sh[tid * BLOCK_KH + kk] = 0;
            }
        }
        // x: (BLOCK_M, BLOCK_K) — flatten and stripe.
        const int x_total = BLOCK_M * BLOCK_K;
        #pragma unroll
        for (int idx = tid; idx < x_total; idx += BLOCK_N) {
            const int mm = idx / BLOCK_K;
            const int kk = idx % BLOCK_K;
            const int m_global = cta_m + mm;
            const int k_global = k_base + kk;
            x_sh[idx] = (m_global < M) ? x[m_global * K + k_global] : T(0.0f);
        }

        __syncthreads();

        // ---- Per-thread compute over this K window ----
        // Each thread owns its column n_global; load (scale, zero) once.
        if (n_global < N) {
            const float scale = to_fp32(wscales[g * N + n_global]);
            const float zero  = to_fp32(wzeros [g * N + n_global]);
            const int8_t* qw_my = qw_sh + tid * BLOCK_KH;

            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; kk += 2) {
                const uint8_t byte = static_cast<uint8_t>(qw_my[kk / 2]);
                const int lo = byte & 0xF;
                const int hi = (byte >> 4) & 0xF;
                const float w_lo = (static_cast<float>(lo) - 8.0f) * scale + zero;
                const float w_hi = (static_cast<float>(hi) - 8.0f) * scale + zero;
                #pragma unroll
                for (int mm = 0; mm < BLOCK_M; ++mm) {
                    const float xv0 = to_fp32(x_sh[mm * BLOCK_K + kk]);
                    const float xv1 = to_fp32(x_sh[mm * BLOCK_K + kk + 1]);
                    acc[mm] += xv0 * w_lo;
                    acc[mm] += xv1 * w_hi;
                }
            }
        }

        __syncthreads();
    }

    // ---- Write outputs ----
    if (n_global < N) {
        #pragma unroll
        for (int mm = 0; mm < BLOCK_M; ++mm) {
            const int m_global = cta_m + mm;
            if (m_global < M) {
                out[m_global * N + n_global] = from_fp32<T>(acc[mm]);
            }
        }
    }
}

template<typename T, int BLOCK_M_, int BLOCK_N_>
void launch_tiled_specific(
    const void* x, const void* qweight,
    const void* wscales, const void* wzeros,
    void* out,
    int M, int N, int K, int G,
    cudaStream_t stream)
{
    constexpr int BLOCK_M = BLOCK_M_;
    constexpr int BLOCK_N = BLOCK_N_;
    constexpr int BLOCK_K = 64;

    dim3 block(BLOCK_N);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    awq_w4a16_tiled_kernel<T, BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(x),
        reinterpret_cast<const int8_t*>(qweight),
        reinterpret_cast<const T*>(wscales),
        reinterpret_cast<const T*>(wzeros),
        reinterpret_cast<T*>(out),
        M, N, K, G);
}

template<typename T>
void launch_tiled(
    const void* x, const void* qweight,
    const void* wscales, const void* wzeros,
    void* out,
    int M, int N, int K, int G,
    cudaStream_t stream)
{
    // Pick BLOCK_M by M: small M → narrow tile (fewer wasted M-rows in
    // the tail CTA), large M → wide tile (better qweight reuse).
    if (M <= 32) {
        launch_tiled_specific<T, 16, 128>(x, qweight, wscales, wzeros, out, M, N, K, G, stream);
    } else {
        launch_tiled_specific<T, 32, 128>(x, qweight, wscales, wzeros, out, M, N, K, G, stream);
    }
}

// ---------------------------------------------------------------------------
// Fused int4 × bf16/fp16 MMA GEMM kernel.
//
// Tile layout:
//   BLOCK_M = 16   (= mma m16, single warp row)
//   BLOCK_N = 128  (4 warps × 32 cols, each warp owns 4 N-MMA tiles of 8 cols)
//   BLOCK_K = 64   (= one quantization group; 4 K-MMA tiles of 16)
//   NUM_WARPS = 4
//
// Per CTA per K-tile (= one quant group):
//   1. Cooperative load qweight tile (BLOCK_N, BLOCK_K/2) int8 + scale/zero
//      (1 entry each per N) → dequantize to bf16 W tile (BLOCK_N, BLOCK_K) in
//      shmem. Each thread handles one N row of dequant work.
//   2. Cooperative load x tile (BLOCK_M, BLOCK_K) bf16 → shmem (vec4 loads).
//   3. __syncthreads.
//   4. Per warp: K-MMA × N-MMA loop using ldmatrix + mma.m16n8k16.f32.
//      Accumulate into per-lane fp32 fragment registers (4 fp32 per N-tile).
//
// Output staging: fragments → shmem (so each thread owns a contiguous output
// row) → coalesced bf16 stores. fp32 accum on chip; downcast on store.
//
// Why this beats dequant + cuBLAS:
//   * No 113 MB intermediate W workspace — qweight stays int4 in HBM.
//   * Dequant happens only inside shmem (small live working set).
//   * MMA tensor cores match cuBLAS bf16 throughput at the compute step.
// ---------------------------------------------------------------------------
template<typename T>
__global__ void awq_w4a16_mma_kernel(
    const T* __restrict__ x,            // (M, K)
    const int8_t* __restrict__ qweight, // (N, K/2)
    const T* __restrict__ wscales,      // (K/G, N)
    const T* __restrict__ wzeros,       // (K/G, N)
    T* __restrict__ out,                // (M, N)
    int M, int N, int K, int G)
{
    constexpr int BLOCK_M  = 16;
    constexpr int BLOCK_N  = 128;
    constexpr int BLOCK_K  = 64;
    constexpr int BLOCK_KH = BLOCK_K / 2;
    constexpr int NUM_WARPS = 4;
    constexpr int CTA_THREADS = NUM_WARPS * 32;
    constexpr int WARP_N = BLOCK_N / NUM_WARPS;   // 32
    constexpr int N_MMA  = WARP_N / 8;            // 4 N-MMA tiles per warp
    constexpr int K_MMA  = BLOCK_K / 16;          // 4 K-MMA tiles per K-tile

    // Shmem row stride padding: BLOCK_K = 64 b16 = 128 bytes is a multiple of
    // the 128-byte shmem bank cycle (32 banks × 4 bytes). With unpadded
    // stride every ldmatrix lane within a sub-tile lands on the same bank,
    // costing a 16-way conflict per call. Bumping stride to 72 b16 (= 144
    // bytes ⇒ 8 b16 / 16-byte step per row) breaks the alignment so the 32
    // ldmatrix lanes spread across all banks.
    constexpr int SMEM_STRIDE_K = BLOCK_K + 8;

    static_assert(BLOCK_K == 64, "current code assumes one quant group per K-tile");
    static_assert(BLOCK_N == NUM_WARPS * WARP_N, "BLOCK_N must split evenly across warps");
    static_assert(BLOCK_N == CTA_THREADS, "this kernel uses 1 thread per N row for the dequant pass");

    __shared__ alignas(16) T  x_sh[BLOCK_M  * SMEM_STRIDE_K];
    __shared__ alignas(16) T  w_sh[BLOCK_N  * SMEM_STRIDE_K];

    const int cta_n = blockIdx.x * BLOCK_N;
    const int cta_m = blockIdx.y * BLOCK_M;
    const int tid   = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_n_base = warp_id * WARP_N;

    const int K_half   = K / 2;
    const int n_groups = K / G;

    // Per-warp accumulators: N_MMA × 4 fp32 (mma m16n8 produces 4 fp32 per lane)
    float acc[N_MMA][4];
    #pragma unroll
    for (int i = 0; i < N_MMA; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        const int K_base  = g * G;
        const int Kh_base = K_base / 2;

        // ---- Load + dequant qweight tile to bf16 W in shmem ------------------
        // 128 threads, BLOCK_N=128 N rows: 1 thread per N row.
        const int n_in_cta  = tid;
        const int n_global  = cta_n + n_in_cta;
        const float scale = (n_global < N) ? to_fp32(wscales[g * N + n_global]) : 0.0f;
        const float zero  = (n_global < N) ? to_fp32(wzeros [g * N + n_global]) : 0.0f;

        if (n_global < N) {
            const int8_t* qw_row = qweight + n_global * K_half + Kh_base;
            #pragma unroll
            for (int kh = 0; kh < BLOCK_KH; ++kh) {
                const uint8_t byte = static_cast<uint8_t>(qw_row[kh]);
                const int lo = byte & 0xF;
                const int hi = (byte >> 4) & 0xF;
                w_sh[n_in_cta * SMEM_STRIDE_K + 2 * kh    ] = from_fp32<T>((float(lo) - 8.0f) * scale + zero);
                w_sh[n_in_cta * SMEM_STRIDE_K + 2 * kh + 1] = from_fp32<T>((float(hi) - 8.0f) * scale + zero);
            }
        } else {
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                w_sh[n_in_cta * SMEM_STRIDE_K + kk] = T(0.0f);
            }
        }

        // ---- Load x tile into shmem (vec4 = 8 bf16 per load) -----------------
        // Total elements = BLOCK_M * BLOCK_K = 1024; threads = 128 → 8 elems/thread.
        // 1 vec4 (uint4 = 8 b16) per thread covers it.
        {
            const int total_vec = (BLOCK_M * BLOCK_K) / 8;   // 128
            const int v = tid;                               // one vec per thread
            if (v < total_vec) {
                const int mm = (v * 8) / BLOCK_K;
                const int kk = (v * 8) % BLOCK_K;
                const int m_global = cta_m + mm;
                const int k_global = K_base + kk;
                uint4* dst = reinterpret_cast<uint4*>(&x_sh[mm * SMEM_STRIDE_K + kk]);
                if (m_global < M) {
                    *dst = *reinterpret_cast<const uint4*>(&x[m_global * K + k_global]);
                } else {
                    *dst = make_uint4(0, 0, 0, 0);
                }
            }
        }

        __syncthreads();

        // ---- MMA inner loop --------------------------------------------------
        // For each k_mma in [0, K_MMA): load A frag (16M × 16K) once, then for
        // each N-MMA load a (8N × 16K) B frag and issue m16n8k16.
        //
        // ldmatrix lane mapping for the A operand (16M × 16K row-major in
        // shmem, stride = BLOCK_K):
        //   lane address = x_sh + (lane%16) * BLOCK_K + (lane/16) * 8 + k_off
        //
        // For the B operand (8N × 16K, our W is N-major in shmem) we use
        // ldmatrix.x4.trans with a (16N × 16K) layout to get TWO B operands
        // packed into the same 4 regs (4 sub-tiles → 2 stacked 8N × 16K).
        // So one ldmatrix.x4.trans serves 2 N-MMAs → call it twice for 4.
        #pragma unroll
        for (int k_mma = 0; k_mma < K_MMA; ++k_mma) {
            const int k_off = k_mma * 16;

            uint32_t a_frag[4];
            {
                const T* a_addr = &x_sh[(lane % 16) * SMEM_STRIDE_K + (lane / 16) * 8 + k_off];
                ldmatrix_x4(a_frag, cvta_smem_u32(a_addr));
            }

            // 4 N-MMAs per warp: 2 ldmatrix.x4 (no trans) calls. Our W tile
            // is (BLOCK_N, BLOCK_K) row-major in shmem; treating it as the
            // mma B operand in (16K × 8N) col-major form is identical to a
            // row-major (8N × 16K) view, which is what ldmatrix.x4 (no trans)
            // produces directly — lane k reg 0 ↔ src[k/4, (k%4)*2..+1] is
            // already the layout mma B expects (reg 0 = B[K=(k%4)*2..+1, N=k/4]).
            //
            // The 4 sub-tiles returned by one ldmatrix.x4 cover a (16N × 16K)
            // block arranged as:
            //   reg 0: W[n_off + 0..7,   k_off + 0..7]    (ST0)
            //   reg 1: W[n_off + 8..15,  k_off + 0..7]    (ST1)
            //   reg 2: W[n_off + 0..7,   k_off + 8..15]   (ST2)
            //   reg 3: W[n_off + 8..15,  k_off + 8..15]   (ST3)
            // → mma B at (n_off,   k_off..+15) = (reg 0, reg 2)
            // → mma B at (n_off+8, k_off..+15) = (reg 1, reg 3)
            #pragma unroll
            for (int b_pair = 0; b_pair < N_MMA / 2; ++b_pair) {
                const int n_off = warp_n_base + b_pair * 16;
                uint32_t b_frag4[4];
                const T* b_addr = &w_sh[(n_off + (lane % 16)) * SMEM_STRIDE_K + (lane / 16) * 8 + k_off];
                ldmatrix_x4(b_frag4, cvta_smem_u32(b_addr));

                {
                    const uint32_t b0[2] = {b_frag4[0], b_frag4[2]};
                    mma_m16n8k16_f32<T>(acc[b_pair * 2 + 0], a_frag, b0);
                }
                {
                    const uint32_t b1[2] = {b_frag4[1], b_frag4[3]};
                    mma_m16n8k16_f32<T>(acc[b_pair * 2 + 1], a_frag, b1);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store accumulators ---------------------------------------------------
    // mma m16n8k16 output fragment: lane k holds 4 fp32 mapped as
    //   c0 → row (k/4),     col (k%4)*2
    //   c1 → row (k/4),     col (k%4)*2 + 1
    //   c2 → row (k/4) + 8, col (k%4)*2
    //   c3 → row (k/4) + 8, col (k%4)*2 + 1
    // We write each fragment (16M × 8N) directly to global `out`.
    const int row_lo = lane / 4;          // 0..7
    const int col_lo = (lane % 4) * 2;    // 0,2,4,6
    #pragma unroll
    for (int n_mma = 0; n_mma < N_MMA; ++n_mma) {
        const int n_global_base = cta_n + warp_n_base + n_mma * 8;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int m_global = cta_m + row_lo + half * 8;
            if (m_global < M) {
                const int n0 = n_global_base + col_lo;
                const int n1 = n0 + 1;
                if (n0 < N) out[m_global * N + n0] = from_fp32<T>(acc[n_mma][half * 2 + 0]);
                if (n1 < N) out[m_global * N + n1] = from_fp32<T>(acc[n_mma][half * 2 + 1]);
            }
        }
    }
}

template<typename T>
void launch_mma(
    const void* x, const void* qweight,
    const void* wscales, const void* wzeros,
    void* out,
    int M, int N, int K, int G,
    cudaStream_t stream)
{
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 128;
    constexpr int CTA_THREADS = 128;
    dim3 block(CTA_THREADS);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    awq_w4a16_mma_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(x),
        reinterpret_cast<const int8_t*>(qweight),
        reinterpret_cast<const T*>(wscales),
        reinterpret_cast<const T*>(wzeros),
        reinterpret_cast<T*>(out),
        M, N, K, G);
}

} // namespace

// ---------------------------------------------------------------------------
// extern "C" launch entry point — called from dlpack_bindings.cpp.
//   dtype_code: 1 = fp16, 2 = bf16 (matches map_dtype_to_code)
// ---------------------------------------------------------------------------
extern "C" void launch_awq_w4a16_kernel(
    const void* x,
    const void* qweight,
    const void* wscales,
    const void* wzeros,
    void* out,
    int M, int N, int K, int G,
    int dtype_code,
    cudaStream_t stream)
{
    // M ≤ kGemvMThreshold: the naive 1-thread-per-output kernel keeps weight
    // re-reads minimal (M is small, so global qweight reads happen ~M times
    // per (N, K) cell, which is fine for tiny M and avoids tile-launch
    // overhead).
    // M >  kGemvMThreshold: the fused MMA kernel — cooperative dequant of
    // the int4 weight tile to bf16 in shmem, then mma.m16n8k16.f32 along K.
    // This eliminates the 113 MB intermediate W workspace that the naive
    // tiled or Python-side fallback paths require.
    const bool use_mma = (M > kGemvMThreshold);

    if (dtype_code == 2) {        // bfloat16
        if (use_mma) launch_mma<__nv_bfloat16>(x, qweight, wscales, wzeros, out, M, N, K, G, stream);
        else         launch_naive<__nv_bfloat16>(x, qweight, wscales, wzeros, out, M, N, K, G, stream);
    } else if (dtype_code == 1) { // float16
        if (use_mma) launch_mma<__half>(x, qweight, wscales, wzeros, out, M, N, K, G, stream);
        else         launch_naive<__half>(x, qweight, wscales, wzeros, out, M, N, K, G, stream);
    } else {
        // Caller validates dtype before reaching here.
    }
}
