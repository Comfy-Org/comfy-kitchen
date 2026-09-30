#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#else
#error "mma_gfx10.h is only intended for ROCm/HIP."
#endif

#include <cstdint>
#include <type_traits>

namespace sageattn_gfx10 {

typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int sdot4_i32_i8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, true);
}

__device__ __forceinline__ float fdot2_f32_f16(unsigned a, unsigned b, float c) {
    return __builtin_amdgcn_fdot2(
        *reinterpret_cast<const v2h*>(&a), *reinterpret_cast<const v2h*>(&b), c, true);
}

__device__ __forceinline__ int load_i8_quad(const int8_t* p) {
    return *reinterpret_cast<const int*>(p);
}

__device__ __forceinline__ int8_t gfx10_q8_round(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int i = static_cast<int>(x);
    if (i > 127) i = 127;
    if (i < -128) i = -128;
    return static_cast<int8_t>(i);
}

__device__ __forceinline__ int gfx10_q8_pack(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (static_cast<int>(a) & 0xff) | ((static_cast<int>(b) & 0xff) << 8) |
           ((static_cast<int>(c) & 0xff) << 16) | ((static_cast<int>(d) & 0xff) << 24);
}

__device__ __forceinline__ float gfx10_q8_2f(unsigned raw_half, bool bf16) {
    if (bf16) {
        __hip_bfloat16 b = *reinterpret_cast<const __hip_bfloat16*>(&raw_half);
        return __bfloat162float(b);
    }
    __half h = *reinterpret_cast<const __half*>(&raw_half);
    return __half2float(h);
}

template <typename ODT> __device__ __forceinline__ ODT gfx10_out_convert(float v);
template <> __device__ __forceinline__ __half gfx10_out_convert<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __hip_bfloat16 gfx10_out_convert<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

// ---------------------------------------------------------------------------
// Attention masks
//
// RDNA2 has no matrix cores, so the mask support the WMMA port gets from the MMA
// fragment layout has to be expressed differently here. The *representations*
// are shared with the WMMA backend on purpose: the same sage_prepare_key_mask /
// sage_prepare_dense_mask producers fill them, and comfy_kitchen/sage_attention.py
// picks which one to build. That is what lets a prepared mask be a device-side
// snapshot which survives the float inputs going away, and it keeps the two
// backends reading one layout instead of two.
//
// Every representation answers the same two questions for one (query, key) pair:
// is this key kept, and what bias (in the base-two softmax domain) does it add?
// A dropped key contributes nothing at all -- the caller never evaluates exp2 on
// its score -- so the sentinel the producer stored for it is never used as a
// number. That removes the -3e38/NaN bookkeeping the WMMA kernels need and makes
// a fully masked row fall out as an exact zero, with no separate validity flag.
//
// The dense layouts pack a 16-query by 64-key tile into 1024 slots, or into 32
// words of 32 keep bits for the Boolean form. Both use the WMMA accumulator's key
// order, which the RDNA2 kernel need not reproduce but can read, because the
// permutation is a pure function of the two tile-local coordinates:
//
//   dense value slot = q_local * 8 + (k % 8) + ((k / 8 % 2) << 7) + (k / 16 << 8)
//   dense keep word  = q_local + 16 * (k / 8 % 2), bit (k / 16) * 8 + (k % 8)
//
// tests/test_int8_attention.py decodes the 16-bit buffer element by element
// against the shared producer, so these two formulas are pinned from Python.
// ---------------------------------------------------------------------------
// The numeric values are the ones sage_attention.py and the WMMA backend's
// MaskMode already use, so one host-side "mode" selects the same representation
// on both backends. Plain ints, not an enum class: the mode is a template
// argument, and `if constexpr` on an enum-typed parameter from another
// translation unit's point of view is needlessly awkward for no benefit.
namespace MaskMode {
constexpr int kNone = 0;
constexpr int kRaw = 2;
constexpr int kPreparedKey = 4;
constexpr int kPreparedDense = 5;
constexpr int kPreparedDenseBool = 7;
constexpr int kPreparedDenseBF16 = 8;
constexpr int kPreparedDenseF16 = 9;
}  // namespace MaskMode

// The producer's tiles, not this kernel's BN: 64 keys everywhere and 16 queries
// for the dense layouts, so one mask is readable by both backends.
constexpr int kGfx10TileK = 64;
constexpr int kGfx10TileQ = 16;
constexpr int kGfx10DenseSlot = 1024;
constexpr int kGfx10BoolWord = 32;

// Stand-in for a dropped key's score. It only ever reaches fmaxf, so the exact
// value is irrelevant as long as it is finite and below every real score;
// -3.0e38f is what the unmasked path already used for out-of-range keys.
constexpr float kGfx10MaskedScore = -3.0e38f;

// -ffast-math implies -ffinite-math-only, which lets the optimizer conclude that
// a float-typed value is finite and delete the branch that tests it. That is
// exactly the test that decides "this bias drops the key", so it always runs on
// the raw bits and never on a float. Same rule as mask_keep() in
// sage_attention/int8_attn.hip.
__device__ __forceinline__ bool gfx10_bits_dropped(uint32_t bits, uint32_t exponent) {
    return (bits & exponent) == exponent;
}

// Reads one raw (unprepared) mask element. False means dropped; otherwise the
// bias comes back in *natural* units and the caller scales it, which is what
// makes the raw and prepared-16 reads agree bit for bit.
__device__ __forceinline__ bool gfx10_raw_mask_bias(
    const void* mask, int64_t offset, int dtype_code, float& bias) {
    if (dtype_code == 3) {
        bias = 0.0f;
        return static_cast<const uint8_t*>(mask)[offset] != 0;
    }
    if (dtype_code == 0) {
        const uint32_t bits = static_cast<const uint32_t*>(mask)[offset];
        if (gfx10_bits_dropped(bits, 0x7F800000u)) return false;
        bias = __uint_as_float(bits);
        return true;
    }
    const uint16_t bits = static_cast<const uint16_t*>(mask)[offset];
    if (dtype_code == 1) {
        if (gfx10_bits_dropped(bits, 0x7C00u)) return false;
        bias = __half2float(*reinterpret_cast<const __half*>(&bits));
        return true;
    }
    if (gfx10_bits_dropped(bits, 0x7F80u)) return false;
    bias = static_cast<float>(*reinterpret_cast<const __bf16*>(&bits));
    return true;
}

// 16-bit dense biases stay in natural units, exactly as the raw 16-bit read
// returns them, so both scale once and land on the same float.
template <bool Bf16>
__device__ __forceinline__ bool gfx10_dense16_mask_bias(uint16_t bits, float& bias) {
    const uint16_t exponent = Bf16 ? 0x7F80u : 0x7C00u;
    if (gfx10_bits_dropped(bits, exponent)) return false;
    bias = Bf16 ? static_cast<float>(__uint_as_float(static_cast<uint32_t>(bits) << 16))
                : __half2float(*reinterpret_cast<const __half*>(&bits));
    return true;
}

__device__ __forceinline__ bool gfx10_dense32_mask_bias(uint32_t bits, float& bias) {
    if (gfx10_bits_dropped(bits, 0x7F800000u)) return false;
    bias = __uint_as_float(bits);
    return true;
}

// Index of a (batch, head, query, key-tile) inside a prepared mask. Every
// prepared buffer is allocated contiguous by its producer, so the row pitch
// follows from the tile counts rather than from strides.
__device__ __forceinline__ int64_t gfx10_mask_row(
    int64_t batch, int64_t head, int64_t mask_batch, int64_t mask_heads,
    int64_t q_tiles, int64_t k_tiles, int64_t q_index, int64_t k_index) {
    const int64_t b = (mask_batch == 1) ? 0 : batch;
    const int64_t h = (mask_heads == 1) ? 0 : head;
    return ((b * mask_heads + h) * q_tiles + (q_index / kGfx10TileQ)) * k_tiles +
           (k_index / kGfx10TileK);
}

// One (query, key) lookup against any of the shared representations. `bias` is
// written only when the key is kept, and is always in the base-two log domain
// the softmax in the kernel below runs in.
__device__ __forceinline__ bool gfx10_mask_lookup(
    int mode, const void* mask, int dtype_code,
    int64_t stride_b, int64_t stride_h, int64_t stride_q, int64_t stride_k,
    int64_t batch, int64_t head, int64_t q_index, int64_t k_index,
    int64_t mask_batch, int64_t mask_heads, int64_t q_tiles, int64_t k_tiles,
    int64_t k_row_pitch, float log2e, float& bias) {
    switch (mode) {
        case MaskMode::kRaw: {
            // A raw mask arrives broadcast to the attention's shape, so a
            // stride-0 batch or head axis means every batch (or head) reads the
            // same row. Indexing with the raw batch/head would run off the end
            // of the smaller buffer, so each axis is clamped to what the mask
            // actually spans. The strides themselves already encode the
            // broadcast, so they are used as given.
            const int64_t mb = (mask_batch == 1) ? 0 : batch;
            const int64_t mh = (mask_heads == 1) ? 0 : head;
            const bool keep = gfx10_raw_mask_bias(
                mask,
                mb * stride_b + mh * stride_h + q_index * stride_q + k_index * stride_k,
                dtype_code, bias);
            if (keep) bias *= log2e;
            return keep;
        }
        case MaskMode::kPreparedKey: {
            // One row per (batch, head): k_tiles * 64 key biases followed by one
            // descriptor per tile, which is -inf when the tile kept nothing at
            // all so a whole tile is one test to skip.
            //
            // The mask has its own batch and head extents, which are usually
            // smaller than the attention's: a [1,1,1,K] mask broadcast over 4
            // query heads and 2 batches packs one row, not eight. Clamping each
            // index to the mask's own extent is what keeps such a mask from
            // reading past its single row -- the out-of-bounds read is what made
            // masked results differ between back-to-back calls, since the memory
            // past the buffer is whatever the allocator handed out next.
            const int64_t mb = (mask_batch == 1) ? 0 : batch;
            const int64_t mh = (mask_heads == 1) ? 0 : head;
            const float* row = static_cast<const float*>(mask) +
                               (mb * mask_heads + mh) * k_row_pitch;
            const int64_t tile = k_index / kGfx10TileK;
            if (gfx10_bits_dropped(__float_as_uint(row[k_tiles * kGfx10TileK + tile]),
                                   0x7F800000u)) {
                return false;
            }
            return gfx10_dense32_mask_bias(
                __float_as_uint(row[tile * kGfx10TileK + (k_index % kGfx10TileK)]), bias);
        }
        case MaskMode::kPreparedDenseBool: {
            // 32 words per 16-query by 64-key tile; within a tile the word is
            // q_local + 16 * (k/8 % 2) and the bit is (k/16) * 8 + k%8.
            const int64_t k_local = k_index % kGfx10TileK;
            const int64_t word =
                gfx10_mask_row(batch, head, mask_batch, mask_heads, q_tiles, k_tiles,
                               q_index, k_index) * kGfx10BoolWord +
                (q_index % kGfx10TileQ) + 16 * ((k_local / 8) % 2);
            const int bit = static_cast<int>((k_local / 16) * 8 + (k_local % 8));
            if ((static_cast<const uint32_t*>(mask)[word] >> bit) & 1u) {
                bias = 0.0f;
                return true;
            }
            return false;
        }
        default: {
            // 1024 slots per tile; within a tile the slot is
            // q_local * 8 + (k%8) + ((k/8%2) << 7) + (k/16 << 8).
            const int64_t k_local = k_index % kGfx10TileK;
            const int64_t slot =
                gfx10_mask_row(batch, head, mask_batch, mask_heads, q_tiles, k_tiles,
                               q_index, k_index) * kGfx10DenseSlot +
                (q_index % kGfx10TileQ) * 8 + (k_local % 8) + ((k_local / 8 % 2) << 7) +
                ((k_local / 16) << 8);
            if (mode == MaskMode::kPreparedDenseBF16) {
                const bool keep = gfx10_dense16_mask_bias<true>(
                    static_cast<const uint16_t*>(mask)[slot], bias);
                if (keep) bias *= log2e;
                return keep;
            }
            if (mode == MaskMode::kPreparedDenseF16) {
                const bool keep = gfx10_dense16_mask_bias<false>(
                    static_cast<const uint16_t*>(mask)[slot], bias);
                if (keep) bias *= log2e;
                return keep;
            }
            // The fp32 producer already scaled by log2(e) when it packed.
            return gfx10_dense32_mask_bias(static_cast<const uint32_t*>(mask)[slot], bias);
        }
    }
}

// MaskMode selects a Gfx10Mask and, for kRaw, the dtype of the operand. It is a
// template parameter rather than a runtime argument because every lookup is
// behind it: a runtime mode would cost a branch and a register per key in the
// inner loop, where the register budget is already the binding constraint.
template <int HD, bool ISC, int BN, int BM, typename ODT, int TM = 2, bool INQ = false,
          int MASK = MaskMode::kNone, int MASK_DTYPE = 0>
__global__ __attribute__((amdgpu_waves_per_eu(2))) __launch_bounds__(BM, 1) void attn_kernel_i8q_f16pv_tiled_pv(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h,
    int64_t ks_stride_b, int64_t ks_stride_h,
    int diag,
    const void* __restrict__ q_fp, int q_src_bf16, float sm_scale_log2e,
    const void* __restrict__ mask = nullptr,
    int64_t mask_stride_b = 0, int64_t mask_stride_h = 0,
    int64_t mask_stride_q = 0, int64_t mask_stride_k = 0,
    int64_t mask_batch = 1, int64_t mask_heads = 1, int64_t mask_q_tiles = 1,
    int64_t mask_k_tiles = 1, int64_t mask_k_row_pitch = 0) {
#if defined(__GFX10__)
    const int diag_qk = diag & 1;
    const int diag_sm = (diag >> 1) & 1;
    const int diag_pv = (diag >> 2) & 1;
    const int diag_st = (diag >> 3) & 1;
    const int diag_wb = (diag >> 4) & 1;
    constexpr bool kMasked = MASK != MaskMode::kNone;
    constexpr int QUADS = HD / 4;
    constexpr int NTHREAD = BM;
    constexpr int K_STRIDE = HD;
    constexpr int V_STRIDE = BN;
    constexpr int KS_TOTAL = BN * K_STRIDE;
    constexpr int VS_TOTAL = HD * V_STRIDE;
    constexpr int VDSW = 8;

    constexpr int TD = HD / TM;
    constexpr int NROWG = BM / TM;
    constexpr int NDMG = HD / TD;
    static_assert(NROWG * NDMG == NTHREAD, "lane tiling must fill the block");
    static_assert(NROWG * NDMG == BM, "lane tiling must fill the block");
    static_assert(BN % 2 == 0, "BN must be even");
    static_assert(KS_TOTAL * 2 + VS_TOTAL * 2 * (int)sizeof(__half)
                  + BM * BN * (int)sizeof(__half)
                  + BM * 2 * (int)sizeof(float) <= 49152, "LDS budget");

    __shared__ __attribute__((aligned(32))) int8_t k_buf[2][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half v_buf[2][VS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half p_buf[BN * BM];
    __shared__ __attribute__((aligned(32))) float alpha_buf[BM];
    __shared__ __attribute__((aligned(32))) float l_buf[BM];

    const int tid = threadIdx.x;
    const int64_t m = blockIdx.x * BM + tid;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len) && (h < q_heads);

    float qsv;
    int q_reg[QUADS];
    if constexpr (INQ) {
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        float row_amax = 1e-7f;
        if (valid) {
            const char* qrow = reinterpret_cast<const char*>(q_fp) + (qb + m * q_stride_n) * 2;
            constexpr int NW = HD / 2;
            #pragma unroll
            for (int x = 0; x < NW; ++x) {
                const unsigned u = reinterpret_cast<const unsigned*>(qrow)[x];
                const float f0 = gfx10_q8_2f(u & 0xffffu, q_src_bf16 != 0);
                const float f1 = gfx10_q8_2f(u >> 16, q_src_bf16 != 0);
                row_amax = fmaxf(row_amax, fabsf(f0));
                row_amax = fmaxf(row_amax, fabsf(f1));
            }
        }

        #pragma unroll
        for (int sh = 1; sh < 32; sh <<= 1)
            row_amax = fmaxf(row_amax, __shfl_xor(row_amax, sh));
        qsv = valid ? (row_amax * (1.0f / 127.0f) * sm_scale_log2e) : 0.0f;
        if (valid) {
            const char* qrow = reinterpret_cast<const char*>(q_fp) + (qb + m * q_stride_n) * 2;
            constexpr int NW = HD / 2;
            const float iscale = 127.0f / row_amax;
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) {
                const unsigned a = reinterpret_cast<const unsigned*>(qrow)[2 * dq];
                const unsigned b_ = reinterpret_cast<const unsigned*>(qrow)[2 * dq + 1];
                const float f0 = gfx10_q8_2f(a & 0xffffu, q_src_bf16 != 0);
                const float f1 = gfx10_q8_2f(a >> 16, q_src_bf16 != 0);
                const float f2 = gfx10_q8_2f(b_ & 0xffffu, q_src_bf16 != 0);
                const float f3 = gfx10_q8_2f(b_ >> 16, q_src_bf16 != 0);
                q_reg[dq] = gfx10_q8_pack(
                    gfx10_q8_round(f0 * iscale), gfx10_q8_round(f1 * iscale),
                    gfx10_q8_round(f2 * iscale), gfx10_q8_round(f3 * iscale));
            }
        } else {
            qsv = 0.0f;
        }
    } else {
        qsv = valid
            ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)] : 0.0f;
        if (valid) {
            const int64_t qb = b * q_stride_b + h * q_stride_h;
            const int8_t* qrow = q + qb + m * q_stride_n;
            #pragma unroll
            for (int dq = 0; dq < QUADS / 4; ++dq) {
                const int4 val = *reinterpret_cast<const int4*>(qrow + dq * 16);
                q_reg[dq * 4 + 0] = val.x;
                q_reg[dq * 4 + 1] = val.y;
                q_reg[dq * 4 + 2] = val.z;
                q_reg[dq * 4 + 3] = val.w;
            }
        } else {
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
        }
    }

    float acc[TM][TD];
    float row_m = -3.0e38f, row_l = 0.0f;
    #pragma unroll
    for (int u = 0; u < TM; ++u)
        #pragma unroll
        for (int dd = 0; dd < TD; ++dd) acc[u][dd] = 0.0f;

    auto stage_kv = [&](int dst, int64_t kb0) {
        #pragma unroll 1
        for (int i = tid; i < BN * QUADS / 4; i += NTHREAD) {
            const int r = i / (QUADS / 4), ck4 = i % (QUADS / 4);
            const int64_t n = kb0 + r;
            int4 val = make_int4(0, 0, 0, 0);
            if (n < kv_len) {
                const int8_t* src = k + b * k_stride_b + kvh * k_stride_h + n * k_stride_n + ck4 * 16;
                val = *reinterpret_cast<const int4*>(src);
            }
            *reinterpret_cast<int4*>(&k_buf[dst][r * K_STRIDE + ck4 * 16]) = val;
        }
        #pragma unroll 1
        for (int u = 0; u < (HD * BN / VDSW) / NTHREAD; ++u) {
            const int slot = tid + u * NTHREAD;
            if (slot < HD * BN / VDSW) {

                const int n_local = slot % BN;
                const int dg = slot / BN;
                const int64_t n = kb0 + n_local;
                if ((dg * VDSW) < HD) {
                    if (n < kv_len) {
                        const __half* src = v + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + dg * VDSW;
                        int4 val = *reinterpret_cast<const int4*>(src);
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] = reinterpret_cast<__half*>(&val)[jj];
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] = __half{0};
                    }
                }
            }
        }
    };

    stage_kv(0, 0);
    __syncthreads();

    const int64_t kb_lim = ISC ? min(kv_len, min(qo_len, blockIdx.x * static_cast<int64_t>(BM) + BM)) : kv_len;

    #pragma unroll 1
    for (int64_t kb = 0; kb < kb_lim; kb += BN) {
        const int buf = static_cast<int>((kb / BN) & 1);
        const int64_t nb = kb + BN;

        if ((nb < kb_lim) && !diag_st) {
            stage_kv(buf ^ 1, nb);
        }

        float scr[BN];
        // Whether each key in this tile survives the mask. Kept as a flag rather
        // than encoded in the score: -ffast-math implies -ffinite-math-only,
        // which lets the optimizer decide a float compare against a sentinel is
        // a constant and fold it away. A bool cannot be reasoned about that way,
        // so "dropped" stays observable to the optimizer that could have removed
        // it. It also costs nothing: BN predicates already fit in one VGPR's
        // worth of predicate state alongside the BN scores the loop keeps live.
        bool keep_b[BN];
        {
            #pragma unroll
            for (int j0 = 0; j0 < BN; j0 += 4) {
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                if (!diag_qk)
                #pragma unroll
                for (int dq = 0; dq < QUADS; dq += 4) {
                    const int4 k0 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 0) * K_STRIDE + dq * 4]);
                    const int4 k1 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 1) * K_STRIDE + dq * 4]);
                    const int4 k2 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 2) * K_STRIDE + dq * 4]);
                    const int4 k3 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 3) * K_STRIDE + dq * 4]);
                    s0 = sdot4_i32_i8(q_reg[dq + 0], k0.x, s0); s0 = sdot4_i32_i8(q_reg[dq + 1], k0.y, s0);
                    s0 = sdot4_i32_i8(q_reg[dq + 2], k0.z, s0); s0 = sdot4_i32_i8(q_reg[dq + 3], k0.w, s0);
                    s1 = sdot4_i32_i8(q_reg[dq + 0], k1.x, s1); s1 = sdot4_i32_i8(q_reg[dq + 1], k1.y, s1);
                    s1 = sdot4_i32_i8(q_reg[dq + 2], k1.z, s1); s1 = sdot4_i32_i8(q_reg[dq + 3], k1.w, s1);
                    s2 = sdot4_i32_i8(q_reg[dq + 0], k2.x, s2); s2 = sdot4_i32_i8(q_reg[dq + 1], k2.y, s2);
                    s2 = sdot4_i32_i8(q_reg[dq + 2], k2.z, s2); s2 = sdot4_i32_i8(q_reg[dq + 3], k2.w, s2);
                    s3 = sdot4_i32_i8(q_reg[dq + 0], k3.x, s3); s3 = sdot4_i32_i8(q_reg[dq + 1], k3.y, s3);
                    s3 = sdot4_i32_i8(q_reg[dq + 2], k3.z, s3); s3 = sdot4_i32_i8(q_reg[dq + 3], k3.w, s3);
                }
                for (int tj = 0; tj < 4; ++tj) {
                    int s = (tj == 0) ? s0 : (tj == 1) ? s1 : (tj == 2) ? s2 : s3;
                    const int64_t n = kb + j0 + tj;
                    bool keep = valid && !(ISC && n > m) && (n < kv_len);
                    float bias = 0.0f;
                    if (keep && kMasked) {
                        keep = gfx10_mask_lookup(
                            MASK, mask, MASK_DTYPE, mask_stride_b, mask_stride_h,
                            mask_stride_q, mask_stride_k, b, h, m, n, mask_batch,
                            mask_heads, mask_q_tiles, mask_k_tiles, mask_k_row_pitch,
                            kLog2e, bias);
                    }
                    // Only a kept key is scaled, so the k_scale read stays inside
                    // the buffer: the last tile's padding can address a group
                    // past the end.
                    float sc = 0.0f;
                    if (keep) {
                        const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h +
                                                 static_cast<int>(n / MIN_BLK_K)];
                        // The product is formed and rounded on its own, then the
                        // bias is added, rather than folding the two into one
                        // fma. A mask with no bias (a Boolean one) or no mask at
                        // all would otherwise let the optimizer see the add as a
                        // no-op and reassociate the scale product, rounding it
                        // differently from the case where the bias arrives from
                        // memory. Two masks that mean the same thing then
                        // disagree by an ulp, which is exactly what
                        // test_hip_compact_dense_bool_matches_float_bias
                        // compares. One extra add per key is free next to the
                        // sdots above.
                        const float scaled = static_cast<float>(s) * (qsv * ksj);
                        // A mask bias arrives in the base-two log domain, so it
                        // lands in the same units the exp2 below is fed.
                        sc = scaled + bias;
                    }
                    scr[j0 + tj] = sc;
                    // Both arrays are written on every path, the dropped one
                    // included. keep_b is uninitialized on entry to the key loop,
                    // and setting it only for kept keys would leave a dropped key
                    // holding the *previous tile's* flag, which resurrects it
                    // with a stale score: invisible on a full tile, but it
                    // silently steals weight from the real keys whenever kv_len
                    // is not a multiple of BN.
                    keep_b[j0 + tj] = keep;
                }
            }
        }

        float P_al = 1.0f;
        if (!diag_sm) {
            // A dropped key must not raise the running maximum: the max is what
            // the exponentials are normalized against, so a sentinel leaking in
            // would push every real score to -inf and the whole row to zero.
            float lm = kGfx10MaskedScore;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                if (keep_b[j]) lm = fmaxf(lm, scr[j]);
            }
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            P_al = alpha;
            row_m = gm;
            row_l *= alpha;
            // Normalize with the same probabilities the P*V product consumes.
            // The probabilities are stored as fp16 (that is what p_buf holds and
            // what the fdot2 below multiplies), so summing the fp32 values here
            // would leave the denominator un-rounded while the numerator is
            // rounded -- a row whose V is constant would then come out a few
            // 1e-3 off the value it should return instead of exactly it. This
            // is the same reason the WMMA kernels sum the U8 probabilities
            // rather than the fp32 ones.
            float ps = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                // Dropped keys contribute exactly nothing, so a fully masked row
                // keeps row_l at 0 and the write-back below turns that into a
                // clean zero without a separate validity flag.
                const __half p16 = __float2half(keep_b[j] ? exp2f(scr[j] - row_m) : 0.0f);
                p_buf[j * BM + (int)(m % BM)] = p16;
                ps += __half2float(p16);
            }
            row_l += ps;
        } else {
            #pragma unroll
            for (int j = 0; j < BN; ++j) p_buf[j * BM + (int)(m % BM)] = __float2half(scr[j]);
        }
        alpha_buf[m % BM] = P_al;
        l_buf[m % BM] = row_l;

        __syncthreads();

        if (!diag_pv) {
            const int rg = tid % NROWG;
            const int dgo = tid / NROWG;
            const int r0 = rg * TM;
            #pragma unroll
            for (int u = 0; u < TM; ++u) {
                const float al = alpha_buf[r0 + u];
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd) acc[u][dd] *= al;
            }
            #pragma unroll
            for (int kp = 0; kp < BN / 2; ++kp) {
                const int k = kp * 2;
                unsigned pp[TM];
                #pragma unroll
                for (int up = 0; up < TM / 2; ++up) {

                    const unsigned pk0 = *reinterpret_cast<const unsigned*>(&p_buf[k * BM + r0 + up * 2]);
                    const unsigned pk1 = *reinterpret_cast<const unsigned*>(&p_buf[(k + 1) * BM + r0 + up * 2]);
                    pp[up * 2]     = (pk0 & 0xffffu) | ((pk1 & 0xffffu) << 16);
                    pp[up * 2 + 1] = (pk0 >> 16)      | (pk1 & 0xffff0000u);
                }
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd) {
                    const int d = dgo * TD + dd;
                    const unsigned vv = *reinterpret_cast<const unsigned*>(&v_buf[buf][d * V_STRIDE + k]);
                    #pragma unroll
                    for (int u = 0; u < TM; ++u)
                        acc[u][dd] = fdot2_f32_f16(pp[u], vv, acc[u][dd]);
                }
            }
        }

        __syncthreads();
    }

    if (!diag_wb) {
        const int rg = tid % NROWG;
        const int dgo = tid / NROWG;
        const int r0 = rg * TM;
        #pragma unroll
        for (int u = 0; u < TM; ++u) {
            const int64_t row = blockIdx.x * BM + r0 + u;
            if ((row < qo_len) && (h < q_heads)) {
                // A row whose keys were all dropped has l == 0 and an acc that is
                // already zero; 0 * inf would be NaN, so guard the reciprocal.
                const float l = l_buf[r0 + u];
                const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
                const int64_t base = b * o_stride_b + row * o_stride_n + h * o_stride_h;
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd)
                    out[base + dgo * TD + dd] = gfx10_out_convert<ODT>(acc[u][dd] * inv);
            }
        }
    }
#endif
}

}