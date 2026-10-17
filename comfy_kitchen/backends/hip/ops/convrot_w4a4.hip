// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ConvRot W4A4 on WMMA (iu4: 16x16x32 on gfx12, 16x16x16 on gfx11). See mma.h.
//
// out = (Aq[M, K/2] @ Wq[N, K/2]^T) * x_scale[row] * w_scale[col] + bias
//
// Both operands are signed int4 packed two per byte (low nibble = even index),
// which is the byte layout the iu4 A-fragment consumes directly. The activation
// rotation is fused into the quantizer.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_wmma.h"
#include "../hadamard.h"

namespace comfy::hip_backend {

__global__ void unpack_int4_kernel(const int8_t* __restrict__ q, int8_t* __restrict__ out,
                                   int64_t nbytes) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nbytes) return;
    const int8_t b = q[i];
    // sign-extend each nibble from 4 bits
    const int lo = static_cast<int>(b & 0xF);
    const int hi = static_cast<int>((b >> 4) & 0xF);
    out[2 * i] = static_cast<int8_t>(lo > 7 ? lo - 16 : lo);
    out[2 * i + 1] = static_cast<int8_t>(hi > 7 ? hi - 16 : hi);
}

// W4A4 tile selection.
//
// These three tiles were never retuned on gfx1103: commit 7247847 only made the
// choice per-device instead of per-process, and the numbers came from upstream.
// That is exactly the state the int8 heuristic was in before ck_tools/ck_tile_ab.py,
// and the comment in gemm_wmma.h is blunt about how that went ("three separate
// attempts to do better all measured worse"). So these numbers are a **starting
// point, not a tuned one** -- unlike the int8 heuristic beside them.
//
// The knob below is what makes a sweep possible: this launcher calls
// gemm_wmma_kernel directly rather than going through launch_gemm_wmma, so
// COMFY_GEMM_WMMA_TILE never reached it and every "the int8 tiles are tuned"
// measurement said nothing about W4A4. Routing through launch_gemm_wmma_forced
// reuses the int8 instantiations (they are the same template, only the Mma differs)
// so a forced tile runs the same kernel the heuristic would have run.
template <typename OutT>
static void launch_w4a4(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int K,
                        EpiRowwise epi, hipStream_t stream) {
    const int kbytes = K / 2;
    if (launch_gemm_wmma_forced<MmaInt4, EpiRowwise, OutT>(A, B, C, M, N, kbytes, N, epi,
                                                           stream)) {
        return;
    }
    if (M >= 512 && N >= 128 && K > N) {
        // The deep-K branch now takes BKB=128 instead of 64, and that is the whole
        // point. gfx11's MmaInt4 has **kStepBytes = 8** (gfx12's is 16 -- mma.h's
        // two MmaInt4 differ exactly there, because gfx11 does K=16 per MMA and
        // gfx12 K=32), so BKB=64 gives kSteps = 64/8 = 8, and K=8192 (kbytes 4096)
        // walks **512** K-iterations, each staging only 8 bytes per row. BKB=128
        // doubles that to 16 steps per iteration and halves the trip count -- the
        // same trade the int8 heuristic already makes for the same reason ("BKB=128
        // halves the LDS round trips per K element once K amortizes the coarser
        // tail", gemm_wmma.h).
        //
        // The tile is the shallow-K one, 64x64x128 with 2x2 warps of 2x2. On deep K
        // it beats 256x128 on a second axis too: 64x64 at BKB=128 stages
        // (64+64)*128 = 16 KB of LDS against 256x128's (256+128)*64 = 24 KB, and
        // 24 KB is already over a third of this part's 64 KB, so 256x128 runs few
        // blocks per CU and has nothing to overlap the K loop with.
        //
        // Measured on the 6-WGP 780M, interleaved in one process against the old
        // 256x128, output bit-identical in every case (6-10 samples per point).
        // benchmark_attn/ab_w4a4_deepk.py, K/N/M swept:
        //
        //   M=8192  N=2048  K=2048    0.985x  <- shallow: BKB=128's coarser tail
        //   M=8192  N=2048  K=4096    1.317x     wins, so the old tile stays
        //   M=8192  N=2048  K=8192    1.349x
        //   M=8192  N=2048  K=12288   1.087x
        //   M=8192  K=8192  N=1024    1.340x
        //   M=8192  K=8192  N=8192    1.230x
        //   M=2048  K=8192  N=2048    1.342x
        //   M=16384 K=8192  N=2048    1.369x
        //
        // benchmark_attn/ab_w4a4_deepk2.py is the version to re-run: it fixes the
        // control arm (that launcher could not name its own old tile until
        // launch_gemm_wmma_forced above learned it), adds a **positive control** at
        // kbytes < 2048 where both arms run the same tile and the ratio must land
        // inside noise (it does: 1.000x / 1.003x), and covers the two points the
        // int8 heuristic's own failed retune missed:
        //
        //   K=5120  (SDXL adaLN, inside the int8 regression band)  1.116x
        //   M=18432 (what ComfyUI CFG at B=2 actually reaches)      1.343x
        //   M=2048..16384 at K=8192                                  1.343-1.386x
        //
        // 20 of 20 shapes bit-identical, no shape slower.
        //
        // So the switch is on kbytes. The threshold sits between K=2048 (kbytes
        // 1024, where one BKB=128 tile is most of the row) and K=4096 (kbytes 2048,
        // eight tiles); 2048 is the midpoint of that gap and the measured crossing
        // is there. No Anima or SDXL shape has a K between them, so the exact value
        // only has to be right for the two shapes that exist.
        //
        // **Gated to gfx1103, like the int8 arm beside it in gemm_wmma.h.** Every
        // number above comes from one 6-WGP part, and the same file's int8 heuristic
        // already records what happens when a gfx1103-tuned tile is let loose on the
        // other architectures: "the branch above was tuned against 6 WGPs and is a
        // regression elsewhere." Nothing here is measured on gfx1035 or gfx12, so
        // every other architecture keeps upstream's 256x128x64 unchanged -- which is
        // the only claim backed by a measurement on all of them.
        if (comfy_small_igpu() && kbytes >= 2048) {
            constexpr int BM = 64, BN = 64, BKB = 128;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 2, 2, 2, 2>
                <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, N, epi);
        } else {
            constexpr int BM = 256, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 4, 4>
                <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, N, epi);
        }
    } else if (M >= 96 && N >= 96) {
        constexpr int BM = 128, BN = 128, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, N, epi);
    } else {
        constexpr int BM = 64, BN = 64, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt4, EpiRowwise, OutT, BM, BN, BKB, 2, 2, 2, 2>
            <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, N, epi);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_convrot_w4a4_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* x_scale, const void* w_scale, const void* bias, int bias_code,
    int M, int N, int K, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // The tile loader reads a row 16 bytes at a time and int4 packs two per byte,
    // so the row length K/2 has to be a whole number of 16-byte chunks. An empty
    // GEMM does no reads, so this only applies once there is real work above.
    if (K % 32 != 0) {
        throw std::runtime_error("convrot_w4a4_linear: K=" + std::to_string(K) +
                                 " must be a multiple of 32");
    }

    EpiRowwise epi{static_cast<const float*>(x_scale), static_cast<const float*>(w_scale), 1,
                   bias, bias_code};

    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

    // RDNA2 has no matrix cores, so MmaInt4 is a trap there. But it has
    // v_dot8_i32_i4, which is the same product sum: eight signed 4x4-bit products
    // per instruction against WMMA's sixteen-then-thirty-two-per-MMA, and the
    // operands are already in the layout it wants (signed int4, two per byte,
    // low nibble = even k). Branch on the launching device, the way
    // launch_int8_gemm_kernel branches on comfy_is_gfx10(), so a box holding both
    // an RDNA2 iGPU and a matrix-core dGPU sends each call to the tile that part
    // has rather than deciding once for the process.
    if (comfy_is_gfx10()) {
        // The VALU tile takes the same EpiRowwise the WMMA path above builds, so
        // the operands and the epilogue are identical; only the product sum moves.
        // K is a multiple of 32, which is what launch_int4_gemm_kernel requires.
        launch_int4_gemm_kernel(a, b, c, x_scale, w_scale, 1, bias, bias_code,
                                M, N, K, N /*ldc*/, out_code, stream);
        return;
    }

    if (out_code == kBF16) {
        launch_w4a4<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, epi, stream);
    } else if (out_code == kF16) {
        launch_w4a4<__half>(A, B, static_cast<__half*>(c), M, N, K, epi, stream);
    } else if (out_code == kF32) {
        launch_w4a4<float>(A, B, static_cast<float*>(c), M, N, K, epi, stream);
    } else {
        throw std::runtime_error("convrot_w4a4_linear: unsupported output dtype");
    }
}

// Fused Hadamard rotation + rowwise int4 quantize + nibble pack. Used for both
// the activation and the weight: both rotate along K and quantize per row.
extern "C" void launch_convrot_quant_int4_kernel(
    const void* x, int in_code, void* qout, void* scales,
    int M, int K, int group_size, hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    check_convrot_k(K, group_size, in_code);
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    launch_convrot_quant<true>(
        x, in_code, static_cast<int8_t*>(qout), static_cast<float*>(scales),
        M, K, group_size, stream);
}

// The largest K the fused rotation can stage in LDS, for the dispatch wrappers.
extern "C" int convrot_max_k_host(int in_code) {
    return comfy::hip_backend::convrot_max_k(in_code);
}

extern "C" void launch_unpack_int4_kernel(
    const void* q, void* out, int64_t nbytes, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (nbytes == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    const int threads = 256;
    const int blocks = static_cast<int>((nbytes + threads - 1) / threads);
    unpack_int4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const int8_t*>(q), static_cast<int8_t*>(out), nbytes);
}
