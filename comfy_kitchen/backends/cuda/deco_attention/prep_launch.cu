// SPDX-License-Identifier: Apache-2.0
#include "prep_internal.cuh"

namespace comfy_deco {
namespace {
__global__ void clear_reductions(PrepBuffers out, uint32_t* counters, int splits) {
    for (int i = threadIdx.x; i < splits; i += blockDim.x) counters[i] = 0;
    if (threadIdx.x < 6) out.maxima[threadIdx.x] = 0;
    if (threadIdx.x == 0) *out.v4_scale = 0;
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
    w.v4_scale = reserve(sizeof(float));
    w.ut = reserve(size_t(heads) * 32 * 128 * sizeof(uint16_t));
    w.side = reserve(2 * units * layout::scale_side_bytes);
    w.chosen_scales = reserve(units * 8 * sizeof(float));
    w.maxima = reserve(6 * sizeof(uint32_t));
    w.split = reserve((units % sms) * (layout::split_pair_bytes + sizeof(uint32_t)));
    w.bytes = reserve(0);
    return w;
}

cudaError_t prepare(const void* q, const void* k, const void* v, const int32_t* rows,
                    const void* rotation, void* workspace, int sequence, int heads, int sms,
                    int64_t qt, int64_t qh, int64_t kt, int64_t kh, int64_t vt, int64_t vh,
                    cudaStream_t stream) {
    const auto w = prep_layout(sequence, heads, sms);
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
    auto* counters = reinterpret_cast<uint32_t*>(p + w.split + splits * layout::split_pair_bytes);
    clear_reductions<<<1, 256, 0, stream>>>(out, counters, splits);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = basis_and_v(shape, query, key, value, rows, rotation, out, sms, stream);
    if (err != cudaSuccess) return err;
    return pack_operands(shape, query, key, value, rotation, out, sms, stream);
}
} // namespace comfy_deco
