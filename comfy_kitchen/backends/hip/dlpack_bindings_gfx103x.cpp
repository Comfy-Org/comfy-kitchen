// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The RDNA2 SageAttention port's binding layer.
//
// The port arrived as its own CMake project with its own .pyd, because it bound
// through the torch stable ABI while this extension binds through nanobind and has
// no torch dependency at all. Registering it here instead is what lets the port's
// kernels sit in HIP_SOURCES with the rest of the backend, and keeps a gfx1035
// build to a single extension the way every other architecture already is.
//
// The cost of speaking nanobind is that there is no Tensor to hand: every output
// is allocated on the Python side and passed in, and the current stream arrives as
// an integer, the same shape every other kernel in dlpack_bindings.cpp already
// takes. What that buys is the reverse -- nothing here knows about torch, so the
// extension does not either.
//
// The launchers themselves are in ../sage_attention/sage_attn_gfx103x.hip. This
// file only marshals: shapes and strides in, one args struct out. Anything that is
// a decision rather than a value -- the mask descriptor, the tile shape, whether Q
// is quantized in-kernel -- stays next to the kernel, where it was measured.

#include "sage_attention/sage_attn_gfx103x.h"

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

// Defined in dlpack_bindings.cpp, shared rather than duplicated: it is the one
// place the DLPack dtype codes are turned into this backend's quantization codes.
int map_dtype_to_code(const nb::dlpack::dtype& dtype);

// tensor_layout == kHND. Spelled out rather than shared because the port's own
// constants live in the .hip's anonymous namespace, and the two must agree on one
// value; this is the value the Python layer passes for [B, heads, seq, dim].
constexpr int64_t kLayoutHND = 1;

namespace {

using sageattn_gfx103x::AttnArgs;
using sageattn_gfx103x::QuantArgs;

[[noreturn]] void fail(const char* fn, const std::string& msg) {
    throw std::runtime_error(std::string(fn) + ": " + msg);
}

// DTYPE_TO_CODE codes this file cares about.
constexpr int kF16 = 1;
constexpr int kBF16 = 2;
constexpr int kI8 = 4;

void require_dtype(const nb::ndarray<>& t, int want, const char* fn, const char* name) {
    const int got = map_dtype_to_code(t.dtype());
    if (got != want) {
        fail(fn, std::string(name) + " has the wrong dtype");
    }
}

// nanobind's ndarray has no is_contiguous(), and the mask needs the answer: every
// prepared mask form is indexed by arithmetic on the row pitch rather than by
// strides, so a non-contiguous one would be read as the wrong buffer. Recomputing
// the packed strides is the same test, and it is what the caller has to satisfy.
bool is_row_major(const nb::ndarray<>& t) {
    int64_t expect = 1;
    for (int64_t nd = t.ndim(); nd > 0; --nd) {
        const int64_t i = nd - 1;
        if (t.shape(i) == 1) continue;  // a length-1 axis may stride anything
        if (t.stride(i) != expect) return false;
        expect *= t.shape(i);
    }
    return true;
}

void require_ndim(const nb::ndarray<>& t, int want, const char* fn, const char* name) {
    if (t.ndim() != want) {
        fail(fn, std::string(name) + " must be " + std::to_string(want) + "D");
    }
}

// Every kernel here indexes with the extents it is launched with and has no bounds
// of its own, so a short buffer is an out-of-bounds device access rather than an
// error. _C is importable on its own, so that is checked here and not left to the
// Python layer.
void require_elems(const nb::ndarray<>& t, int64_t need, const char* fn, const char* name) {
    if (t.size() < static_cast<size_t>(need)) {
        fail(fn, std::string(name) + " has " + std::to_string(t.size()) +
                    " elements, needs " + std::to_string(need));
    }
}

// The two head dims the port instantiates. launch_attn_cf and
// gfx103x_launch_quant_qk both dispatch `head_dim == 64 ? HD64 : HD128`, so any
// other width would silently run the HD128 kernel over rows of a different length
// and read past the end of K, V and Q.
void require_head_dim(int64_t head_dim, const char* fn) {
    if (head_dim != 64 && head_dim != 128) {
        fail(fn, "head_dim must be 64 or 128, got " + std::to_string(head_dim));
    }
}

// The kernel derives a query head's KV head as `head / (q_heads / kv_heads)`, an
// integer division. kv_heads == 0 divides by zero on the device, and kv_heads not
// dividing q_heads reads a key from a head that does not exist.
void require_head_ratio(int64_t q_heads, int64_t kv_heads, const char* fn) {
    if (kv_heads <= 0 || q_heads % kv_heads != 0) {
        fail(fn, "q_heads=" + std::to_string(q_heads) +
                    " must be a positive multiple of kv_heads=" + std::to_string(kv_heads));
    }
}

// One scale per quant group the kernel indexes, so a short scale buffer is the same
// out-of-bounds read as a short operand row.
int64_t quant_groups(int64_t length, int64_t group) {
    return (length + group - 1) / group;
}

hipStream_t as_stream(uintptr_t stream_ptr) {
    return reinterpret_cast<hipStream_t>(stream_ptr);
}

}  // namespace

// Q and K to int8, with a scale per (b, h, group). Returns nothing: all four
// outputs are allocated by the caller, which is the only way this can share a
// module with the rest of the backend.
//
// query and key are [B, heads, seq, head_dim] in whichever layout tensor_layout
// names; key_mean is the optional per-(b,h,d) mean the port's quantizer subtracts
// and may be empty. When skip_q is set, q_int8 and q_scale are allocated but never
// written, because the attention kernel quantizes Q itself.
void gfx103x_quant_qk_int8(nb::ndarray<> query, nb::ndarray<> key, nb::ndarray<> key_mean,
                           nb::ndarray<> q_int8, nb::ndarray<> q_scale, nb::ndarray<> k_int8,
                           nb::ndarray<> k_scale, int64_t tensor_layout, double sm_scale,
                           int64_t skip_q, uintptr_t stream_ptr) {
    constexpr const char* fn = "gfx103x_quant_qk_int8";

    require_ndim(query, 4, fn, "query");
    require_ndim(key, 4, fn, "key");

    const int code = map_dtype_to_code(query.dtype());
    if (code != kF16 && code != kBF16) {
        fail(fn, "query must be float16 or bfloat16");
    }
    if (map_dtype_to_code(key.dtype()) != code) {
        fail(fn, "key must have the same dtype as query");
    }
    require_dtype(q_int8, kI8, fn, "q_int8");
    require_dtype(k_int8, kI8, fn, "k_int8");
    require_dtype(q_scale, 0, fn, "q_scale");
    require_dtype(k_scale, 0, fn, "k_scale");

    const bool hnd = tensor_layout == kLayoutHND;
    const int64_t batch = query.shape(0);
    const int64_t q_heads = hnd ? query.shape(1) : query.shape(2);
    const int64_t kv_heads = hnd ? key.shape(1) : key.shape(2);
    const int64_t q_len = hnd ? query.shape(2) : query.shape(1);
    const int64_t kv_len = hnd ? key.shape(2) : key.shape(1);
    const int64_t head_dim = query.shape(3);
    require_head_dim(head_dim, fn);
    require_head_ratio(q_heads, kv_heads, fn);
    if (key.shape(3) != head_dim) {
        fail(fn, "key head_dim " + std::to_string(key.shape(3)) +
                    " does not match query head_dim " + std::to_string(head_dim));
    }

    QuantArgs a{};
    a.q = query.data();
    a.k = key.data();
    a.q_int8 = static_cast<int8_t*>(q_int8.data());
    a.k_int8 = static_cast<int8_t*>(k_int8.data());
    a.q_scale = static_cast<float*>(q_scale.data());
    a.k_scale = static_cast<float*>(k_scale.data());
    a.batch = batch;
    a.q_heads = q_heads;
    a.kv_heads = kv_heads;
    a.q_len = q_len;
    a.kv_len = kv_len;
    a.head_dim = head_dim;
    a.q_stride_b = query.stride(0);
    a.q_stride_n = hnd ? query.stride(2) : query.stride(1);
    a.q_stride_h = hnd ? query.stride(1) : query.stride(2);
    a.k_stride_b = key.stride(0);
    a.k_stride_n = hnd ? key.stride(2) : key.stride(1);
    a.k_stride_h = hnd ? key.stride(1) : key.stride(2);
    a.q_groups = (q_len + sageattn_gfx103x::kQuantGroupQ - 1) / sageattn_gfx103x::kQuantGroupQ;
    a.k_groups = (kv_len + sageattn_gfx103x::kQuantGroupK - 1) / sageattn_gfx103x::kQuantGroupK;
    a.sm_scale = sm_scale;
    a.src_bf16 = (code == kBF16) ? 1 : 0;
    a.skip_q = skip_q ? 1 : 0;

    // The key mean is per (b, h, d) and optional; an empty tensor is how the port
    // said "there is none" and is still how it is said here.
    if (key_mean.size() > 0) {
        a.key_mean = key_mean.data();
        require_dtype(key_mean, code, fn, "key_mean");
    }

    if (!a.skip_q) {
        require_elems(q_int8, batch * q_heads * q_len * head_dim, fn, "q_int8");
        require_elems(q_scale, batch * q_heads * a.q_groups, fn, "q_scale");
    }
    require_elems(k_int8, batch * kv_heads * kv_len * head_dim, fn, "k_int8");
    require_elems(k_scale, batch * kv_heads * a.k_groups, fn, "k_scale");

    sageattn_gfx103x::gfx103x_launch_quant_qk(a, as_stream(stream_ptr));
}

// The attention itself.
//
// Q and K are int8, V is fp16 in [B, kv_heads, seq, head_dim] and output is fp16 or
// bf16; all four keep the caller's own layout and are read and written in place.
//
// query is 4D when it is already packed, and the kernel takes its geometry from
// there. When it is not, q_fp carries an fp16/bf16 Q source instead: the geometry
// comes from q_fp, and the kernel quantizes Q in place -- but only for short KV,
// since that is the condition it was measured under, and the launcher decides it.
void gfx103x_qk_int8_sv_attn(nb::ndarray<> query, nb::ndarray<> key, nb::ndarray<> value,
                             nb::ndarray<> output, nb::ndarray<> q_scale, nb::ndarray<> k_scale,
                             int64_t tensor_layout, int64_t is_causal, double sm_scale,
                             nb::ndarray<> q_fp, int64_t mask_mode, int64_t mask_dtype,
                             nb::ndarray<> mask, uintptr_t stream_ptr) {
    constexpr const char* fn = "gfx103x_qk_int8_sv_attn";

    require_ndim(key, 4, fn, "key");
    require_ndim(value, 4, fn, "value");
    require_ndim(output, 4, fn, "output");
    require_dtype(key, kI8, fn, "key");
    require_dtype(value, kF16, fn, "value");

    const bool q_packed = query.ndim() == 4;
    if (q_packed) {
        require_dtype(query, kI8, fn, "query");
    } else if (q_fp.size() == 0) {
        // Neither a packed Q nor a source to quantize one from: there is nothing
        // for the kernel to read as Q.
        fail(fn, "query is not 4D and q_fp is empty, so there is no Q");
    }

    // Geometry comes from whichever of the two Q operands is the real one.
    const nb::ndarray<>& qgeo = q_packed ? query : q_fp;
    if (!q_packed) {
        require_ndim(q_fp, 4, fn, "q_fp");
        const int code = map_dtype_to_code(q_fp.dtype());
        if (code != kF16 && code != kBF16) {
            fail(fn, "q_fp must be float16 or bfloat16");
        }
    }

    const int out_code = map_dtype_to_code(output.dtype());
    if (out_code != kF16 && out_code != kBF16) {
        fail(fn, "output must be float16 or bfloat16");
    }

    AttnArgs a{};
    a.batch = qgeo.shape(0);
    a.q_heads = qgeo.shape(1);
    a.kv_heads = key.shape(1);
    a.qo_len = qgeo.shape(2);
    a.kv_len = key.shape(2);
    a.head_dim = qgeo.shape(3);

    // Same reasons as in gfx103x_quant_qk_int8: the tile is instantiated per head
    // dim, the KV head of a query head is an integer division of the head counts,
    // and K/V/scales are read at the extents named here. _C is importable on its
    // own, so this has to be refused here rather than left to the Python layer.
    require_head_dim(a.head_dim, fn);
    require_head_ratio(a.q_heads, a.kv_heads, fn);
    if (key.shape(3) != a.head_dim || value.shape(3) != a.head_dim) {
        fail(fn, "key/value head_dim " + std::to_string(key.shape(3)) + "/" +
                    std::to_string(value.shape(3)) + " does not match query head_dim " +
                    std::to_string(a.head_dim));
    }
    require_elems(key, a.batch * a.kv_heads * a.kv_len * a.head_dim, fn, "key");
    require_elems(value, a.batch * a.kv_heads * a.kv_len * a.head_dim, fn, "value");

    if (q_packed) {
        a.q = static_cast<const int8_t*>(query.data());
        a.q_stride_b = query.stride(0);
        a.q_stride_n = query.stride(2);
        a.q_stride_h = query.stride(1);
    } else {
        a.q_fp = q_fp.data();
        a.q_fp_stride_b = q_fp.stride(0);
        a.q_fp_stride_n = q_fp.stride(2);
        a.q_fp_stride_h = q_fp.stride(1);
        a.q_fp_bf16 = (map_dtype_to_code(q_fp.dtype()) == kBF16) ? 1 : 0;
        a.has_q_fp = 1;
    }

    a.k = static_cast<const int8_t*>(key.data());
    a.k_stride_b = key.stride(0);
    a.k_stride_n = key.stride(2);
    a.k_stride_h = key.stride(1);

    // V and the scales are read in the caller's own layouts; nothing is transposed
    // and nothing is staged, which is why this port needs no quantizer of its own
    // for V.
    a.v = static_cast<const __half*>(value.data());
    a.v_stride_b = value.stride(0);
    a.v_stride_n = value.stride(1);
    a.v_stride_h = value.stride(2);

    a.out = output.data();
    a.o_stride_b = output.stride(0);
    a.o_stride_seq = output.stride(1);
    a.o_stride_head = output.stride(2);
    a.layout_is_hnd = tensor_layout == kLayoutHND ? 1 : 0;
    a.out_bf16 = (out_code == kBF16) ? 1 : 0;

    // The prepass path hands over an empty q_scale to say "no per-block scale",
    // which is how it said so before; the launcher zeroes the strides to match.
    require_dtype(q_scale, 0, fn, "q_scale");
    require_dtype(k_scale, 0, fn, "k_scale");
    // The pointers, not just the strides below. Both are needed for every call, and a
    // launcher that takes raw pointers cannot report a null one -- it dequantizes
    // against address zero and returns a plausible-looking answer.
    a.q_scale = static_cast<const float*>(q_scale.data());
    a.k_scale = static_cast<const float*>(k_scale.data());
    if (q_scale.size() > 0) {
        a.qs_stride_b = q_scale.stride(0);
        a.qs_stride_h = q_scale.stride(1);
    }
    a.ks_stride_b = k_scale.stride(0);
    a.ks_stride_h = k_scale.stride(1);

    // One scale per quant group, at the same extents as the packed operand. An
    // empty q_scale is the "no per-block scale" case the prepass hands over and the
    // launcher zeroes the strides for; k_scale is never optional here.
    require_elems(k_scale,
                  a.batch * a.kv_heads *
                      quant_groups(a.kv_len, sageattn_gfx103x::kQuantGroupK),
                  fn, "k_scale");
    if (q_scale.size() > 0) {
        require_elems(q_scale,
                      a.batch * a.q_heads *
                          quant_groups(a.qo_len, sageattn_gfx103x::kQuantGroupQ),
                      fn, "q_scale");
    }

    a.sm_scale = sm_scale;
    a.causal = is_causal ? 1 : 0;
    a.q_is_packed = q_packed ? 1 : 0;

    // The mask's rank, extents, strides and contiguity are handed over rather than
    // interpreted: what a mode means, and which of those it requires, is a
    // property of the kernel, so the checks live next to it.
    a.mask_mode = static_cast<int>(mask_mode);
    a.mask_dtype = static_cast<int>(mask_dtype);
    if (mask_mode != 0) {
        a.mask = mask.data();
        a.mask_ndim = mask.ndim();
        a.mask_contiguous = is_row_major(mask) ? 1 : 0;
        for (int i = 0; i < mask.ndim() && i < 4; ++i) {
            a.mask_shape[i] = mask.shape(i);
            a.mask_stride[i] = mask.stride(i);
        }
    }

    require_elems(output, a.batch * a.q_heads * a.qo_len * a.head_dim, fn, "output");

    sageattn_gfx103x::gfx103x_launch_attn(a, as_stream(stream_ptr));
}

void register_gfx103x_ops(nb::module_& m) {
    // The names are the port's own rather than a mainline one, because the operand
    // layouts are its own: V read in place as fp16, a skip-Q flag, and the mask
    // modes below. Registration is unconditional and on every architecture, because
    // a module has no per-arch scoping: outside __GFX10__ the kernels behind these
    // compile to empty bodies, so on gfx11/gfx12 they exist and do nothing. The
    // Python-side gate is what keeps them off a non-RDNA2 device.
    m.def("gfx103x_quant_qk_int8", &gfx103x_quant_qk_int8, nb::arg("query"), nb::arg("key"),
          nb::arg("key_mean"), nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"),
          nb::arg("k_scale"), nb::arg("tensor_layout"), nb::arg("sm_scale"), nb::arg("skip_q"),
          nb::arg("stream_ptr"));

    m.def("gfx103x_qk_int8_sv_attn", &gfx103x_qk_int8_sv_attn, nb::arg("query"), nb::arg("key"),
          nb::arg("value"), nb::arg("output"), nb::arg("q_scale"), nb::arg("k_scale"),
          nb::arg("tensor_layout"), nb::arg("is_causal"), nb::arg("sm_scale"), nb::arg("q_fp"),
          nb::arg("mask_mode"), nb::arg("mask_dtype"), nb::arg("mask"), nb::arg("stream_ptr"));
}

