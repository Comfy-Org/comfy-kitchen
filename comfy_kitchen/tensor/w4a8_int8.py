# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped W4A8 on the tuned INT8 CUTLASS GEMM (symmetric default).

Calibration-free 4-bit weight storage at near-int8-tensor-core speed, reusing comfy's
tuned INT8 GEMM. Weights are ConvRot-rotated (data-free Hadamard) then quantized per
``group_size``. Default = symmetric group-16 with a per-tensor Lloyd-Max 16-level codebook
+ fp8 (e4m3) group scales: ~0.073 weight relL2 on real DiT weights (NVFP4 0.094),
~0.56 B/elem, no zero-point correction -> ~1.09x int8 speed. ``symmetric=False`` adds a
rank-(K/group) zero-point correction (~1.4x int8) and is dominated by the default.

At matmul the int4 weights dequant to "grouped int8" ``round((q-8)*s_rel)`` -- group scale
folded in, per-channel ``s_channel`` left for the GEMM epilogue -- feeding the INT8 GEMM.
Symmetric packs ``q_signed+8``; asymmetric adds the zero-point back as ``Sx @ Cᵀ``.
Weights stay int4 in memory; the int8 dequant target is a transient per-call buffer.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

import torch

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_op,
)
from .int8 import _dtype_code
from .int8_utils import _build_hadamard, _rotate_weight

logger = logging.getLogger(__name__)

# The tuned W4A8 kernels live in the compiled CUDA backend. Without it (ROCm/AMD, CPU)
# the layout dispatches dequant + INT8 GEMM through the registry (Triton, else eager) and
# rotates in portable torch -- runnable on any torch device.
try:
    from comfy_kitchen.backends.cuda import _C as _CUDA_C  # noqa: F401
    _CUDA_AVAILABLE = True
except Exception as _e:  # pragma: no cover - non-CUDA builds
    _CUDA_AVAILABLE = False
    # Loud only when CUDA hardware is present but the extension failed to load; quiet on
    # ROCm/CPU where the fallback is expected.
    if torch.cuda.is_available():
        logger.warning(
            "W4A8: CUDA backend import failed though CUDA is available (%s); falling "
            "back to the portable Triton/eager path (slower).", _e)


def _convrot_rotate(w: torch.Tensor, groupsize: int) -> torch.Tensor:
    """Involutory ConvRot (Regular-Hadamard) rotation. Uses the CUDA kernel when
    available, else the portable int8_utils implementation (bit-matches to ~bf16 eps)."""
    if _CUDA_AVAILABLE and w.is_cuda:
        from comfy_kitchen.backends.cuda import rotate_int8_convrot_weight
        return rotate_int8_convrot_weight(w, groupsize)
    h = _build_hadamard(groupsize, device=w.device, dtype=w.dtype)
    return _rotate_weight(w, h, groupsize)

# Chunked fused path: dequant int4->int8 in L2-sized column chunks feeding the strided
# int8 GEMM, so each int8 chunk stays cache-resident (no full [N,K] global round-trip).
# COMFY_KITCHEN_W4A8_CHUNKED=0 forces the 2-pass path.
_W4A8_CHUNKED = os.environ.get("COMFY_KITCHEN_W4A8_CHUNKED", "1") != "0"


class AsymW4A8Int8Layout(QuantizedLayout):
    """Asymmetric grouped int4 weights, run on the tuned int8 GEMM."""

    MIN_SM_VERSION = (8, 0)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        # scale (inherited) holds s_rel: per-group relative scale [N, K//group] fp32.
        s_channel: torch.Tensor = None   # [N] fp32 per-channel scale
        correction: torch.Tensor = None  # [K//group, N] = 8*s_g+min (asym only; None=symmetric)
        codebook: torch.Tensor = None    # [16] fp32 non-uniform levels (None=uniform q-8)
        group_size: int = 16
        convrot_groupsize: int = 256
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            f = ["scale", "s_channel"]
            if self.correction is not None:
                f.append("correction")
            if self.codebook is not None:
                f.append("codebook")
            return f

        def _validate_tensor_fields(self):
            pass

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float8_e4m3fn,
        codebook: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        n, k = tensor.shape
        # K must be a multiple of 16: the CUDA dequant kernel is vectorized (uint2 load ->
        # uint4 store, 16 int8/thread, n_vec = N*K/16), so a non-16 K truncates the tail and
        # misaligns the row start. G must be >=4 and in {4,8,16} or a multiple of 16: the
        # kernel loads at most 4 group scales per 16-col vec (G=2 would mis-scale cols 8-15).
        if (k % 16 != 0 or k % group_size != 0 or group_size < 4
                or (16 % group_size != 0 and group_size % 16 != 0)):
            raise ValueError(
                f"K={k} must be a multiple of 16 and G={group_size} >=4 & "
                f"(divides 16 or mult of 16)")
        groups = k // group_size

        # Data-free rotation (paired with the fused convrot activation quant).
        w = _convrot_rotate(tensor, convrot_groupsize).float().view(n, groups, group_size)

        cb = None
        if symmetric and codebook:
            # Non-uniform 16-level codebook (Lloyd-Max, calibration-free). ConvRot makes
            # weights ~Gaussian; a matched codebook beats uniform int4 ~14% at coarse
            # groups, same storage/speed. Code indexes cb directly; kernel does cb[q]*s_rel.
            s_g = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # group absmax
            wn = (w / s_g)  # [n, groups, group] normalized to [-1, 1]
            cb = cls._fit_codebook(wn)  # [16] fp32
            q = cls._assign_codes(wn, cb)  # [n,groups,group] indices
            # Least-squares refit of the group scale given the codes (absmax is only
            # optimal when the extreme code sits at +-1, which Lloyd-Max levels don't).
            for _ in range(3):
                cbq = cb[q]
                s_g = ((w * cbq).sum(-1, keepdim=True)
                       / (cbq * cbq).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
                q = cls._assign_codes(w / s_g, cb)
            q_u = q.to(torch.int32).view(n, k)  # [0,15]
            wshift = (cb[q] * s_g)
            correction = None
        elif symmetric:
            # Symmetric grouped int4 (levels -8..7); no zero-point => no correction pass.
            # Quality beats NVFP4 at group 16 (~0.085 vs 0.093 relL2) calibration-free.
            s_g = (w.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
            q_signed = (w / s_g).round().clamp(-8, 7).to(torch.int32)
            # pack q_signed+8 as uint4; the dequant kernel computes (q-8)*s_rel = q_signed*s_rel
            q_u = (q_signed + 8).view(n, k)
            wshift = q_signed * s_g
            correction = None
        else:
            mn = w.amin(dim=-1, keepdim=True)
            s_g = ((w.amax(dim=-1, keepdim=True) - mn) / 15.0).clamp(min=1e-8)
            q_u = ((w - mn) / s_g).round().clamp(0, 15).to(torch.int32).view(n, k)
            wshift = (q_u.view(n, groups, group_size) - 8) * s_g
            correction = (8.0 * s_g + mn).squeeze(-1).t().contiguous().to(orig_dtype)  # [groups, N]

        s_channel = (wshift.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)  # [N]
        s_rel = (s_g.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()  # [N, groups]
        # fp8 e4m3 group scale halves the scale metadata (~half int8 storage),
        # still beats NVFP4 (~0.089 vs 0.093); fp32 keeps max quality (~0.085).
        if scale_dtype != torch.float32:
            s_rel = s_rel.to(scale_dtype).contiguous()
        if cb is not None:
            # Final code assignment against the values the kernel actually decodes
            # (round(clamp(cb_j * s_rel)) * s_channel, s_rel in its stored dtype) -- folds
            # the fp8-scale and int8 re-rounding into the choice (0.0730 -> 0.0727 g16-fp8).
            levels = (cb.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)
            q_u = cls._assign_grid(w, levels, s_channel).view(n, k)
        # pack uint4: even col -> low nibble, odd col -> high nibble
        qpacked = ((q_u[:, 0::2] & 0xF) | ((q_u[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()

        params = cls.Params(
            scale=s_rel,
            s_channel=s_channel.float().contiguous(),
            correction=correction,
            codebook=cb,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
        )
        return qpacked, params

    @staticmethod
    def _assign_codes(xn: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        """Nearest-codebook index, looping the 16 levels to avoid a [..., 16] distance
        tensor. int32 (not uint8 -- byte index tensors are treated as boolean masks)."""
        best = (xn - cb[0]).abs()
        idx = torch.zeros_like(xn, dtype=torch.int32)
        for j in range(1, cb.numel()):
            d = (xn - cb[j]).abs()
            m = d < best
            best = torch.where(m, d, best)
            idx = torch.where(m, j, idx)
        return idx

    @staticmethod
    def _assign_grid(wv: torch.Tensor, levels: torch.Tensor, s_channel: torch.Tensor) -> torch.Tensor:
        """Nearest decoded-int8 level: argmin_j |wv/s_channel - levels[n,g,j]|, looped
        over j. `levels` is [N, groups, 16] (the per-channel/group decoded values)."""
        t = wv / s_channel.view(-1, 1, 1)
        best = (t - levels[..., 0:1].expand_as(wv)).abs()
        idx = torch.zeros_like(wv, dtype=torch.int32)
        for j in range(1, 16):
            d = (t - levels[..., j:j + 1].expand_as(wv)).abs()
            m = d < best
            best = torch.where(m, d, best)
            idx = torch.where(m, j, idx)
        return idx

    @staticmethod
    def _fit_codebook(wn: torch.Tensor, levels: int = 16, iters: int = 25,
                      sample: int = 300000) -> torch.Tensor:
        """MSE-optimal (Lloyd-Max) 16-level codebook on normalized weights, data-free."""
        x = wn.flatten()
        if x.numel() > sample:
            idx = torch.randint(0, x.numel(), (sample,), device=x.device)
            x = x[idx]
        x = x.float()
        cb = torch.quantile(x, torch.linspace(0, 1, levels, device=x.device))
        for _ in range(iters):
            a = (x.unsqueeze(-1) - cb).abs().argmin(-1)
            new = cb.clone()
            for j in range(levels):
                m = a == j
                if m.any():
                    new[j] = x[m].mean()
            cb = new
        return cb.contiguous()

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        # Reference dequant (original-basis weight); for fallbacks/inspection.
        n, k_half = qdata.shape
        k = k_half * 2
        g = params.group_size
        groups = k // g
        b = qdata.to(torch.int32) & 0xFF
        q = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
        q[:, 0::2] = b & 0xF
        q[:, 1::2] = (b >> 4) & 0xF
        s_rel = params.scale.float()  # fp8/fp32 -> fp32
        s_g = (s_rel * params.s_channel.unsqueeze(1)).unsqueeze(-1)  # [N,groups,1]
        if params.codebook is not None:
            # w = codebook[q] * s_g (q is a direct [0,15] index)
            lvl = params.codebook.to(q.device).float()[q.view(n, groups, g)]
            w_rot = (lvl * s_g).view(n, k)
        else:
            qc = (q.view(n, groups, g).float() - 8.0) * s_g
            if params.correction is None:
                w_rot = qc.view(n, k)  # symmetric: w = (q-8)*s_g
            else:
                corr = params.correction.t().unsqueeze(-1).float()  # [N,groups,1] = 8*s_g + min
                w_rot = (qc + corr).view(n, k)  # w = q*s_g + min = (q-8)*s_g + corr
        # undo rotation (involutory) -> storage-basis weight [N, K]. The base
        # QuantizedTensor.dequantize() applies the lazy-transpose flag (returns W.T for a
        # transposed tensor), so this must always return the [N, K] storage orientation.
        return _convrot_rotate(w_rot.to(params.orig_dtype), params.convrot_groupsize).to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor):
        p = qtensor._params
        return qtensor._qdata, p.scale, p.s_channel, p.correction

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        # Export-only (checkpoint serialization). The ``_s_rel`` suffix intentionally does
        # NOT match the ``scale`` field name: it names the per-group *relative* scale to
        # distinguish it from ``_s_channel``, and matches the checkpoint format the loader
        # reads. On load, ComfyUI's ``_load_quantized_module`` remaps weight_s_rel->scale,
        # weight_s_channel->s_channel, etc. explicitly (not by field name). In-memory
        # flatten/unflatten (__tensor_flatten__) uses field names, so it is unaffected.
        out = {
            "": qdata,
            "_s_rel": params.scale,
            "_s_channel": params.s_channel,
        }
        if params.correction is not None:
            out["_correction"] = params.correction
        if params.codebook is not None:
            out["_codebook"] = params.codebook
        return out

    @classmethod
    def requantize_kwargs(cls, qtensor: QuantizedTensor) -> dict[str, object]:
        # Preserve quant options across the dequantize->patch->requantize LoRA path.
        p = qtensor._params
        return {"group_size": p.group_size, "convrot_groupsize": p.convrot_groupsize,
                "symmetric": p.correction is None, "codebook": p.codebook is not None}


def _storage_dequant(weight):
    """Dequantize to the [N, K] storage orientation, undoing the lazy-transpose flag.
    The GEMM fallbacks compute x @ W.T from the [N, K] weight, so they must not see the
    (K, N) orientation that ``weight.dequantize()`` returns for a transposed tensor."""
    w = weight.dequantize()
    return w.t() if weight._params.transposed else w


def _w4a8_int8_matmul_portable(x, weight, bias, out_dtype):
    """Backend-agnostic W4A8 linear via the registry (Triton on AMD, else eager). Symmetric
    weights take the dequant + INT8 GEMM path; asymmetric has no INT8-GEMM form so it falls
    back to dequantize + F.linear."""
    from comfy_kitchen.registry import registry

    p = weight._params
    if p.correction is not None:
        return torch.nn.functional.linear(x, _storage_dequant(weight).to(x.dtype), bias)
    kwargs = {
        "x": x,
        "qdata": weight._qdata,
        "s_rel": p.scale,
        "s_channel": p.s_channel,
        "bias": bias,
        "group_size": p.group_size,
        "convrot_groupsize": p.convrot_groupsize,
    }
    impl = registry.get_implementation("w4a8_int8_linear", kwargs=kwargs)
    return impl(
        x, weight._qdata, p.scale, p.s_channel,
        codebook=p.codebook, bias=bias, group_size=p.group_size,
        convrot_groupsize=p.convrot_groupsize, out_dtype=out_dtype,
    )


def _w4a8_int8_matmul(x, weight, bias, out_dtype):
    # Portable path (Triton on AMD, else pure-torch eager) when the tuned CUDA kernels
    # are unavailable or the input is not on CUDA.
    if not (_CUDA_AVAILABLE and x.is_cuda):
        return _w4a8_int8_matmul_portable(x, weight, bias, out_dtype)

    from comfy_kitchen.backends.cuda import (
        _C,
        _empty_cuda_tensor,
        _wrap_for_dlpack,
        quantize_int8_rowwise_convrot,
    )

    p = weight._params
    n, k_half = weight._qdata.shape
    k = k_half * 2
    g = p.group_size
    groups = k // g
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]

    sp = torch.cuda.current_stream(x.device).cuda_stream
    cb = _wrap_for_dlpack(p.codebook) if p.codebook is not None else None
    # fused rotate + int8 activation quant
    xq, xs = quantize_int8_rowwise_convrot(x2, p.convrot_groupsize)
    out = torch.empty(m, n, dtype=out_dtype, device=x.device)
    # Bias fuses into the GEMM epilogue (D = acc*xs*s_channel + bias). Cast fresh each
    # call -- don't cache on the bias tensor (inference tensors have no _version).
    biasf = bias.float().contiguous() if bias is not None else None

    # Chunked fused path (symmetric codebook, fp8 scale): dequant int4->int8 in
    # L2-sized column chunks feeding the strided int8 GEMM -- no full [N,K] round-trip.
    if (_W4A8_CHUNKED and p.correction is None and p.codebook is not None
            and p.scale.dtype == torch.float8_e4m3fn):
        from comfy_kitchen.backends.cuda import _int4_int8_weight_chunk_cols
        chunk_cols = _int4_int8_weight_chunk_cols(m, n)
        workspace = torch.empty(min(chunk_cols, n), k, dtype=torch.int8, device=x.device)
        ok = _C.w4a8_codebook_gemm_chunked(
            _wrap_for_dlpack(xq), _wrap_for_dlpack(weight._qdata),
            _wrap_for_dlpack(p.scale.view(torch.uint8)), cb,
            _wrap_for_dlpack(p.s_channel), _wrap_for_dlpack(xs.reshape(m).contiguous()),
            _wrap_for_dlpack(biasf) if biasf is not None else None,
            _wrap_for_dlpack(workspace), _wrap_for_dlpack(out),
            g, chunk_cols, _dtype_code(out_dtype), sp)
        if ok:
            return out.reshape(*x.shape[:-1], n)

    # 2-pass fallback: full int4 -> int8 dequant, then the int8 GEMM (+ asym correction).
    int8_w = torch.empty(n, k, dtype=torch.int8, device=x.device)
    if p.scale.dtype == torch.float8_e4m3fn:
        _C.dequant_int4_grouped_to_int8_e4m3(
            _wrap_for_dlpack(weight._qdata), _wrap_for_dlpack(p.scale.view(torch.uint8)),
            cb, _wrap_for_dlpack(int8_w), g, sp)
    else:
        _C.dequant_int4_grouped_to_int8(
            _wrap_for_dlpack(weight._qdata), _wrap_for_dlpack(p.scale), cb,
            _wrap_for_dlpack(int8_w), g, sp)
    bf = biasf if biasf is not None else _empty_cuda_tensor(x.device, torch.float32)
    used = _C.cutlass_int8_dequant(
        _wrap_for_dlpack(xq), _wrap_for_dlpack(int8_w), _wrap_for_dlpack(xs),
        _wrap_for_dlpack(p.s_channel), _wrap_for_dlpack(bf),
        _wrap_for_dlpack(out), _dtype_code(out_dtype), sp)
    if not used:
        return torch.nn.functional.linear(x, _storage_dequant(weight), bias)

    # asymmetric zero-point correction (rank-(K/group)); symmetric path has none.
    if p.correction is not None:
        sx = (xq.view(m, groups, g).sum(-1, dtype=torch.int32).to(out_dtype) * xs.to(out_dtype))
        out.addmm_(sx, p.correction.to(out_dtype))
    return out.reshape(*x.shape[:-1], n)


def _is_w4a8(w):
    return isinstance(w, QuantizedTensor) and w._layout_cls == "AsymW4A8Int8Layout"


@register_layout_op(torch.ops.aten.t.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_t(qt, args, kwargs):
    inp = args[0]
    if not isinstance(inp, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = inp._params
    new = dataclasses.replace(
        old, orig_shape=(old.orig_shape[1], old.orig_shape[0]), transposed=not old.transposed)
    return QuantizedTensor(inp._qdata, "AsymW4A8Int8Layout", new)


@register_layout_op(torch.ops.aten.linear.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_linear(qt, args, kwargs):
    x, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    if not _is_w4a8(weight) or getattr(weight._params, "transposed", False):
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))
    if isinstance(x, QuantizedTensor):
        x = x.dequantize()
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)
    return _w4a8_int8_matmul(x, weight, bias, out_dtype)


@register_layout_op(torch.ops.aten.mm.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_mm(qt, args, kwargs):
    x, weight = args[0], args[1]
    # F.linear -> x @ W.t(): weight arrives transposed, storage still [N,K].
    if not _is_w4a8(weight) or not getattr(weight._params, "transposed", False):
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))
    if isinstance(x, QuantizedTensor):
        x = x.dequantize()
    return _w4a8_int8_matmul(x, weight, None, weight._params.orig_dtype)


@register_layout_op(torch.ops.aten.addmm.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_addmm(qt, args, kwargs):
    bias, x, weight = args[0], args[1], args[2]
    if not _is_w4a8(weight) or not getattr(weight._params, "transposed", False):
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))
    if isinstance(x, QuantizedTensor):
        x = x.dequantize()
    return _w4a8_int8_matmul(x, weight, bias, weight._params.orig_dtype)
