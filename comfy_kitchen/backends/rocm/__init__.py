import re
import sys
import torch

__all__ = [
    "apply_rope",
    "apply_rope1",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "scaled_mm_mxfp8",
    "scaled_mm_nvfp4",
]

_FP8_FNUZ = torch.float8_e4m3fnuz
_FP8_MAX  = torch.finfo(_FP8_FNUZ).max


def _gfx_arch():
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(props, "gcnArchName", None)
        if arch:
            return arch.split(":")[0]
        m = re.search(r"gfx\d+[a-z]?", props.name.lower())
        return m.group(0) if m else None
    except Exception:
        return None


def _gfx_num(arch):
    if not arch:
        return 0
    m = re.search(r"\d+", arch)
    return int(m.group(0)) if m else 0


def _supports_fp8_scaled_mm():
    n = _gfx_num(_gfx_arch())
    return n >= 1100 or (940 <= n <= 942)


def quantize_per_tensor_fp8(x, scale, output_type=torch.float8_e4m3fn):
    if not x.is_contiguous():
        x = x.contiguous()
    xf = x.float() * (1.0 / scale.float())
    xf = xf.clamp(-_FP8_MAX, _FP8_MAX)
    return xf.to(_FP8_FNUZ)


def dequantize_per_tensor_fp8(x, scale, output_type=torch.bfloat16):
    return (x.float() * scale.float()).to(output_type)


def quantize_nvfp4(x, per_tensor_scale, epsilon=0.0, pad_16x=False):
    from comfy_kitchen.backends.eager import quantize_nvfp4 as _q
    return _q(x, per_tensor_scale, epsilon, pad_16x)


def dequantize_nvfp4(qx, per_tensor_scale, block_scales, output_type=torch.bfloat16):
    from comfy_kitchen.backends.eager import dequantize_nvfp4 as _dq
    return _dq(qx, per_tensor_scale, block_scales, output_type)


def scaled_mm_nvfp4(a, b, tensor_scale_a, tensor_scale_b,
                    block_scale_a, block_scale_b,
                    bias=None, out_dtype=torch.bfloat16, alpha=None):
    from comfy_kitchen.backends.eager import dequantize_nvfp4 as _dq
    a_fp = _dq(a, tensor_scale_a, block_scale_a, torch.bfloat16)
    b_fp = _dq(b, tensor_scale_b, block_scale_b, torch.bfloat16)
    out = torch.mm(a_fp, b_fp.t())
    if bias is not None:
        out = out + bias.to(out.dtype)
    if alpha is not None:
        out = out * alpha.float()
    return out.to(out_dtype)


def quantize_mxfp8(x, pad_32x=False):
    from comfy_kitchen.backends.eager import quantize_mxfp8 as _q
    return _q(x, pad_32x)


def dequantize_mxfp8(qx, block_scales, output_type=torch.bfloat16):
    from comfy_kitchen.backends.eager import dequantize_mxfp8 as _dq
    return _dq(qx, block_scales, output_type)


def scaled_mm_mxfp8(a, b, block_scale_a, block_scale_b,
                    bias=None, out_dtype=torch.bfloat16):
    if _supports_fp8_scaled_mm():
        # ROCm 7 / RDNA4: hipBLASLt supports float8_e4m3fn (OCP) with
        # TensorWise scaling only. float8_e4m3fnuz is not supported.
        # Layout: a must be row-major (contiguous), b must be col-major
        # (.t() without .contiguous() gives the col-major view hipBLASLt needs).
        a_fp8 = a.contiguous() if a.dtype == torch.float8_e4m3fn else a.to(torch.float8_e4m3fn)
        b_fp8 = b if b.dtype == torch.float8_e4m3fn else b.to(torch.float8_e4m3fn)
        scale_a = block_scale_a.float().max().reshape(())
        scale_b = block_scale_b.float().max().reshape(())
        return torch._scaled_mm(
            a_fp8,
            b_fp8.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
            bias=bias,
        )
    else:
        from comfy_kitchen.backends.eager import dequantize_mxfp8 as _dq
        a_fp = _dq(a, block_scale_a, torch.bfloat16)
        b_fp = _dq(b, block_scale_b, torch.bfloat16)
        out = torch.mm(a_fp, b_fp.t())
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.to(out_dtype)


def apply_rope(xq, xk, freqs_cis):
    from comfy_kitchen.backends.eager import apply_rope as _e
    return _e(xq, xk, freqs_cis)


def apply_rope1(x, freqs_cis):
    from comfy_kitchen.backends.eager import apply_rope1 as _e
    return _e(x, freqs_cis)


def _register():
    from comfy_kitchen.registry import registry
    from comfy_kitchen.constraints import (
        DivisibleBy, ExactDims, FunctionConstraints, ParamConstraint,
    )
    if getattr(torch.version, "hip", None) is None:
        registry.mark_unavailable("rocm", "not a ROCm PyTorch build")
        return
    if not torch.cuda.is_available():
        registry.mark_unavailable("rocm", "no ROCm device available")
        return

    rocm_dev = frozenset({"cuda"})
    fp8_in   = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, _FP8_FNUZ})
    floats   = frozenset({torch.float32, torch.float16, torch.bfloat16})
    uint8    = frozenset({torch.uint8})

    caps = {
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x":           ParamConstraint(dtypes=floats),
                "scale":       ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=fp8_in),
            },
            default_devices=rocm_dev),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x":           ParamConstraint(dtypes=fp8_in),
                "scale":       ParamConstraint(dtypes=floats),
                "output_type": ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev),
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x":                ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=rocm_dev),
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx":               ParamConstraint(dtypes=uint8, shape_rules=(ExactDims(2),)),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scales":     ParamConstraint(dtypes=fp8_in),
                "output_type":      ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev),
        "scaled_mm_nvfp4": FunctionConstraints(
            params={
                "a":              ParamConstraint(dtypes=uint8, shape_rules=(ExactDims(2),)),
                "b":              ParamConstraint(dtypes=uint8, shape_rules=(ExactDims(2),)),
                "tensor_scale_a": ParamConstraint(dtypes=frozenset({torch.float32})),
                "tensor_scale_b": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scale_a":  ParamConstraint(dtypes=fp8_in),
                "block_scale_b":  ParamConstraint(dtypes=fp8_in),
                "out_dtype":      ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev),
        "apply_rope": FunctionConstraints(
            params={
                "xq":       ParamConstraint(dtypes=floats),
                "xk":       ParamConstraint(dtypes=floats),
                "freqs_cis": ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev),
        "apply_rope1": FunctionConstraints(
            params={
                "x":        ParamConstraint(dtypes=floats),
                "freqs_cis": ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev),
    }

    if hasattr(torch, "float8_e8m0fnu"):
        e8m0 = frozenset({torch.float8_e8m0fnu})
        caps["quantize_mxfp8"] = FunctionConstraints(
            params={"x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),))},
            default_devices=rocm_dev)
        caps["dequantize_mxfp8"] = FunctionConstraints(
            params={
                "qx":          ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn}), shape_rules=(ExactDims(2),)),
                "block_scales": ParamConstraint(dtypes=e8m0),
                "output_type":  ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev)
        caps["scaled_mm_mxfp8"] = FunctionConstraints(
            params={
                "a":            ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn}), shape_rules=(ExactDims(2),)),
                "b":            ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn}), shape_rules=(ExactDims(2),)),
                "block_scale_a": ParamConstraint(dtypes=e8m0),
                "block_scale_b": ParamConstraint(dtypes=e8m0),
                "out_dtype":     ParamConstraint(dtypes=floats),
            },
            default_devices=rocm_dev)

    registry.register(
        name="rocm",
        module=__import__(__name__, fromlist=__all__),
        capabilities=caps,
    )


_register()
