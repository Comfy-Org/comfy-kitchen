__all__ = [
    "adaln",
    "na3d",
    "rms_adaln",
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "int8_linear",
    "quantize_int8_rowwise",
    "quantize_and_rotate_rowwise",
    "w4a8_int8_linear",
    "flash_attn",
    "l2_norm_silu",
]

# Try to import triton and register if available
import torch

from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    ParamConstraint,
    ValidationResult,
    na3d_common_call_rule,
)
from comfy_kitchen.registry import registry

_TRITON_AVAILABLE = True
_TRITON_ERROR = None

try:
    import triton  # noqa: F401
    from comfy_kitchen.backends.eager.quantization import (
        quantize_and_rotate_rowwise as _eager_quantize_and_rotate_rowwise,
    )
    from comfy_kitchen.backends.eager.quantization import (
        quantize_int8_rowwise as _eager_quantize_int8_rowwise,
    )

    from .adaln import adaln, rms_adaln
    from .attention import flash_attn
    from .l2_norm_silu import l2_norm_silu
    from .na import na3d
    from .quantization import (
        dequantize_nvfp4,
        dequantize_per_tensor_fp8,
        int8_linear,
        quantize_mxfp8,
        quantize_nvfp4,
        quantize_per_tensor_fp8,
    )
    from .quantization import (
        triton_quantize_and_rotate_rowwise as _triton_quantize_and_rotate_rowwise,
    )
    from .quantization import (
        triton_quantize_rowwise as _triton_quantize_int8_rowwise,
    )
    from .rms_rope import (
        rms_rope,
        rms_rope1,
        rms_rope1_,
        rms_rope_,
        rms_rope_split_half,
        rms_rope_split_half1,
        rms_rope_split_half1_,
        rms_rope_split_half_,
    )
    from .rope import (
        apply_rope,
        apply_rope1,
        apply_rope1_,
        apply_rope_,
        apply_rope_split_half,
        apply_rope_split_half1,
        apply_rope_split_half1_,
        apply_rope_split_half_,
    )
    from .w4a8_int8 import w4a8_int8_linear
except ImportError as e:
    _TRITON_AVAILABLE = False
    _TRITON_ERROR = f"ImportError: {e!s}"


if _TRITON_AVAILABLE:
    def quantize_int8_rowwise(
        x: torch.Tensor,
        stochastic_rounding: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stochastic_rounding is not None and stochastic_rounding > 0:
            return _eager_quantize_int8_rowwise(x, stochastic_rounding=stochastic_rounding)
        return _triton_quantize_int8_rowwise(x)

    def quantize_and_rotate_rowwise(
        x: torch.Tensor,
        h: torch.Tensor,
        group_size: int,
        stochastic_rounding: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stochastic_rounding is not None and stochastic_rounding > 0:
            return _eager_quantize_and_rotate_rowwise(
                x, h, group_size, stochastic_rounding=stochastic_rounding
            )
        return _triton_quantize_and_rotate_rowwise(x, h, group_size)


def _build_constraints() -> dict:
    def _na3d_call_rule(kwargs):
        common = na3d_common_call_rule(kwargs)
        if not common.success:
            return common
        q = kwargs.get("q")
        if q is not None and q.shape[-1] > 128:
            return ValidationResult.fail("q", "head_dim > 128 not supported by triton na3d")
        return ValidationResult.ok()

    def _rms_rope_call_rule(kwargs):
        # only the norm_over_heads form has extra layout needs; the per-head form is unchanged
        if not kwargs.get("norm_over_heads"):
            return ValidationResult.ok()
        q, k, freqs = kwargs.get("q"), kwargs.get("k"), kwargs.get("freqs_cis")
        q_scale = kwargs.get("q_scale")
        k_scale = kwargs.get("k_scale") if kwargs.get("k_scale") is not None else q_scale
        if q is None or k is None or freqs is None or q_scale is None:
            return ValidationResult.ok()
        if q.dim() != 4 or k.shape != q.shape or k.dtype != q.dtype:
            return ValidationResult.fail("k", "norm_over_heads: q and k must be [B, S, H, head_dim] of one shape and dtype")
        if q.dtype not in (torch.float16, torch.bfloat16):
            return ValidationResult.fail("q", f"norm_over_heads: fp16/bf16 only, got {q.dtype}")
        if not q.is_contiguous() or not k.is_contiguous():
            return ValidationResult.fail("q", "norm_over_heads: q and k must be contiguous")
        _, seq_len, heads, head_dim = q.shape
        if head_dim % 2 or heads * head_dim > 16384:
            return ValidationResult.fail("q", f"norm_over_heads: H * head_dim {heads * head_dim} too wide or head_dim odd")
        if (freqs.dim() != 6 or freqs.dtype != torch.float32 or freqs.shape[0] != 1
                or freqs.shape[1] not in (1, seq_len) or freqs.shape[2] != 1
                or freqs.shape[3:] != (head_dim // 2, 2, 2)):
            return ValidationResult.fail("freqs_cis", "norm_over_heads: freqs must be float32 [1, S | 1, 1, head_dim/2, 2, 2]")
        if q_scale.numel() != heads * head_dim or k_scale.numel() != heads * head_dim:
            return ValidationResult.fail("q_scale", "norm_over_heads: scales need H * head_dim elements")
        return ValidationResult.ok()

    def _flash_attn_call_rule(kwargs):
        q, k, v = kwargs.get("q"), kwargs.get("k"), kwargs.get("v")
        if q is None or k is None or v is None:
            return ValidationResult.ok()
        if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            return ValidationResult.fail("k", "q [B, S_q, H, D] and k, v [B, S_k, H, D] must agree")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return ValidationResult.fail("k", "q, k, v must share one dtype")
        if q.shape[-1] not in (64, 128):
            return ValidationResult.fail("q", f"head_dim must be 64 or 128, got {q.shape[-1]}")
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            return ValidationResult.fail("q", "q, k, v need a unit stride on head_dim")
        return ValidationResult.ok()

    def _l2_norm_silu_call_rule(kwargs):
        x, weight = kwargs.get("x"), kwargs.get("weight")
        if x is None or weight is None:
            return ValidationResult.ok()
        if x.dim() < 3:
            return ValidationResult.fail("x", "x must be [B, C, *spatial]")
        if weight.numel() != x.shape[1]:
            return ValidationResult.fail("weight", f"weight needs {x.shape[1]} elements")
        return ValidationResult.ok()

    cuda_devices = frozenset({"cuda"})
    triton_devices = frozenset({"cuda", "xpu"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    out = {
        "adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
                "shift": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
                "shift": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=triton_devices,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=cuda_devices,
        ),
        "quantize_mxfp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
        ),
        # Uses inline PTX: cvt.rn.f16x2.e2m1x2 (SM100/Blackwell instruction)
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scales": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            min_compute_capability=(10, 0),  # SM100 required for cvt.rn.f16x2.e2m1x2
        ),
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=standard_floats),
                "k": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "q_scale": ParamConstraint(dtypes=standard_floats),
                "k_scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
            call_rules=(_rms_rope_call_rule,),
        ),
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                "weight_scale": ParamConstraint(dtypes=standard_floats),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
            },
            default_devices=triton_devices,
            min_compute_capability=(8, 0),  # Required for Triton INT8 dot
        ),
        "w4a8_int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "qdata": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "s_rel": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn, torch.float32}), shape_rules=(ExactDims(2),)),
                "s_channel": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
                "codebook": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
                "correction": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "bias": ParamConstraint(dtypes=standard_floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
            min_compute_capability=(8, 0),  # Required for Triton INT8 dot
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=triton_devices,
        ),
        "quantize_and_rotate_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "H": ParamConstraint(dtypes=standard_floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=triton_devices,
        ),
        "apply_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "apply_rope_split_half": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope_split_half": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=standard_floats),
                "k": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "q_scale": ParamConstraint(dtypes=standard_floats),
                "k_scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
    }
    for inplace_name, functional_name in {
        "apply_rope_": "apply_rope",
        "apply_rope1_": "apply_rope1",
        "apply_rope_split_half_": "apply_rope_split_half",
        "apply_rope_split_half1_": "apply_rope_split_half1",
        "rms_rope_": "rms_rope",
        "rms_rope1_": "rms_rope1",
        "rms_rope_split_half_": "rms_rope_split_half",
        "rms_rope_split_half1_": "rms_rope_split_half1",
    }.items():
        out[inplace_name] = out[functional_name]
    half_floats = frozenset({torch.float16, torch.bfloat16})
    out["flash_attn"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(4),)),
            "k": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(4),)),
            "v": ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(4),)),
        },
        default_devices=triton_devices,
        call_rules=(_flash_attn_call_rule,),
    )
    out["l2_norm_silu"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=half_floats),
            "weight": ParamConstraint(dtypes=standard_floats),
        },
        default_devices=triton_devices,
        call_rules=(_l2_norm_silu_call_rule,),
    )
    out["na3d"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(6),)),
            "k": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(6),)),
            "v": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(6),)),
        },
        default_devices=cuda_devices,
        # tl.dot needs Ampere matrix capability for the non-IEEE float paths
        min_compute_capability=(8, 0),
        call_rules=(_na3d_call_rule,),
    )
    return out


def _register():
    if not _TRITON_AVAILABLE:
        registry.mark_unavailable("triton", _TRITON_ERROR or "Triton not available")
        return

    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if not has_cuda and not has_xpu:
        registry.mark_unavailable("triton", "Neither CUDA nor XPU available on this system")
        return

    registry.register(
        name="triton",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
