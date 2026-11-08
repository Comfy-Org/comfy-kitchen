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
    "qkv_interleaved_qk_rms_norm",
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
    from .na import na3d
    from .qkv_interleaved_qk_rms_norm import qkv_interleaved_qk_rms_norm
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

    def _qkv_interleaved_qk_rms_norm_call_rule(kwargs):
        # No size threshold (the shapes measured are in the PR that added this op).
        names = ("q", "k", "v", "q_weight", "k_weight")
        tensors = [kwargs.get(n) for n in names]
        heads = kwargs.get("heads")
        if any(t is None for t in tensors) or heads is None:
            return ValidationResult.ok()
        q, k = tensors[0], tensors[1]
        if any(t.device != q.device for t in tensors):
            return ValidationResult.fail("q", "all tensors must be on one device")
        batch_size, seq, channels = q.shape
        seq_kv = k.shape[1]
        if batch_size == 0 or seq == 0 or seq_kv == 0:
            return ValidationResult.fail("q", "empty inputs are not supported by triton")
        # the sum of squares follows torch's rms kernel order for head_dim 128
        if channels != heads * 128:
            return ValidationResult.fail("q", "head_dim must be 128")
        # with one head the stock v (and the q / k norm inputs) are strided views, not copies
        if heads not in (2, 4, 8, 16, 32):
            return ValidationResult.fail("q", "heads must be a power of 2 from 2 to 32")
        if any(not t.is_contiguous() for t in tensors):
            return ValidationResult.fail("q", "q, k, v and the weights must be contiguous")
        if batch_size * max(seq, seq_kv) >= 2**31:
            return ValidationResult.fail("q", "q, k and v must have fewer than 2**31 tokens")
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
    # cuda covers ROCm builds of PyTorch (device type "cuda"); measured on both
    out["qkv_interleaved_qk_rms_norm"] = FunctionConstraints(
        params={
            name: ParamConstraint(dtypes=frozenset({torch.float16}), shape_rules=(ExactDims(3),))
            for name in ("q", "k", "v")
        }
        | {
            name: ParamConstraint(dtypes=frozenset({torch.float16}), shape_rules=(ExactDims(1),))
            for name in ("q_weight", "k_weight")
        },
        default_devices=cuda_devices,
        call_rules=(_qkv_interleaved_qk_rms_norm_call_rule,),
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
