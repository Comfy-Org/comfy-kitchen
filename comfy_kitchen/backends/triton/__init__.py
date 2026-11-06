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
    "linear_geglu",
    "linear_swiglu",
    "quantize_int8_rowwise",
    "quantize_and_rotate_rowwise",
    "w4a8_int8_linear",
]

# Try to import triton and register if available
import torch

from comfy_kitchen.constraints import (
    DivisibleBy,
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
    from .linear_act import linear_geglu, linear_swiglu
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

    from .linear_act import GLU_N_MULTIPLE, K_MULTIPLE, MIN_M

    def _linear_act_operands(kwargs, pairs):
        # Shared by the linear_* epilogue ops. The kernel reads weights / biases as
        # bare pointers sized from x and the weights, so anything it does not serve
        # must fall through to eager here. pairs: ((weight name, bias name), ...).
        x = kwargs.get("x")
        if x is None or any(kwargs.get(w) is None for w, _ in pairs):
            return ValidationResult.ok()
        for w_name, b_name in pairs:
            weight = kwargs[w_name]
            if x.dim() < 1 or x.shape[-1] == 0 or x.shape[-1] != weight.shape[1]:
                return ValidationResult.fail(
                    "x", f"last dim must equal {w_name}.shape[1] ({weight.shape[1]}) and be > 0"
                )
            if weight.dtype != x.dtype or weight.device != x.device:
                return ValidationResult.fail(w_name, "must match x's dtype and device")
            if weight.shape[0] == 0 or weight.stride(1) != 1:
                return ValidationResult.fail(w_name, "must be non-empty with contiguous rows")
            bias = kwargs.get(b_name)
            if bias is not None:
                if bias.dtype != x.dtype or bias.device != x.device:
                    return ValidationResult.fail(b_name, "must match x's dtype and device")
                if bias.shape[0] != weight.shape[0] or bias.stride(0) != 1:
                    return ValidationResult.fail(
                        b_name, f"must be contiguous with {w_name}.shape[0] elements"
                    )
        if x.numel() // x.shape[-1] < MIN_M:
            return ValidationResult.fail("x", f"fewer than {MIN_M} rows; left to the unfused path")
        return ValidationResult.ok()

    def _linear_geglu_call_rule(kwargs):
        if kwargs.get("approximate", "none") not in ("none", "tanh"):
            return ValidationResult.fail("approximate", "must be 'none' or 'tanh'")
        return _linear_act_operands(kwargs, (("weight", "bias"),))

    def _linear_swiglu_call_rule(kwargs):
        w1, w3 = kwargs.get("w1"), kwargs.get("w3")
        if w1 is not None and w3 is not None and w1.shape != w3.shape:
            return ValidationResult.fail("w3", "must have w1's shape")
        return _linear_act_operands(kwargs, (("w1", "b1"), ("w3", "b3")))

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
    half_floats = frozenset({torch.float16, torch.bfloat16})
    glu_weight = ParamConstraint(
        dtypes=half_floats,
        shape_rules=(ExactDims(2), DivisibleBy(0, GLU_N_MULTIPLE), DivisibleBy(1, K_MULTIPLE)),
    )
    glu_bias = ParamConstraint(dtypes=half_floats, shape_rules=(ExactDims(1),))
    out["linear_geglu"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=half_floats),
            # packed [h | gate]: each half must tile evenly
            "weight": ParamConstraint(
                dtypes=half_floats,
                shape_rules=(
                    ExactDims(2),
                    DivisibleBy(0, 2 * GLU_N_MULTIPLE),
                    DivisibleBy(1, K_MULTIPLE),
                ),
            ),
            "bias": glu_bias,
            "approximate": ParamConstraint(dtypes=frozenset({str})),
        },
        default_devices=cuda_devices,
        # fp16 / bf16 tl.dot
        min_compute_capability=(8, 0),
        call_rules=(_linear_geglu_call_rule,),
    )
    out["linear_swiglu"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=half_floats),
            "w1": glu_weight,
            "w3": glu_weight,
            "b1": glu_bias,
            "b3": glu_bias,
        },
        default_devices=cuda_devices,
        min_compute_capability=(8, 0),
        call_rules=(_linear_swiglu_call_rule,),
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
