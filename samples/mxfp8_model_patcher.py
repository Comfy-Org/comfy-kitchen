"""ComfyUI node that patches model linear layers to use MXFP8 quantization."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from comfy_kitchen.tensor import QuantizedTensor

logger = logging.getLogger(__name__)


class MXFP8Linear(nn.Module):
    """Linear layer with MXFP8 quantized weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: nn.Parameter | None = None,
        weight_qt: QuantizedTensor | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(weight_qt, requires_grad=False) if weight_qt is not None else None
        self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, device: str = "cuda") -> MXFP8Linear:
        weight = linear.weight.data
        compute_dtype = weight.dtype

        # Move to CUDA if needed (quantization requires CUDA)
        if weight.device.type != "cuda":
            weight = weight.to(device)

        weight_qt = QuantizedTensor.from_float(weight, "TensorCoreMXFP8Layout")

        bias = linear.bias
        if bias is not None and bias.device.type != "cuda":
            bias = nn.Parameter(bias.data.to(device))

        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            weight_qt=weight_qt,
            compute_dtype=compute_dtype,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        tensor_3d = x.ndim == 3

        if not isinstance(x, QuantizedTensor):
            if tensor_3d:
                x = x.reshape(-1, input_shape[2])

            if x.ndim != 2:
                return torch.nn.functional.linear(x.reshape(input_shape), self.weight.dequantize(), self.bias)

            x = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")

        output = torch.nn.functional.linear(x, self.weight, self.bias)

        if tensor_3d:
            output = output.reshape(input_shape[0], input_shape[1], self.out_features)

        return output


def patch_model_with_mxfp8(
    model: nn.Module,
    min_features: int = 64,
    skip_names: list[str] | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, int]:
    """Patch all suitable Linear layers in a model to use MXFP8 quantization.

    Args:
        model: The model to patch
        min_features: Minimum feature size to quantize (smaller layers are skipped)
        skip_names: List of layer name patterns to skip
        verbose: Print detailed progress

    Returns:
        Tuple of (patched model, number of layers patched)
    """
    skip_names = skip_names or []
    patched_count = 0
    skipped_reasons: dict[str, list[str]] = {
        "skip_pattern": [],
        "too_small": [],
        "failed": [],
    }

    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_names)

    def patch_recursive(module: nn.Module, prefix: str = "") -> None:
        nonlocal patched_count

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                if should_skip(full_name):
                    skipped_reasons["skip_pattern"].append(full_name)
                    continue

                if child.in_features < min_features or child.out_features < min_features:
                    skipped_reasons["too_small"].append(f"{full_name} ({child.in_features}x{child.out_features})")
                    continue

                try:
                    mxfp8_linear = MXFP8Linear.from_linear(child)
                    setattr(module, name, mxfp8_linear)
                    patched_count += 1
                    if verbose:
                        print(f"    ✓ {full_name} ({child.in_features}x{child.out_features})")
                except Exception as e:
                    skipped_reasons["failed"].append(f"{full_name}: {e}")
            else:
                patch_recursive(child, full_name)

    patch_recursive(model)

    if verbose:
        if skipped_reasons["skip_pattern"]:
            print(f"  Skipped (pattern match): {len(skipped_reasons['skip_pattern'])}")
        if skipped_reasons["too_small"]:
            print(f"  Skipped (too small): {len(skipped_reasons['too_small'])}")
        if skipped_reasons["failed"]:
            print(f"  Failed to patch: {len(skipped_reasons['failed'])}")
            for reason in skipped_reasons["failed"]:
                print(f"    ✗ {reason}")

    return model, patched_count


def debug_mxfp8_capability():
    """Print detailed MXFP8 capability information for debugging."""
    import torch

    from comfy_kitchen.scaled_mm_v2 import get_pytorch_version_info
    from comfy_kitchen.tensor import TensorCoreMXFP8Layout

    print("=" * 70)
    print("MXFP8 Debug Information")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    sm_major, sm_minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: SM {sm_major}.{sm_minor}")
    print("Native MXFP8 Required: SM >= 10.0 (Blackwell)")
    print(f"Native MXFP8 Supported: {sm_major >= 10}")
    print()

    version_info = get_pytorch_version_info()
    print(f"PyTorch Version: {version_info['torch_version']}")
    print(f"CUDA Version: {version_info['cuda_version']}")
    print(f"Has scaled_mm V2 API: {version_info['has_scaled_mm_v2']}")
    print()

    print(f"TensorCoreMXFP8Layout.supports_fast_matmul(): {TensorCoreMXFP8Layout.supports_fast_matmul()}")
    print()

    if sm_major < 10:
        print("⚠️  Your GPU does not support native MXFP8 matmul.")
        print("   Path: MXFP8 weights → dequantize → BF16 matmul")
        print("   Benefit: Memory savings from compressed weights")
        print("   No compute speedup (dequantization overhead)")
    else:
        print("✓ Native MXFP8 matmul supported!")
        print("   Path: MXFP8 weights → native block-scaled GEMM")
        print("   Benefits: Memory savings + compute speedup")

    print("=" * 70)


def enable_mxfp8_debug_logging():
    """Enable verbose logging for MXFP8 debugging."""
    import logging
    logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.DEBUG)
    logging.getLogger("comfy_kitchen.tensor.base").setLevel(logging.DEBUG)
    logging.getLogger("comfy_kitchen.tensor.mxfp8").setLevel(logging.DEBUG)
    print("MXFP8 debug logging enabled. Watch for:")
    print("  - 'Backend X selected for Y' - which backend handles each op")
    print("  - 'Unhandled op X for Y, dequantizing' - ops falling back")


class ApplyMXFP8Quantization:
    """ComfyUI node that applies MXFP8 quantization to model linear layers."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "min_features": ("INT", {"default": 64, "min": 1, "max": 4096}),
                "skip_patterns": ("STRING", {"default": "", "multiline": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "comfy_kitchen/quantization"

    def apply(
        self,
        model: Any,
        min_features: int = 64,
        skip_patterns: str = "",
        debug: bool = False,
    ) -> tuple[Any]:
        if debug:
            debug_mxfp8_capability()
            enable_mxfp8_debug_logging()

        skip_names = [s.strip() for s in skip_patterns.split("\n") if s.strip()]

        model_patcher = model.clone()
        diffusion_model = model_patcher.model.diffusion_model

        # Count layers before patching
        total_linear = sum(1 for m in diffusion_model.modules() if isinstance(m, nn.Linear))
        eligible_linear = sum(
            1 for m in diffusion_model.modules()
            if isinstance(m, nn.Linear) and m.in_features >= min_features and m.out_features >= min_features
        )

        if debug:
            print("\nMXFP8 Patching:")
            print(f"  Total nn.Linear layers: {total_linear}")
            print(f"  Eligible (>= {min_features} features): {eligible_linear}")
            if skip_names:
                print(f"  Skip patterns: {skip_names}")

        _, patched_count = patch_model_with_mxfp8(
            diffusion_model,
            min_features=min_features,
            skip_names=skip_names,
            verbose=debug,
        )

        if debug:
            print(f"  Successfully patched: {patched_count}")
            print()

        logger.info(f"MXFP8: Patched {patched_count} linear layers")
        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "ApplyMXFP8Quantization": ApplyMXFP8Quantization,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyMXFP8Quantization": "Apply MXFP8 Quantization",
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.small = nn.Linear(16, 16)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available, skipping test")
        exit(0)

    from comfy_kitchen.scaled_mm_v2 import get_pytorch_version_info
    from comfy_kitchen.tensor import TensorCoreMXFP8Layout

    print("=" * 80)
    print("MXFP8 Capability Check")
    print("=" * 80)

    sm_major, sm_minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: SM {sm_major}.{sm_minor}")
    print("MXFP8 Required: SM >= 10.0 (Blackwell)")
    print(f"Native MXFP8 Matmul Supported: {TensorCoreMXFP8Layout.supports_fast_matmul()}")
    print()

    version_info = get_pytorch_version_info()
    print(f"PyTorch Version: {version_info['torch_version']}")
    print(f"CUDA Version: {version_info['cuda_version']}")
    print(f"Has scaled_mm V2 API: {version_info['has_scaled_mm_v2']}")
    print()

    if sm_major < 10:
        print("⚠️  WARNING: Your GPU does not support native MXFP8 matmul.")
        print("   Fallback path: MXFP8 weights → dequantize → BF16 matmul")
        print("   Benefits: Memory reduction from MXFP8 weight storage")
        print("   Drawback: No compute speedup (dequantize overhead)")
    else:
        print("✓ Native MXFP8 matmul is supported on this GPU!")

    print("=" * 80)
    print()

    logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.DEBUG)

    model = SimpleModel().to(device).to(torch.bfloat16)
    print(f"Before patching: {model}")

    print("\nMXFP8 Patching:")
    model, count = patch_model_with_mxfp8(model, min_features=64, verbose=True)
    print(f"  Total patched: {count}")
    print(f"\nAfter patching: {model}")

    print("\n" + "=" * 80)
    print("Running forward pass (with dispatch logging):")
    print("=" * 80)

    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16)

    output = model(x)
    torch.cuda.synchronize()

    logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.WARNING)

    for _ in range(3):
        output = model(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(10):
            output = model(x)
        torch.cuda.synchronize()

    print(f"\nOutput shape: {output.shape}")
    print("\n" + "=" * 80)
    print("CUDA Kernels:")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

