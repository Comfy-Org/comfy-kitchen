"""INT8 tensor-wise quantized Linear layer implementation (from dxqb/OneTrainer).

This sample demonstrates using TensorWiseINT8Layout for simpler INT8 quantization:
- Weights: Single scale per tensor
- Activations: Per-row scales (dynamic quantization)

Uses torch._int_mm / cuBLASLt IMMA tensor cores for fast matmul.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_kitchen.tensor import TensorWiseINT8Layout, QuantizedTensor
from comfy_kitchen.backends.eager.quantization import (
    quantize_int8_tensorwise,
    quantize_int8_rowwise,
    int8_linear,
)

logger = logging.getLogger(__name__)


class INT8LinearTensorwise(nn.Module):
    """Linear layer with tensor-wise INT8 quantized weights.

    Uses single scale per weight tensor + per-row activation quantization.
    Faster than block-wise due to simpler scaling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: nn.Parameter | None = None,
        weight_int8: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        if weight_int8 is not None:
            self.weight = nn.Parameter(weight_int8, requires_grad=False)
            self.register_buffer("weight_scale", weight_scale)
        else:
            self.weight = None
            self.register_buffer("weight_scale", None)
        self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, device: str = "cuda") -> INT8LinearTensorwise:
        """Create from a standard nn.Linear."""
        weight = linear.weight.data
        compute_dtype = weight.dtype

        if weight.device.type != device.split(":")[0]:
            weight = weight.to(device)

        # Tensor-wise quantization: single scale per tensor
        weight_int8, weight_scale = quantize_int8_tensorwise(weight)

        bias = linear.bias
        if bias is not None and bias.device.type != device.split(":")[0]:
            bias = nn.Parameter(bias.data.to(device))

        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            weight_int8=weight_int8,
            weight_scale=weight_scale,
            compute_dtype=compute_dtype,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dynamic per-row activation quantization."""
        orig_shape = x.shape

        # Flatten to 2D and use int8_linear
        x_2d = x.reshape(-1, x.shape[-1])

        if x_2d.shape[0] > 16:
            # Use fast INT8 matmul path
            y = int8_linear(x_2d, self.weight, self.weight_scale, self.bias, self.compute_dtype)
        else:
            # Small batch - dequantize for accuracy
            w_float = self.weight.float() * self.weight_scale
            y = F.linear(x_2d.to(self.compute_dtype), w_float.to(self.compute_dtype), self.bias)

        return y.reshape(*orig_shape[:-1], self.out_features)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Load tensor-wise INT8 weights from state dict."""
        weight = state_dict.pop(prefix + "weight", None)
        weight_scale = state_dict.pop(prefix + "weight_scale", None)

        if weight is not None and weight_scale is not None:
            if weight.dtype == torch.int8:
                self.weight = nn.Parameter(weight, requires_grad=False)
                self.weight_scale = weight_scale
            else:
                # Convert FP weights to INT8 on the fly
                w_int8, w_scale = quantize_int8_tensorwise(weight)
                self.weight = nn.Parameter(w_int8, requires_grad=False)
                self.weight_scale = w_scale
        else:
            if weight is not None:
                missing_keys.append(prefix + "weight_scale")
            if weight_scale is not None:
                missing_keys.append(prefix + "weight")

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                       strict, missing_keys, unexpected_keys, error_msgs)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return state dict with INT8 weight + scale."""
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict


def patch_model_with_int8_tensorwise(
    model: nn.Module,
    min_features: int = 64,  # No block alignment requirement!
    skip_names: list[str] | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, int]:
    """Patch all suitable Linear layers to use tensor-wise INT8 quantization.

    Simpler than block-wise: no alignment requirements.
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
                    skipped_reasons["too_small"].append(
                        f"{full_name} ({child.in_features}x{child.out_features})"
                    )
                    continue

                try:
                    int8_linear = INT8LinearTensorwise.from_linear(child)
                    setattr(module, name, int8_linear)
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


class ApplyINT8TensorwiseQuantization:
    """ComfyUI node that applies tensor-wise INT8 quantization."""

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
        skip_names = [s.strip() for s in skip_patterns.split("\n") if s.strip()]

        model_patcher = model.clone()
        diffusion_model = model_patcher.model.diffusion_model

        total_linear = sum(1 for m in diffusion_model.modules() if isinstance(m, nn.Linear))

        if debug:
            print("\nINT8 Tensor-wise Patching:")
            print(f"  Total nn.Linear layers: {total_linear}")
            print(f"  Min features: {min_features}")
            if skip_names:
                print(f"  Skip patterns: {skip_names}")

        _, patched_count = patch_model_with_int8_tensorwise(
            diffusion_model,
            min_features=min_features,
            skip_names=skip_names,
            verbose=debug,
        )

        if debug:
            print(f"  Successfully patched: {patched_count}")
            print()

        logger.info(f"INT8 Tensorwise: Patched {patched_count} linear layers")
        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "ApplyINT8TensorwiseQuantization": ApplyINT8TensorwiseQuantization,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyINT8TensorwiseQuantization": "Apply INT8 Quantization (Tensorwise)",
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

    print("=" * 80)
    print("INT8 Tensor-wise Quantization Test")
    print("=" * 80)

    model = SimpleModel().to(device).to(torch.bfloat16)
    print(f"Before patching: {model}")

    print("\nINT8 Tensor-wise Patching:")
    model, count = patch_model_with_int8_tensorwise(model, min_features=64, verbose=True)
    print(f"  Total patched: {count}")
    print(f"\nAfter patching: {model}")

    print("\n" + "=" * 80)
    print("Running forward pass:")
    print("=" * 80)

    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16)
    output = model(x)
    torch.cuda.synchronize()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
