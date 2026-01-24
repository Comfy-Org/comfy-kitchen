"""NVFP4 quantized Linear layer implementation."""
from __future__ import annotations

import torch
import torch.nn as nn

from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout


class NVFP4Linear(nn.Module):
    """Linear layer with NVFP4 quantized weights.

    A memory-efficient linear layer that stores weights directly in NVFP4 (E2M1)
    format. Unlike nn.Linear, this class does not maintain a full-precision
    weight tensor, significantly reducing memory usage.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()

            self.device = "cpu" if device is None else device
            self.compute_dtype = torch.bfloat16 if dtype is None else dtype

            self.in_features = in_features
            self.out_features = out_features

            self.weight = None # Init during state dict loading
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=self.compute_dtype, device=self.device))
            else:
                self.register_parameter("bias", None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                            strict, missing_keys, unexpected_keys, error_msgs):
        weight = state_dict.pop(prefix + "weight", None).to(device=self.device)
        weight_scale = state_dict.pop(prefix + "weight_scale_2", None).to(device=self.device)
        weight_block_scale = state_dict.pop(prefix + "weight_scale", None).to(device=self.device).view(dtype=torch.float8_e4m3fn)

        params = TensorCoreNVFP4Layout.Params(
            scale=weight_scale,
            orig_dtype=self.compute_dtype,
            orig_shape=(self.out_features, self.in_features),
            block_scale=weight_block_scale,
        )

        self.weight = nn.Parameter(QuantizedTensor(weight, "TensorCoreNVFP4Layout", params), requires_grad=False)
        manually_loaded_keys = [prefix + "weight", prefix + "weight_scale_2", prefix + "weight_scale"]

        input_scale = state_dict.pop(prefix + "input_scale", None)
        if input_scale is None:
            # This will result in dynamic quantization
            missing_keys.append(prefix + "input_scale")
            self.input_scale = None
        else:
            self.input_scale = nn.Parameter(input_scale.to(device=self.device), requires_grad=False)
            manually_loaded_keys.append(prefix + "input_scale")

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        for key in manually_loaded_keys:
            if key in missing_keys:
                missing_keys.remove(key)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        state_dict = super().state_dict(*args, **kwargs)
        del state_dict["weight"]  # Remove the QuantizedTensor
        state_dict.update(self.weight.state_dict("weight"))
        return state_dict


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, QuantizedTensor):
            x = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout", scale=self.input_scale)
        return torch.nn.functional.linear(x, self.weight, self.bias)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    in_features = 1024
    out_features = 2048

    scales = in_features * out_features // 16

    state_dict = {
        "weight": torch.randn(out_features, in_features // 2, device=device).to(dtype=torch.uint8), # NVFP stores 2 values per byte
        "bias": torch.randn(out_features, device=device, dtype=dtype),
        "weight_scale_2": torch.randn(1, device=device, dtype=torch.float32),
        "weight_scale": torch.ones(scales, device=device, dtype=torch.uint8), # float8_e4m3fn support is spotty in SD
        "input_scale": torch.randn(1, device=device, dtype=torch.float32),
    }

    model = NVFP4Linear(in_features, out_features, device=device, dtype=dtype)
    model.load_state_dict(state_dict)

    print(model.state_dict().keys())

    input = torch.randn(128, in_features, device=device, dtype=dtype)
    output = model(input)
