"""CPU regression for the chunk driver's Python-owned tensor lifetimes.

Only the Metal prologue/epilogue boundary is substituted. Tensor allocation,
decode destinations, GEMMs, chunk iteration, and LoRA activation copies run
through the production driver with real CPU tensors.
"""
import weakref

import pytest
import torch

from comfy_kitchen.backends.mps import linear


@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize("with_lora", [False, True])
def test_previous_chunk_result_is_released_before_next_decode(monkeypatch, with_lora, m):
    generator = torch.Generator().manual_seed(4)
    k, n, rank = 4, 12, 8
    x = torch.randn(m, k, generator=generator)
    weight = torch.randn(n, k, generator=generator)
    down = torch.randn(rank, k, generator=generator)
    up = torch.randn(n, rank, generator=generator)
    seen_results = []
    seen_decode_views = []
    decoded_starts = []

    def prologue(value, group, dtype):
        assert group is None
        return value.to(dtype).clone(), torch.zeros(value.shape[0], dtype=torch.int32)

    class MetalBoundary:
        def __getattr__(self, name):
            if name.startswith("lora_up_prep_"):
                def prepare(upt, up_exp, source, dexp, scale, outputs, ranks, ranks8, stride, **dispatch):
                    upt.zero_()
                    upt[:ranks, :outputs] = source.t() * scale
                    up_exp.zero_()

                return prepare

            def epilogue(out, c, row_exp, ch_exp, bias, residual, residual_scale, *args, **dispatch):
                seen_results.append(weakref.ref(c))
                if "lora" in name:
                    la, upt, _, _, row0, _, _, _, _, ranks, rows, _, _, offset, _ = args
                    out[:, row0:row0 + rows] = c[:, :rows] + la[:, offset:offset + ranks] @ upt[:, row0:row0 + rows]
                else:
                    _, row0, *_ = args
                    out[:, row0:row0 + c.shape[1]] = c

            return epilogue

    def decode(row0, rows, dtype, transposed, ch_exp, out=None):
        assert not transposed
        assert all(ref() is None for ref in seen_results), "a previous chunk's GEMM output remains live"
        assert all(ref() is None for ref in seen_decode_views), "the first LoRA decode view remains live"
        decoded_starts.append(row0)
        chunk = weight[row0:row0 + rows].to(dtype)
        if out is not None:
            seen_decode_views.append(weakref.ref(out))
            out.copy_(chunk)
            return out
        return chunk.clone()

    monkeypatch.setattr(linear, "prologue", prologue)
    monkeypatch.setattr(linear, "_shaders", MetalBoundary)
    monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 4 * k * x.element_size())
    monkeypatch.setattr(linear, "_TRANSPOSED_DECODE_MIN_K", 12288)
    monkeypatch.delenv("COMFY_KITCHEN_MPS_COMPUTE_DTYPE", raising=False)

    result = linear.weight_only_linear(
        x, n, k, decode, bias=None, out_dtype=x.dtype, group=None,
        lora=(down, up, 1.0) if with_lora else None,
    )

    expected = x @ weight.t()
    if with_lora:
        expected += (x @ down.t()) @ up.t()
    torch.testing.assert_close(result, expected)
    assert decoded_starts == [0, 4, 8]
    assert len(seen_results) == 3
    assert all(ref() is None for ref in seen_results)
    assert all(ref() is None for ref in seen_decode_views)
