"""CK integer attention over a growing, fixed-capacity decode prefix."""
import torch

if torch.version.hip:
    from .backends import hip as backend
    from .backends.hip import _dl as wrap_for_dlpack
else:
    from .backends import cuda as backend
    from .backends.cuda import _wrap_for_dlpack as wrap_for_dlpack


def is_available(device: torch.device | None = None) -> bool:
    """Return whether the native growing INT8 cache supports this device."""
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    return (
        backend._EXT_AVAILABLE
        and hasattr(backend._C, "_int8_decode_update")
        and hasattr(backend._C, "_int8_decode")
        and (backend.has_wmma() if torch.version.hip else torch.cuda.get_device_capability(device) >= (8, 0))
    )


class Int8DecodeCache:
    def __init__(self, batch, heads, capacity, device):
        self.format = "ck_int8_hnd_hip_v1" if torch.version.hip else "ck_int8_hnd_cuda_v1"
        self.chunk_size = 512
        chunks = (capacity + self.chunk_size - 1) // self.chunk_size
        capacity = chunks * self.chunk_size
        self.key = torch.zeros((batch, heads, capacity, 256), dtype=torch.int8, device=device)
        self.value = torch.empty((batch * heads * 256, capacity), dtype=torch.int8, device=device)
        self.key_scale = torch.empty((batch, heads, capacity // 64 * 4), dtype=torch.float32, device=device)
        self.value_scale = torch.empty((batch, chunks, heads, 256), dtype=torch.float32, device=device)

    def update(self, key, value, length, initialize=False):
        """Refresh the last two quantization chunks, or initialize the prefix.

        All batches share the length. After initialization the committed length may
        advance by at most one chunk per call. BF16 sources are [B,H,rows,256].
        Initialization reads the full prefix. Updates can instead read a chunk-aligned
        circular source: absolute row i lives at i % rows. The caller must retain both
        chunks being refreshed, including through speculative writes and rollback.
        """
        backend._C._int8_decode_update(
            *map(wrap_for_dlpack, (key, value, self.key, self.value, self.key_scale, self.value_scale, length)),
            initialize, torch.cuda.current_stream(key.device).cuda_stream)

    def attend(self, query, length):
        """BF16 query [B,H,S,256]; return [B,S,H*256] and natural-log LSE [B,H,S].

        Every query attends the same committed prefix. The caller merges the current
        causal/tree rows separately, preserving speculative rollback semantics.
        """
        batch, heads, seq, dim = query.shape
        kv_heads = self.key.shape[1]
        rows = heads // kv_heads * seq
        q = query.reshape(batch, kv_heads, rows, dim).contiguous()
        qi = torch.empty_like(q, dtype=torch.int8)
        qs = torch.empty((batch, kv_heads, 64), dtype=torch.float32, device=q.device)
        splits = self.value_scale.shape[1] * (8 if torch.version.hip else 1)
        partial = torch.empty((batch, splits, kv_heads, rows, dim), dtype=q.dtype, device=q.device)
        lse = torch.empty(partial.shape[:-1], dtype=torch.float32, device=q.device)
        out = torch.empty((batch, seq, heads * dim), dtype=q.dtype, device=q.device)
        out_lse = torch.empty((batch, heads, seq), dtype=torch.float32, device=q.device)
        backend._C._int8_decode(
            *map(wrap_for_dlpack, (q, qi, qs, self.key, self.value, self.key_scale, self.value_scale,
                                       length, partial, lse, out, out_lse)),
            seq, torch.cuda.current_stream(q.device).cuda_stream)
        return out, out_lse
