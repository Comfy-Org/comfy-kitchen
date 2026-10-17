"""Paged CK integer attention over a growing, fixed-capacity decode prefix."""
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
        # The attention half of this cache is a WMMA kernel (int8_attn_kernel with
        # Paged=true), so it needs matrix cores: `has_wmma()` on ROCm, SM80 on CUDA.
        # That is the same requirement `is_available` above reports, and this class
        # has to enforce it rather than assume the caller did.
        #
        # Without the check, a cache built on a device without matrix cores -- RDNA2
        # (gfx103x), where MmaInt8::mma compiles to __builtin_trap() and leaves the
        # context unusable -- kills the process instead of raising. A trap is not a
        # launch error either: hipGetLastError() reports success, so nothing catches
        # it, and the failure surfaces later as "unspecified launch failure" at some
        # unrelated synchronization. Checked here rather than in attend() because the
        # cache is unusable from construction on such a device, and allocating the
        # (large) buffers first would only delay the error.
        #
        # RDNA2 is not left without a decode path by this: it runs the ported
        # SageAttention kernel for attention (rdna2_sageattn, head_dim <= 128) and
        # the upstream flash-decode for KV, just not this paged int8 kernel.
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if not is_available(dev):
            raise RuntimeError(
                "Int8DecodeCache requires the paged INT8 decode kernels, which are "
                "WMMA-only: the HIP extension on an AMD device with matrix cores "
                "(RDNA3 or newer), or the CUDA extension on SM80 or newer"
                if torch.version.hip
                else "Int8DecodeCache requires the CUDA extension on SM80 or newer"
            )
        page_size = 512
        pages = (capacity + page_size - 1) // page_size
        self.key = torch.zeros((batch, pages, heads, page_size, 256), dtype=torch.int8, device=device)
        self.value = torch.empty((batch, pages, heads, 256, page_size), dtype=torch.int8, device=device)
        self.key_scale = torch.empty((batch, pages, heads, page_size // 64 * 4), dtype=torch.float32, device=device)
        self.value_scale = torch.empty((batch, pages, heads, 256), dtype=torch.float32, device=device)

    def update(self, key, value, length, initialize=False):
        """Refresh the last two pages, or initialize the prefix. length is one CUDA int64.

        All batches share the length. After initialization the committed length may
        advance by at most one page per call. BF16 sources are [B,H,rows,256].
        Initialization reads the full prefix. Updates can instead read a page-aligned
        circular source: absolute row i lives at i % rows. The caller must retain both
        pages being refreshed, including through speculative writes and rollback.
        """
        backend._C._int8_decode_update(
            *map(wrap_for_dlpack, (key, value, self.key, self.value, self.key_scale, self.value_scale, length)),
            initialize, torch.cuda.current_stream(key.device).cuda_stream)

    def attend(self, query, length):
        """BF16 query [B,H,S,256]; return [B,S,H*256] and natural-log LSE [B,H,S].

        Every query attends the same committed prefix. The caller merges the current
        causal rows separately, preserving speculative rollback semantics.
        """
        batch, heads, seq, dim = query.shape
        kv_heads = self.key.shape[2]
        rows = heads // kv_heads * seq
        q = query.reshape(batch, kv_heads, rows, dim).contiguous()
        qi = torch.empty_like(q, dtype=torch.int8)
        qs = torch.empty((batch, kv_heads, 64), dtype=torch.float32, device=q.device)
        splits = self.key.shape[1] * (8 if torch.version.hip else 1)
        partial = torch.empty((batch, splits, kv_heads, rows, dim), dtype=q.dtype, device=q.device)
        lse = torch.empty(partial.shape[:-1], dtype=torch.float32, device=q.device)
        out = torch.empty((batch, seq, heads * dim), dtype=q.dtype, device=q.device)
        out_lse = torch.empty((batch, heads, seq), dtype=torch.float32, device=q.device)
        backend._C._int8_decode(
            *map(wrap_for_dlpack, (q, qi, qs, self.key, self.value, self.key_scale, self.value_scale,
                                       length, partial, lse, out, out_lse)),
            seq, torch.cuda.current_stream(q.device).cuda_stream)
        return out, out_lse
