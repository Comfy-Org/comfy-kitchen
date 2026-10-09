"""Paged CK integer attention over a growing, fixed-capacity decode prefix."""
import torch

from .backends import cuda


class Int8DecodeCache:
    def __init__(self, batch, heads, capacity, device, page_size=512):
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
        cuda._C._int8_decode_update(
            *map(cuda._wrap_for_dlpack, (key, value, self.key, self.value, self.key_scale, self.value_scale, length)),
            initialize, torch.cuda.current_stream(key.device).cuda_stream)

    def attend(self, query, length):
        """BF16 query [B,H,S,256]; return [B,S,H*256] and natural-log LSE [B,H,S].

        Every query attends the same committed prefix. The caller merges the current
        causal/tree rows separately, preserving speculative rollback semantics.
        """
        batch, heads, seq, dim = query.shape
        kv_heads = self.key.shape[2]
        rows = heads // kv_heads * seq
        q = query.reshape(batch, kv_heads, rows, dim).contiguous()
        qi = torch.empty_like(q, dtype=torch.int8)
        qs = torch.empty((batch, kv_heads, 64), dtype=torch.float32, device=q.device)
        partial = torch.empty((batch, self.key.shape[1], kv_heads, rows, dim), dtype=q.dtype, device=q.device)
        lse = torch.empty(partial.shape[:-1], dtype=torch.float32, device=q.device)
        out = torch.empty((batch, seq, heads * dim), dtype=q.dtype, device=q.device)
        out_lse = torch.empty((batch, heads, seq), dtype=torch.float32, device=q.device)
        cuda._C._int8_decode(
            *map(cuda._wrap_for_dlpack, (q, qi, qs, self.key, self.value, self.key_scale, self.value_scale,
                                       length, partial, lse, out, out_lse)),
            seq, torch.cuda.current_stream(q.device).cuda_stream)
        return out, out_lse
