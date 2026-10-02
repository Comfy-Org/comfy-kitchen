# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded W4A4 preparation for one caller-owned inference operation."""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass

import torch

from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major


@dataclass
class _Entry:
    source: weakref.ReferenceType
    signature: tuple
    stream: int | None
    snapshot: torch.Tensor
    unpacked: torch.Tensor
    size: int


class W4A4WeightCache:
    """Cache decoded weights within an explicit ``with`` scope.

    ``max_bytes`` counts packed snapshots AND decoded INT8 tensors. Entries
    never retain source weights or evict live entries to admit new ones; a
    smaller budget therefore accelerates a subset without sequential-scan LRU
    thrashing. Scope exit releases everything, including on exceptions.

    Hits compare packed contents, not just pointers or version counters:
    inference tensors have no version counter and cast buffers can be reused.
    On NPU this comparison synchronizes; benchmark it against unpacking costs.
    Callers must not mutate weights concurrently with execution, and must
    reserve this budget in addition to normal model/activation memory.
    """

    def __init__(self, max_bytes: int):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self._entries: dict[int, _Entry] = {}
        self._owner: int | None = None

    def __enter__(self):
        if self._owner is not None:
            raise RuntimeError("W4A4WeightCache scopes cannot be nested")
        self.clear()
        self.hits = self.misses = 0
        self._owner = threading.get_ident()
        return self

    def __exit__(self, *exc):
        self.clear()
        self._owner = None

    def clear(self):
        """Release prepared tensors; preserve counters for caller inspection."""
        self._entries.clear()
        self.bytes_used = 0

    def _drop(self, key, reference=None):
        entry = self._entries.get(key)
        if entry is not None and (reference is None or entry.source is reference):
            self.bytes_used -= entry.size
            del self._entries[key]

    def _unpack(self, weight: torch.Tensor) -> torch.Tensor:
        if self._owner is None:
            raise RuntimeError("Use W4A4WeightCache inside a with scope")
        # Cross-thread/stream calls retain the original stateless behavior.
        if (
            threading.get_ident() != self._owner
            or self.max_bytes == 0
            or weight.device.type not in {"cpu", "npu"}
        ):
            return _unpack_int4_row_major(weight).contiguous()
        if weight.dtype not in (torch.int8, torch.uint8) or weight.ndim != 2:
            raise ValueError(
                "Expected a two-dimensional packed INT4 tensor stored as int8 or uint8"
            )
        stream = None
        if weight.device.type == "npu":
            if torch.npu.is_current_stream_capturing():
                return _unpack_int4_row_major(weight).contiguous()
            stream = torch.npu.current_stream(weight.device).npu_stream
        key = id(weight)
        signature = (
            weight.device,
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.storage_offset(),
            weight.data_ptr(),
        )
        entry = self._entries.get(key)
        if entry is not None:
            if entry.stream != stream:
                return _unpack_int4_row_major(weight).contiguous()
            if (
                entry.source() is weight
                and entry.signature == signature
                and torch.equal(weight, entry.snapshot)
            ):
                self.hits += 1
                return entry.unpacked
            self._drop(key)
        self.misses += 1
        unpacked = _unpack_int4_row_major(weight).contiguous()
        size = weight.numel() * weight.element_size() + unpacked.numel() * unpacked.element_size()
        if size > self.max_bytes - self.bytes_used or size == 0:
            return unpacked
        snapshot = weight.clone(memory_format=torch.contiguous_format)
        owner = weakref.ref(self)

        def expired(reference):
            cache = owner()
            if cache is not None:
                cache._drop(key, reference)

        reference = weakref.ref(weight, expired)
        self._entries[key] = _Entry(reference, signature, stream, snapshot, unpacked, size)
        self.bytes_used += size
        return unpacked
