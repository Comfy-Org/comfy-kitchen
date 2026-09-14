# SPDX-License-Identifier: Apache-2.0
# Adapted from anemoi-project/anemoi revision 4e85afba741bdeaf2d9486cab19cb76d3e7985a4.
"""Shared Anemoi option resolution and ragged video layout construction.

Three concerns live here in call order: validating the algorithm options every
backend must agree on, partitioning a video frame grid into connected query
blocks, and materializing that partition as the index/count tensors consumed
by the reference and native executors. All work is host-side; no GPU tensors
are cached across calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch


@dataclass(frozen=True, slots=True)
class AnemoiOptions:
    query_block_size: int
    ratios: tuple[float, float, float, float]
    prefix_kv_precision: str
    prefix_query_precision: str
    draftmap_proxy: str
    diag_jensen: bool
    maxpool_weight: float
    enable_anchors: bool
    smooth_k: bool
    nvfp4_scales: tuple[float, float, float]


def resolve_options(
    *,
    architecture,
    query_block_size=64,
    nvfp4_ratio=0.0,
    int8_ratio=1.0,
    mxfp8_ratio=0.0,
    fp16_ratio=0.0,
    prefix_kv_precision="auto",
    prefix_query_precision="auto",
    draftmap_proxy="mean",
    diag_jensen=False,
    maxpool_weight=0.0,
    enable_anchors=False,
    smooth_k=False,
    nvfp4_scales=(1.0, 1.0, 1.0),
    prefix_tokens=0,
):
    if architecture not in ("sm89", "sm120", "reference"):
        raise ValueError("architecture must be 'sm89', 'sm120' or 'reference'")
    if type(query_block_size) is not int or query_block_size not in (64, 128):
        raise ValueError("query_block_size must be 64 or 128")
    ratios = (nvfp4_ratio, int8_ratio, mxfp8_ratio, fp16_ratio)
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
        for x in ratios
    ):
        raise ValueError("precision ratios must be finite nonnegative numbers")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("precision ratios must sum to one")
    if int8_ratio and mxfp8_ratio:
        raise ValueError("INT8 and MXFP8 cannot share the middle phase")
    phases = tuple(
        name
        for name, value in zip(("nvfp4", "int8", "mxfp8", "fp16"), ratios, strict=True)
        if value
    )
    if architecture == "sm89" and (nvfp4_ratio or mxfp8_ratio):
        raise ValueError("SM89 supports INT8/E4M3 and FP16 phases only")
    for name, value in (
        ("diag_jensen", diag_jensen),
        ("enable_anchors", enable_anchors),
        ("smooth_k", smooth_k),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be bool")
    if smooth_k and architecture == "sm120":
        raise ValueError("smooth_k is available on SM89 only")
    if draftmap_proxy not in ("mean", "k_tail_r1", "k_tail_r2"):
        raise ValueError("draftmap_proxy must be mean, k_tail_r1 or k_tail_r2")
    if draftmap_proxy != "mean" and (
        architecture == "sm89" or query_block_size != 64 or diag_jensen
    ):
        raise ValueError("K-tail requires SM120 Q64 without Jensen correction")
    if (
        isinstance(maxpool_weight, bool)
        or not isinstance(maxpool_weight, (int, float))
        or not math.isfinite(maxpool_weight)
        or not 0 <= maxpool_weight <= 1
    ):
        raise ValueError("maxpool_weight must be finite and in [0, 1]")
    if maxpool_weight and (diag_jensen or draftmap_proxy != "mean"):
        raise ValueError("max-pool cannot be combined with Jensen or K-tail")
    valid_prefix = (
        ("auto", "int8", "fp16")
        if architecture == "sm89"
        else ("auto", "nvfp4", "int8", "mxfp8", "fp16")
    )
    if prefix_kv_precision not in valid_prefix:
        raise ValueError("unsupported prefix_kv_precision")
    # Donor executes non-INT8 prefix queries through dense SDPA, even when the
    # selected label is NVFP4/MXFP8. Preserve execution, not a false low-bit claim.
    if prefix_query_precision not in valid_prefix:
        raise ValueError("unsupported prefix_query_precision")
    if prefix_kv_precision == "auto":
        prefix_kv_precision = "fp16" if architecture == "sm89" or "fp16" in phases else phases[0]
    if prefix_query_precision == "auto":
        prefix_query_precision = "int8" if architecture != "sm89" and "int8" in phases else "fp16"
    if prefix_tokens and (
        (prefix_kv_precision == "int8" and mxfp8_ratio)
        or (prefix_kv_precision == "mxfp8" and int8_ratio)
    ):
        raise ValueError("prefix and video cannot require both INT8 and MXFP8 middle phases")
    if len(nvfp4_scales) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
        for x in nvfp4_scales
    ):
        raise ValueError("nvfp4_scales must contain three finite positive numbers")
    return AnemoiOptions(
        query_block_size,
        tuple(float(x) for x in ratios),
        prefix_kv_precision,
        prefix_query_precision,
        draftmap_proxy,
        diag_jensen,
        float(maxpool_weight),
        enable_anchors,
        smooth_k,
        tuple(float(x) for x in nvfp4_scales),
    )


@dataclass(frozen=True, slots=True)
class Ragged2DPartition:
    """Immutable metadata for one exact-cover rectangular partition."""

    height: int
    width: int
    capacity: int
    blocks: tuple[tuple[int, ...], ...]
    token_to_block: tuple[int, ...]
    adjacency: tuple[tuple[bool, ...], ...] | None

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def counts(self) -> tuple[int, ...]:
        """Return the real-token denominator of every physical block."""

        return tuple(len(block) for block in self.blocks)


_Cost = tuple[float, float, float, float, int]


def _balanced_segment_sizes(tokens: int, blocks: int) -> tuple[int, ...]:
    base, larger = divmod(tokens, blocks)
    return (base + 1,) * larger + (base,) * (blocks - larger)


def _band_serpentine_blocks(
    height: int,
    width: int,
    block_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Split a complete row band into balanced connected intervals."""

    path = tuple(
        row * width + column
        for column in range(width)
        for row in (range(height) if column % 2 == 0 else range(height - 1, -1, -1))
    )
    blocks: list[tuple[int, ...]] = []
    offset = 0
    for size in _balanced_segment_sizes(height * width, block_count):
        blocks.append(path[offset : offset + size])
        offset += size
    return tuple(blocks)


def _block_shape_terms(
    block: tuple[int, ...],
    width: int,
) -> tuple[int, int, float, float]:
    cells = {divmod(token, width) for token in block}
    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]
    box_height = max(rows) - min(rows) + 1
    box_width = max(columns) - min(columns) + 1
    perimeter = sum(
        neighbor not in cells
        for row, column in cells
        for neighbor in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
    )
    centroid_row = sum(rows) / len(rows)
    centroid_column = sum(columns) / len(columns)
    moment = sum(
        (row - centroid_row) ** 2 + (column - centroid_column) ** 2 for row, column in cells
    )
    aspect_error = abs(math.log(box_width / box_height))
    return (
        perimeter,
        box_height * box_width - len(block),
        moment,
        aspect_error,
    )


def _partition_cost(
    blocks: tuple[tuple[int, ...], ...],
    width: int,
) -> _Cost:
    terms = [_block_shape_terms(block, width) for block in blocks]
    return (
        float(sum(term[0] for term in terms)),
        float(sum(term[1] for term in terms)),
        sum(term[2] for term in terms),
        sum(term[3] for term in terms),
        1,
    )


def _add_cost(lhs: _Cost, rhs: _Cost) -> _Cost:
    return tuple(
        round(left + right, 12) if index < 4 else left + right
        for index, (left, right) in enumerate(zip(lhs, rhs, strict=True))
    )  # type: ignore[return-value]


def _orientation_candidate(
    height: int,
    width: int,
    capacity: int,
    target_blocks: int,
) -> tuple[tuple[tuple[int, ...], ...], _Cost]:
    """Return the best horizontal-band partition with an exact block count."""

    small_block_size, large_blocks = divmod(height * width, target_blocks)
    if small_block_size + bool(large_blocks) > capacity:
        raise RuntimeError("balanced block size exceeds physical capacity")
    zero: _Cost = (0.0, 0.0, 0.0, 0.0, 0)
    states: dict[
        tuple[int, int],
        tuple[_Cost, int, tuple[tuple[int, int], ...]],
    ] = {(0, 0): (zero, 0, ())}
    band_cache: dict[
        tuple[int, int],
        tuple[tuple[tuple[int, ...], ...], _Cost],
    ] = {}

    for row_offset in range(height):
        for blocks_used in range(target_blocks + 1):
            state = states.get((row_offset, blocks_used))
            if state is None:
                continue
            old_cost, old_center_cost, old_bands = state
            for band_height in range(1, height - row_offset + 1):
                band_tokens = band_height * width
                minimum_band_blocks = math.ceil(band_tokens / (small_block_size + 1))
                maximum_band_blocks = min(
                    band_tokens // small_block_size,
                    target_blocks - blocks_used,
                )
                remaining_rows = height - row_offset - band_height
                for band_blocks in range(minimum_band_blocks, maximum_band_blocks + 1):
                    band_large = band_tokens - band_blocks * small_block_size
                    if not 0 <= band_large <= band_blocks:
                        continue
                    remaining_blocks = target_blocks - blocks_used - band_blocks
                    remaining_tokens = remaining_rows * width
                    remaining_large = remaining_tokens - remaining_blocks * small_block_size
                    if not 0 <= remaining_large <= remaining_blocks:
                        continue
                    cache_key = (band_height, band_blocks)
                    cached = band_cache.get(cache_key)
                    if cached is None:
                        local_blocks = _band_serpentine_blocks(band_height, width, band_blocks)
                        cached = (
                            local_blocks,
                            _partition_cost(local_blocks, width),
                        )
                        band_cache[cache_key] = cached
                    next_row = row_offset + band_height
                    candidate = (
                        _add_cost(old_cost, cached[1]),
                        old_center_cost + (abs(2 * next_row - height) if next_row < height else 0),
                        (*old_bands, (band_height, band_blocks)),
                    )
                    key = (next_row, blocks_used + band_blocks)
                    incumbent = states.get(key)
                    if incumbent is None or candidate < incumbent:
                        states[key] = candidate

    final = states.get((height, target_blocks))
    if final is None:
        raise RuntimeError("stripe DP failed to find the capacity-minimum cover")

    score, _, bands = final
    blocks: list[tuple[int, ...]] = []
    row_offset = 0
    for band_height, band_blocks in bands:
        for block in band_cache[(band_height, band_blocks)][0]:
            blocks.append(
                tuple((token // width + row_offset) * width + token % width for token in block)
            )
        row_offset += band_height
    return tuple(blocks), score


def _transpose_to_raster(
    blocks: tuple[tuple[int, ...], ...],
    height: int,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            transposed_column * width + transposed_row
            for token in block
            for transposed_row, transposed_column in (divmod(token, height),)
        )
        for block in blocks
    )


def _balanced_binary_patterns(length: int, ones: int) -> tuple[tuple[int, ...], ...]:
    """Return a small deterministic set of balanced zero/one arrangements."""

    if not 0 <= ones <= length:
        raise ValueError("ones must be within the binary-pattern length")
    if ones == 0:
        return ((0,) * length,)
    if ones == length:
        return ((1,) * length,)

    even = tuple((index + 1) * ones // length - index * ones // length for index in range(length))
    patterns = {
        (1,) * ones + (0,) * (length - ones),
        (0,) * (length - ones) + (1,) * ones,
        even,
        tuple(reversed(even)),
    }
    # Keeping this set constant-sized bounds host construction independently
    # of the number of CUDA blocks while covering clustered and even seams.
    return tuple(sorted(patterns))


def _block_is_connected(block: tuple[int, ...], width: int) -> bool:
    """Return whether raster tokens form one four-connected component."""

    remaining = set(block)
    frontier = [remaining.pop()]
    while frontier:
        token = frontier.pop()
        row, column = divmod(token, width)
        neighbors = []
        if row:
            neighbors.append(token - width)
        if column:
            neighbors.append(token - 1)
        if column + 1 < width:
            neighbors.append(token + 1)
        neighbors.append(token + width)
        for neighbor in neighbors:
            if neighbor in remaining:
                remaining.remove(neighbor)
                frontier.append(neighbor)
    return not remaining


def _nested_serpentine_blocks(
    height: int,
    width: int,
    sizes: tuple[int, ...],
    band_block_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Split ragged horizontal bands by their perpendicular compact walks."""

    outer_path = tuple(
        row * width + column
        for row in range(height)
        for column in (range(width) if row % 2 == 0 else range(width - 1, -1, -1))
    )
    blocks: list[tuple[int, ...]] = []
    token_offset = 0
    block_offset = 0
    for band_blocks in band_block_counts:
        band_sizes = sizes[block_offset : block_offset + band_blocks]
        band_tokens = sum(band_sizes)
        band_cells = set(outer_path[token_offset : token_offset + band_tokens])
        rows_by_column: list[list[int]] = [[] for _ in range(width)]
        for token in band_cells:
            row, column = divmod(token, width)
            rows_by_column[column].append(row)
        for rows in rows_by_column:
            rows.sort()
        best_band: (
            tuple[
                _Cost,
                bool,
                int,
                tuple[tuple[int, ...], ...],
            ]
            | None
        ) = None
        for reverse_columns in (False, True):
            columns = range(width - 1, -1, -1) if reverse_columns else range(width)
            for phase in (0, 1):
                inner_path = tuple(
                    row * width + column
                    for column_index, column in enumerate(columns)
                    for row in (
                        rows_by_column[column]
                        if (column_index + phase) % 2 == 0
                        else reversed(rows_by_column[column])
                    )
                )
                trial: list[tuple[int, ...]] = []
                offset = 0
                for size in band_sizes:
                    block = inner_path[offset : offset + size]
                    if not _block_is_connected(block, width):
                        break
                    trial.append(block)
                    offset += size
                if len(trial) != band_blocks:
                    continue
                trial_blocks = tuple(trial)
                candidate = (
                    _partition_cost(trial_blocks, width),
                    reverse_columns,
                    phase,
                    trial_blocks,
                )
                if best_band is None or candidate < best_band:
                    best_band = candidate
        if best_band is None:
            return None
        blocks.extend(best_band[3])
        token_offset += band_tokens
        block_offset += band_blocks

    if token_offset != height * width or block_offset != len(sizes):
        raise RuntimeError("compact band construction did not consume the exact grid")
    return tuple(blocks)


def _compact_candidate(
    height: int,
    width: int,
    capacity: int,
    target_blocks: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Return the best lower-bound-guided two-level compact candidate."""

    small_size, large_blocks = divmod(height * width, target_blocks)
    if small_size + bool(large_blocks) > capacity:
        raise RuntimeError("balanced block size exceeds physical capacity")

    best: (
        tuple[
            _Cost,
            bool,
            int,
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
        ]
        | None
    ) = None
    for transposed in (False, True):
        oriented_height, oriented_width = (width, height) if transposed else (height, width)
        ideal_bands = math.sqrt(target_blocks * oriented_height / oriented_width)
        band_counts = {
            max(1, min(oriented_height, target_blocks, math.floor(ideal_bands))),
            max(1, min(oriented_height, target_blocks, math.ceil(ideal_bands))),
        }
        for band_count in sorted(band_counts):
            small_band, large_bands = divmod(target_blocks, band_count)
            for band_pattern in _balanced_binary_patterns(band_count, large_bands):
                band_block_counts = tuple(small_band + bit for bit in band_pattern)
                for size_pattern in _balanced_binary_patterns(target_blocks, large_blocks):
                    sizes = tuple(small_size + bit for bit in size_pattern)
                    oriented_blocks = _nested_serpentine_blocks(
                        oriented_height,
                        oriented_width,
                        sizes,
                        band_block_counts,
                    )
                    if oriented_blocks is None:
                        continue
                    blocks = (
                        _transpose_to_raster(oriented_blocks, height, width)
                        if transposed
                        else oriented_blocks
                    )
                    candidate = (
                        _partition_cost(blocks, width),
                        transposed,
                        band_count,
                        band_block_counts,
                        sizes,
                        blocks,
                    )
                    if best is None or candidate < best:
                        best = candidate
    return None if best is None else best[5]


def _partition_adjacency(
    height: int,
    width: int,
    token_to_block: list[int],
    block_count: int,
) -> tuple[tuple[bool, ...], ...]:
    adjacency = [[False] * block_count for _ in range(block_count)]
    for block_id in range(block_count):
        adjacency[block_id][block_id] = True
    for row in range(height):
        for column in range(width):
            token = row * width + column
            source = token_to_block[token]
            if column + 1 < width:
                target = token_to_block[token + 1]
                adjacency[source][target] = True
                adjacency[target][source] = True
            if row + 1 < height:
                target = token_to_block[token + width]
                adjacency[source][target] = True
                adjacency[target][source] = True
    return tuple(tuple(row) for row in adjacency)


@lru_cache(maxsize=128)
def make_ragged_2d_partition(
    height: int,
    width: int,
    capacity: int = 64,
    *,
    include_adjacency: bool = True,
) -> Ragged2DPartition:
    """Return the frozen compact partition for an arbitrary grid."""

    if any(type(value) is not int or value <= 0 for value in (height, width)):
        raise ValueError("height and width must be positive integers")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if type(include_adjacency) is not bool:
        raise TypeError("include_adjacency must be a built-in bool")

    block_count = math.ceil(height * width / capacity)
    if height == 1 or width == 1:
        # A connected subset of a one-dimensional grid is an interval.  The
        # moment-minimizing exact cover therefore has balanced consecutive
        # interval sizes; this is the same candidate selected by the general
        # two-orientation DP without its quadratic degenerate-axis search.
        selected = _band_serpentine_blocks(1, height * width, block_count)
    else:
        horizontal_blocks, _ = _orientation_candidate(height, width, capacity, block_count)
        transposed_blocks, _ = _orientation_candidate(width, height, capacity, block_count)
        vertical_blocks = _transpose_to_raster(transposed_blocks, height, width)
        # Score in the original coordinate system.  The Boolean makes exact
        # ties deterministic and favors the ordinary construction.
        stripe_selected = (
            horizontal_blocks
            if (_partition_cost(horizontal_blocks, width), False)
            <= (_partition_cost(vertical_blocks, width), True)
            else vertical_blocks
        )
        compact = _compact_candidate(height, width, capacity, block_count)
        # Keep the proven full-band construction as a fallback and as the
        # deterministic winner of exact ties.  The new search therefore only
        # changes layouts when it improves the complete shape objective.
        selected = (
            compact
            if compact is not None
            and _partition_cost(compact, width) < _partition_cost(stripe_selected, width)
            else stripe_selected
        )
    # Serpentine order proves connectivity; canonical raster order makes
    # rectangular blocks byte-equivalent to the historical aligned layout.
    blocks = tuple(tuple(sorted(block)) for block in selected)

    token_to_block = [-1] * (height * width)
    for block_id, block in enumerate(blocks):
        if not block or len(block) > capacity:
            raise RuntimeError("ragged block violates physical K capacity")
        for token in block:
            if not 0 <= token < height * width:
                raise RuntimeError("ragged block contains an invalid token")
            if token_to_block[token] != -1:
                raise RuntimeError("ragged partition assigned a token twice")
            token_to_block[token] = block_id
    if any(block_id < 0 for block_id in token_to_block):
        raise RuntimeError("ragged partition did not cover the complete grid")
    if len(blocks) != block_count:
        raise RuntimeError("ragged partition did not reach the capacity lower bound")

    return Ragged2DPartition(
        height=height,
        width=width,
        capacity=capacity,
        blocks=blocks,
        token_to_block=tuple(token_to_block),
        adjacency=(
            _partition_adjacency(height, width, token_to_block, block_count)
            if include_adjacency
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class AnemoiLayout:
    indices: torch.Tensor
    slot_valid: torch.Tensor
    counts: torch.Tensor
    inverse: torch.Tensor
    anchors: torch.Tensor | None
    anchor_ids: torch.Tensor | None
    anchor_count: int
    blocks_per_frame: int
    logical_block: int
    _ready_event: torch.cuda.Event | None = field(default=None, repr=False, compare=False)


def materialize_layout(
    device: torch.device | str,
    video_shape: tuple[int, int, int] | list[int],
    query_block_size: int = 64,
    enable_anchors: bool = False,
) -> AnemoiLayout:
    if query_block_size not in (64, 128):
        raise ValueError("query_block_size must be 64 or 128")
    if len(video_shape) != 3 or any(type(x) is not int or x <= 0 for x in video_shape):
        raise ValueError("video_shape must contain three positive integers")
    layout = _materialize_layout(
        torch.device(device), tuple(video_shape), query_block_size, enable_anchors
    )
    if layout.indices.is_cuda:
        stream = torch.cuda.current_stream(layout.indices.device)
        if layout._ready_event is not None:
            stream.wait_event(layout._ready_event)
        # Native DLPack launches are invisible to the caching allocator. Record
        # every cached tensor on each consumer stream so cache eviction cannot
        # recycle its storage before raw-pointer kernels finish.
        layout.indices.record_stream(stream)
        layout.slot_valid.record_stream(stream)
        layout.counts.record_stream(stream)
        layout.inverse.record_stream(stream)
        if layout.anchors is not None:
            layout.anchors.record_stream(stream)
        if layout.anchor_ids is not None:
            layout.anchor_ids.record_stream(stream)
    return layout


@lru_cache(maxsize=16)
def _materialize_layout(
    device: torch.device,
    video_shape: tuple[int, int, int],
    query_block_size: int,
    enable_anchors: bool,
) -> AnemoiLayout:
    """Build the packing tensors once per (device, shape, block, anchors) key.

    The values are pure functions of the key, so a run with a fixed video shape
    (every attention call of a diffusion sampling loop) reuses one layout
    instead of rebuilding index tables and re-uploading them on each call.
    """
    frames, height, width = video_shape
    partition = make_ragged_2d_partition(
        height, width, query_block_size, include_adjacency=enable_anchors
    )
    frame_tokens = height * width
    local = torch.tensor(
        tuple(block + (0,) * (query_block_size - len(block)) for block in partition.blocks),
        device=device,
        dtype=torch.int64,
    )
    counts = torch.tensor(partition.counts, device=device, dtype=torch.int32).repeat(frames)
    frame_offsets = (torch.arange(frames, device=device, dtype=torch.int64) * frame_tokens).view(
        frames, 1, 1
    )
    indices = local.view(1, partition.block_count, query_block_size) + frame_offsets
    slot_valid = torch.arange(query_block_size, device=device).view(
        1, query_block_size
    ) < counts.view(-1, 1)
    indices.masked_fill_(~slot_valid.view(frames, partition.block_count, query_block_size), 0)
    indices = indices.reshape(-1).contiguous()
    slot_valid = slot_valid.reshape(-1).contiguous()
    inverse = torch.empty(frames * frame_tokens, device=device, dtype=torch.int64)
    slots = torch.arange(indices.numel(), device=device, dtype=torch.int64)
    inverse.scatter_(0, indices[slot_valid], slots[slot_valid])
    counts = counts.contiguous()

    anchor_ids = anchors = None
    anchor_count = 0
    if enable_anchors:
        total_blocks = frames * partition.block_count
        ids = [
            (frame * partition.block_count + source) * total_blocks
            + frame * partition.block_count
            + target
            for frame in range(frames)
            for source, row in enumerate(partition.adjacency)
            for target, adjacent in enumerate(row)
            if adjacent
        ]
        anchor_count = len(ids)
        anchor_ids = torch.tensor(ids, dtype=torch.int32, device=device)
        anchors = torch.zeros((total_blocks, total_blocks), dtype=torch.bool, device=device)
        anchors.view(-1)[anchor_ids.long()] = True

    ready_event = None
    if indices.is_cuda:
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(indices.device))
    return AnemoiLayout(
        indices=indices,
        slot_valid=slot_valid,
        counts=counts,
        inverse=inverse,
        anchors=anchors,
        anchor_ids=anchor_ids,
        anchor_count=anchor_count,
        blocks_per_frame=partition.block_count,
        logical_block=query_block_size,
        _ready_event=ready_event,
    )
