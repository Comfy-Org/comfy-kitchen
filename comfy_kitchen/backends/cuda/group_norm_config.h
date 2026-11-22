// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace comfy {

inline int group_norm_chunk_rows(int64_t frames, int64_t channels, int64_t rows) {
    // Keep enough blocks for small frames without excessive partials on large ones.
    int chunk_rows = 64;
    while (chunk_rows < 1024 && rows * frames * ((channels + 63) / 64) > 512 * chunk_rows)
        chunk_rows *= 2;
    return chunk_rows;
}

inline int64_t group_norm_chunks(int64_t frames, int64_t channels, int64_t rows) {
    const int chunk_rows = group_norm_chunk_rows(frames, channels, rows);
    return (rows + chunk_rows - 1) / chunk_rows;
}

}  // namespace comfy
