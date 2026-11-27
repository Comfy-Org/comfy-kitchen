// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>

namespace comfy_deco {

// cc is major * 10 + minor. Only shapes and devices with a compiled image match.
bool core_supported(int sequence, int heads, int cc);

// q4 is [sequence, heads, 96] packed bytes. out is contiguous BF16
// [sequence, heads, 128]. For r = (heads * ceil(sequence/128)) % sms,
// scratch holds 139264*r bytes followed by r zero-initialized uint32 counters.
cudaError_t launch_core(int sequence, int heads, int sms, bool single_writer, const void* q4,
                        const void* q_sf, const void* score_scale, const void* v4_scale,
                        const void* image, void* out, void* scratch, cudaStream_t stream);

} // namespace comfy_deco
