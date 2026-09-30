# Float-input indexed gate composition

This follow-up to #219 exposes the existing ConvRot256 input quantizer and the indexed GEMM epilogue as one compile-safe operation. No new CUDA kernel is introduced. The new operation accepts BF16 activations, supports the existing BF16 SwiGLU rounding, and preserves the prequantized operation's gate/residual semantics. CPU/other-device fallback uses the existing `int8_linear` result before addcmul to preserve backend rounding.

27 tests passed on RTX 5090 D v2 / Torch 2.12.0+cu130: 21 indexed GEMM cases and 6 composed CPU/CUDA cases, including real Inductor. Earlier fallback rounding failures are fixed; the passing result is from the corrected implementation.

A ComfyUI FC2-only consumer was tested on current master 8cfe5e1, Larry v4 INT8, 8 steps, 768×512/124 frames, 14,850 packed tokens, seed42, CFG1, shift12/4. Four alternating formal pairs after warmup: complete sampler mean **15.802266 → 15.583501 s (1.3844% reduction)**. Video/audio latent hashes match in all eight formal requests. Peak Torch allocation **1833455616 → 1833612288 bytes (+153 KiB)**, excluding external aimdo allocation. Raw records: `indexed-linear-sampler.json`.

This is independent from the existing SpeedKit outproj+FC2 consumer and historical complete optimizations. Percentages cannot be added. No VAE/decode or service throughput result is claimed. The wrapper is a candidate public integration boundary; keep this follow-up draft until #219 is accepted.


2026-10-01 review fix: the float-input wrapper now checks the existing ConvRot K/shared-memory limits before selecting its quantizer. Larger K and insufficient shared memory use `int8_linear` plus the indexed gate. On the isolated RTX 5090 D v2 container, **33 tests passed** in 2.57 seconds, including CPU/CUDA K=16640 with/without SwiGLU, Inductor, and forced insufficient shared memory. These results cover the wrapper and indexed-GEMM test files, not the entire Kitchen suite.
