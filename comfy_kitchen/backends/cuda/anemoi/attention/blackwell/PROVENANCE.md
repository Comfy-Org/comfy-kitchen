# SM120 source provenance

## Current functional layout

Paths below are relative to this document. Attention kernels, primitives and instantiations remain here; the SM120 preparation implementation is `../../preparation/sm120_microscaling.cu`, draft scorer is `../../draft/sm120_probability.cu`, and routing implementation is `../../route/sm120_precision.cu`. Private launch headers are in `../../include/`. The original `csrc/attention/cuda/sm120/` paths cited below describe the donor repository, not the current Kitchen layout.

Vendored from [Anemoi](https://github.com/anemoi-project/anemoi), revision `4e85afba741bdeaf2d9486cab19cb76d3e7985a4`, source directory `csrc/attention/cuda/sm120/`. Apache-2.0 license is preserved in `LICENSE`; per-file SpargeAttn/SageAttention/project attribution is preserved.

All seven primitive headers, 16 explicit instantiation translation units, kernel declaration and phase-composer bodies are preserved. The kernel header and two metadata instantiation functions replace `C10_CUDA_CHECK` with a framework-independent CUDA error checker. Host attention validation/dispatch is adapted to non-owning raw-memory descriptors, caller-owned output and explicit streams. Preparation, draft and route device kernels are retained; host launch wrappers are allocation-free adaptations. Python registration now lives only in `../../anemoi_bindings.cpp`; obsolete per-stage bindings were removed. No Torch/ATen/c10 headers or library ABI are used.

The migrated code keeps the original sparse routing, INT8/E4M3, MXFP8, NVFP4, FP16 and mixed precision math, prefix handling and compact MXFP8 path. Existing donor comments/warnings (including pure-INT8 dead FP16 bookkeeping and a softmax TODO comment) are not silently removed or presented as newly validated numerical behavior. GPU numerical verification remains required on SM120.
