# MIOpen Winograd RxS kernels (gfx10, fp32)

Unmodified copies of MIOpen's `Conv_Winograd_v30_3_1` fp32 kernels from
[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) `projects/miopen/src/kernels`
(develop at 52ddd479a4e5, 2026-10-14; the gfx10 bodies last changed in d8ab1dd1d0b5),
MIT licensed (see LICENSE). They are what MIOpen's `ConvBinWinogradRxSf2x3g1` and
`ConvBinWinogradRxSf3x2` solvers run on gfx1010, for forward convs and, with the filter
reversed, for backward data (a transposed conv; `dilation2` serves stride 2). MIOpen picks one after timing both on
the first call with each new shape; the rdna1 backend assembles them at build time
(CMakeLists.txt) and picks by rule instead (`fp32_winograd_conv2d` in `../__init__.py`).

Only the `.inc` bodies for gfx10 are kept: the `.s` wrappers include the other
targets' bodies under `.if` branches the assembler does not take for gfx1010.
