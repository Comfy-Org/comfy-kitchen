#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementComputeEpilogue = int32_t;
using ElementInputA = int8_t;
using ElementInputB = int8_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

int main() {
    std::cout << "Cutlass test compiled successfully." << std::endl;
    return 0;
}
