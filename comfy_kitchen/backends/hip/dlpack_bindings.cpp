#include <cstring>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Matches comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE:
// 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8, 5=float8_e4m3fn, 6=float8_e5m2
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
    if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) {
        if (dtype.bits == 32) return 0;
        if (dtype.bits == 16) return 1;
        if (dtype.bits == 8) return 5;
    } else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat) && dtype.bits == 16) {
        return 2;
    } else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) && dtype.bits == 8) {
        return 3;
    } else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int) && dtype.bits == 8) {
        return 4;
    }
    return -1;
}

extern "C" {
    void launch_quantize_per_tensor_fp8_kernel(
        const void* input,
        const void* scale,
        void* output,
        int64_t numel,
        int input_dtype_code,
        int output_dtype_code,
        hipStream_t stream);

    void launch_dequantize_per_tensor_fp8_kernel(
        const void* input,
        const void* scale,
        void* output,
        int64_t numel,
        int input_dtype_code,
        int output_dtype_code,
        hipStream_t stream);

    void launch_stochastic_round_fp8_kernel(
        void* rng_and_output,
        const void* input,
        int64_t numel,
        int rng_dtype_code,
        int input_dtype_code,
        int output_dtype_code,
        hipStream_t stream);
}

void check_hip_launch() {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel launch failed: ") + hipGetErrorString(err));
    }
}

void quantize_per_tensor_fp8(
    nb::ndarray<> input,
    nb::ndarray<> scale,
    nb::ndarray<> output,
    int input_dtype_code,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    launch_quantize_per_tensor_fp8_kernel(
        input.data(), scale.data(), output.data(), numel,
        input_dtype_code, output_dtype_code, stream);
    check_hip_launch();
}

void dequantize_per_tensor_fp8(
    nb::ndarray<> input,
    nb::ndarray<> scale,
    nb::ndarray<> output,
    int input_dtype_code,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    launch_dequantize_per_tensor_fp8_kernel(
        input.data(), scale.data(), output.data(), numel,
        input_dtype_code, output_dtype_code, stream);
    check_hip_launch();
}

void stochastic_round_fp8(
    nb::ndarray<> rng_and_output,
    nb::ndarray<> input,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {

    int rng_dtype_code = map_dtype_to_code(rng_and_output.dtype());
    if (rng_dtype_code != 3) {
        throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
    }

    int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for stochastic_round_fp8");
    }

    if (output_dtype_code < 5 || output_dtype_code > 6) {
        throw std::runtime_error("Unsupported output dtype for stochastic_round_fp8");
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    launch_stochastic_round_fp8_kernel(
        rng_and_output.data(),
        input.data(),
        numel,
        rng_dtype_code,
        input_dtype_code,
        output_dtype_code,
        stream);
    check_hip_launch();
}

NB_MODULE(_C, m) {
    m.doc() = "ComfyKitchen HIP backend native operations";
    m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8,
          "Quantize FP32/FP16/BF16 tensor to per-tensor FP8");
    m.def("dequantize_per_tensor_fp8", &dequantize_per_tensor_fp8,
          "Dequantize per-tensor FP8 tensor");
    m.def("stochastic_round_fp8", &stochastic_round_fp8,
          "Stochastically round FP32/FP16/BF16 tensor to FP8 in-place over uint8 RNG storage");
}
