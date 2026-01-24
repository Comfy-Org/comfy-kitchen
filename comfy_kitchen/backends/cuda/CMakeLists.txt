cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(comfy_kitchen_cuda LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
    # Use standard-conforming preprocessor (required by CCCL 2.8+)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
    # Static runtime linking - embeds C/C++ runtime (no DLL dependencies)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /Od /MTd")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /MT")
    endif()
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fdiagnostics-color=always")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    else()
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif()
endif()

# Find Python (required by nanobind - must use "Python" not "Python3")
# Only require Development.Module for extension modules, not full Development
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED Development.SABIModule)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_RESULT
)

if(NOT NB_RESULT EQUAL 0)
    message(FATAL_ERROR "nanobind not found. Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "12.8")
    message(FATAL_ERROR "ComfyKitchen CUDA backend requires CUDA 12.8 or newer. Found ${CUDAToolkit_VERSION}")
endif()

message(STATUS "Found CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA toolkit root: ${CUDAToolkit_ROOT}")

# Configure CUDA architectures
# Defaults are defined in setup.py and passed via -DCOMFY_CUDA_ARCHS
# Use setup.py --cuda-archs="..." to override
if(NOT DEFINED COMFY_CUDA_ARCHS)
    message(FATAL_ERROR "COMFY_CUDA_ARCHS must be set (should be passed from setup.py)")
endif()

message(STATUS "CUDA architectures: ${COMFY_CUDA_ARCHS}")
set(CMAKE_CUDA_ARCHITECTURES "${COMFY_CUDA_ARCHS}")

# CUDA-specific flags
set(CUDA_NVCC_FLAGS
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_BFLOAT162_OPERATORS__
    -U__CUDA_NO_BFLOAT162_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
)

# Pass MSVC standard-conforming preprocessor flag to nvcc (required by CCCL 2.8+)
if(MSVC)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler=/Zc:preprocessor)
    # Suppress harmless warnings from CCCL headers (unused variables in constexpr contexts)
    list(APPEND CUDA_NVCC_FLAGS --diag-suppress=177)
endif()

# Optimization flags
list(APPEND CUDA_NVCC_FLAGS -O3)

# Line info for profiling (Nsight Compute/Systems)
# Can be enabled via setup.py --lineinfo
if(NOT DEFINED COMFY_ENABLE_LINEINFO)
    set(COMFY_ENABLE_LINEINFO OFF)
endif()

if(COMFY_ENABLE_LINEINFO OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling NVCC line information for profiling")
    list(APPEND CUDA_NVCC_FLAGS -lineinfo)
endif()

# Source files
set(CUDA_SOURCES
    ops/per_tensor_quantize.cu
    ops/quantize_nvfp4.cu
    ops/quantize_mxfp8.cu
    ops/cublas_gemm_nvfp4.cu
    ops/apply_rope.cu
)

set(CPP_SOURCES
    dlpack_bindings.cpp
)

# Create the nanobind module
# Python 3.12+: Use stable ABI (abi3) for forward compatibility with 3.13+
# Python 3.10/3.11: Version-specific module (nanobind stable ABI requires 3.12+)
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
    message(STATUS "Python ${Python_VERSION} detected - enabling stable ABI (abi3)")
    nanobind_add_module(_C NB_STATIC STABLE_ABI LTO ${CPP_SOURCES} ${CUDA_SOURCES})
    # Explicitly set abi3 suffix
    if(WIN32)
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.pyd")
    else()
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
    endif()
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000)
else()
    message(STATUS "Python ${Python_VERSION} detected - building version-specific wheel")
    nanobind_add_module(_C NB_STATIC LTO ${CPP_SOURCES} ${CUDA_SOURCES})
endif()

target_compile_definitions(_C PRIVATE NB_STATIC)

# Set target properties
target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)

# Include directories
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
# Note: cuBLASLt is loaded dynamically at runtime via dlopen/LoadLibrary
target_link_libraries(_C PRIVATE
    CUDA::cudart_static
)

# On Windows, static cudart requires additional system libraries
if(WIN32)
    target_link_libraries(_C PRIVATE ws2_32)
else()
    # Linux: need libdl for dlopen/dlsym
    target_link_libraries(_C PRIVATE ${CMAKE_DL_LIBS})
endif()

# Ensure Python libraries are properly linked on Windows
if(WIN32)
    target_link_directories(_C PRIVATE ${Python_LIBRARY_DIRS})
endif()

# Set RPATH for runtime library loading
if(UNIX AND NOT APPLE)
    set_target_properties(_C PROPERTIES
        INSTALL_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIR}"
        BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    )
endif()

# Fix multi-config generator output directory issue
# By default, multi-config generators put output in Release/Debug subdirectories
# This ensures the library is placed directly in CMAKE_LIBRARY_OUTPUT_DIRECTORY
if(CMAKE_CONFIGURATION_TYPES)
    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
endif()

# Install target
install(TARGETS _C
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

