cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The WMMA kernels are gfx12-only (v_wmma_*_w32_gfx12 has no gfx11 encoding).
# Override with --hip-archs / COMFY_HIP_ARCHS. Must precede project(), which
# fails HIP compiler detection when CMAKE_HIP_ARCHITECTURES is unset.
if(NOT DEFINED COMFY_HIP_ARCHS OR COMFY_HIP_ARCHS STREQUAL "")
    set(COMFY_HIP_ARCHS "gfx1200;gfx1201")
endif()
set(CMAKE_HIP_ARCHITECTURES "${COMFY_HIP_ARCHS}")

# A pip-installed rocm-sdk places the device bitcode under
# lib/llvm/amdgcn/bitcode rather than the <root>/amdgcn/bitcode clang probes for.
# Must precede project(), which compiles a test program with these flags.
if(DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
    set(_ck_devlib "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    if(EXISTS "${_ck_devlib}" AND NOT EXISTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/amdgcn/bitcode")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --rocm-device-lib-path=${_ck_devlib}")
    endif()
endif()

project(comfy_kitchen_hip LANGUAGES CXX HIP)

message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fdiagnostics-color=always")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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

find_package(hip REQUIRED CONFIG)

set(HIP_SOURCES
    ops/per_tensor_fp8.hip
    ops/stochastic_round_fp8.hip
    ops/quantize_int8.hip
    ops/gemm_fp8.hip
    ops/gemm_int8.hip
    ops/convrot_w4a4.hip
    ops/adaln.hip
    ops/apply_rope.hip
    ops/gemv_awq.hip
    ops/svdquant_w4a4.hip
)

set(CPP_SOURCES
    dlpack_bindings.cpp
)

# NOMINSIZE: nanobind_add_module() otherwise puts -Os on the whole target, which the
# -O3 below would have to override by flag order alone. The kernels are the point of
# this module, so opt out rather than rely on which -O wins.
nanobind_add_module(_C NB_STATIC LTO NOMINSIZE ${CPP_SOURCES} ${HIP_SOURCES})

target_compile_definitions(_C PRIVATE NB_STATIC)

target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
)

if(COMFY_ENABLE_LINEINFO OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>)
endif()

target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(_C PRIVATE hip::host)

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

install(TARGETS _C
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
