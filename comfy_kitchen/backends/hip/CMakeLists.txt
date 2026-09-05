# 3.21 for the HIP language (CMAKE_HIP_ARCHITECTURES), 3.26 for the
# Development.SABIModule component the abi3 module below needs.
cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The exact target list and WMMA family classification are shared with setup.py
# and runtime dispatch. Keep this before project(): invalid targets must fail
# before compiler detection, and CMAKE_HIP_ARCHITECTURES must already be set.
set(_CK_HIP_ARCH_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/architectures.json")
file(READ "${_CK_HIP_ARCH_MANIFEST}" _CK_HIP_ARCH_JSON)

function(_ck_read_arch_group output group_name)
    string(JSON _group_length LENGTH "${_CK_HIP_ARCH_JSON}" "${group_name}")
    if(_group_length EQUAL 0)
        message(FATAL_ERROR "${_CK_HIP_ARCH_MANIFEST}: ${group_name} must not be empty")
    endif()

    math(EXPR _last_index "${_group_length} - 1")
    set(_group)
    foreach(_index RANGE 0 ${_last_index})
        string(JSON _arch GET "${_CK_HIP_ARCH_JSON}" "${group_name}" ${_index})
        list(APPEND _group "${_arch}")
    endforeach()
    set(${output} "${_group}" PARENT_SCOPE)
endfunction()

_ck_read_arch_group(COMFY_HIP_ELEMENTWISE_ARCHS elementwise_only)
_ck_read_arch_group(COMFY_HIP_WMMA_GFX11_ARCHS wmma_gfx11)
_ck_read_arch_group(COMFY_HIP_WMMA_GFX12_ARCHS wmma_gfx12)
set(COMFY_HIP_SUPPORTED_ARCHS
    ${COMFY_HIP_ELEMENTWISE_ARCHS}
    ${COMFY_HIP_WMMA_GFX11_ARCHS}
    ${COMFY_HIP_WMMA_GFX12_ARCHS}
)
list(LENGTH COMFY_HIP_SUPPORTED_ARCHS _arch_count)
list(REMOVE_DUPLICATES COMFY_HIP_SUPPORTED_ARCHS)
list(LENGTH COMFY_HIP_SUPPORTED_ARCHS _unique_arch_count)
if(NOT _arch_count EQUAL _unique_arch_count)
    message(FATAL_ERROR "${_CK_HIP_ARCH_MANIFEST} contains duplicate targets")
endif()

if(NOT DEFINED COMFY_HIP_ARCHS OR COMFY_HIP_ARCHS STREQUAL "")
    set(COMFY_HIP_ARCHS ${COMFY_HIP_SUPPORTED_ARCHS})
endif()
foreach(_arch IN LISTS COMFY_HIP_ARCHS)
    if(NOT _arch IN_LIST COMFY_HIP_SUPPORTED_ARCHS)
        message(FATAL_ERROR
            "Unsupported COMFY_HIP_ARCHS target '${_arch}'. "
            "Validated targets: ${COMFY_HIP_SUPPORTED_ARCHS}"
        )
    endif()
endforeach()
set(CMAKE_HIP_ARCHITECTURES "${COMFY_HIP_ARCHS}")

function(_ck_make_arch_condition output)
    set(_condition "")
    foreach(_arch IN ITEMS ${ARGN})
        if(NOT _condition STREQUAL "")
            string(APPEND _condition " || ")
        endif()
        string(APPEND _condition "defined(__${_arch}__)")
    endforeach()
    set(${output} "${_condition}" PARENT_SCOPE)
endfunction()

_ck_make_arch_condition(COMFY_HIP_GFX11_CONDITION ${COMFY_HIP_WMMA_GFX11_ARCHS})
_ck_make_arch_condition(COMFY_HIP_GFX12_CONDITION ${COMFY_HIP_WMMA_GFX12_ARCHS})
set(_CK_HIP_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_CK_HIP_GENERATED_INCLUDE_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/architecture_config.h.in"
    "${_CK_HIP_GENERATED_INCLUDE_DIR}/architecture_config.h"
    @ONLY
)

# A pip-installed rocm-sdk places the device bitcode under
# lib/llvm/amdgcn/bitcode rather than the <root>/amdgcn/bitcode clang probes for.
# Must precede project(), which compiles a test program with these flags.
if(DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
    set(_ck_devlib "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    if(EXISTS "${_ck_devlib}" AND NOT EXISTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/amdgcn/bitcode")
        # Quote the path: a ROCm root with spaces would otherwise split into extra
        # compiler arguments and fail the project() HIP probe.
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --rocm-device-lib-path=\"${_ck_devlib}\"")
    endif()

    # A stale HIP_PATH from an older SDK install outranks both the compiler's own
    # search and --rocm-path, so pin the runtime to the root that was resolved.
    if(EXISTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/include/hip/hip_version.h")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --hip-path=\"${CMAKE_HIP_COMPILER_ROCM_ROOT}\"")
    endif()
endif()

project(comfy_kitchen_hip LANGUAGES CXX HIP)

message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

# After project(), so ccache never wraps the compiler detection probes. HIP is
# the same clang++ driver as CXX here, so it defaults to the C++ launcher.
if(DEFINED COMFY_CXX_COMPILER_LAUNCHER)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${COMFY_CXX_COMPILER_LAUNCHER}")
    message(STATUS "C++ compiler launcher: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()
if(NOT DEFINED COMFY_HIP_COMPILER_LAUNCHER)
    set(COMFY_HIP_COMPILER_LAUNCHER "${COMFY_CXX_COMPILER_LAUNCHER}")
endif()
if(COMFY_HIP_COMPILER_LAUNCHER)
    set(CMAKE_HIP_COMPILER_LAUNCHER "${COMFY_HIP_COMPILER_LAUNCHER}")
    message(STATUS "HIP compiler launcher: ${CMAKE_HIP_COMPILER_LAUNCHER}")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fdiagnostics-color=always")

# Same components as the CUDA backend. A wheel only keeps the abi3 tag if every
# extension in it is abi3, so this has to match.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

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

# hip-config.cmake hard-errors ("Unexpected HIP_PLATFORM") when this is unset;
# only some ROCm layouts default it. amd is the only platform these kernels target.
if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
endif()

find_package(hip REQUIRED CONFIG)

set(HIP_SOURCES
    ops/per_tensor_fp8.hip
    ops/stochastic_round_fp8.hip
    ops/quantize_int8.hip
    ops/gemm_fp8.hip
    ops/gemm_int8.hip
    ops/convrot_w4a4.hip
    ops/adaln.hip
    ops/na3d.hip
    sage_attention/quant_qk_int8.hip
    sage_attention/quant_v_int8.hip
    sage_attention/int8_attn.hip
    sage_attention/sol_attn.hip
    sage_attention/sol_attn_preprocess.hip
    sage_attention/sol_attn_vtranspose.hip
    sage_attention/sol_attn_producer.hip
    sage_attention/sol_attn_route.hip
    sage_attention/sol_attn_exact.hip
    sage_attention/sol_attn_token.hip
    ops/apply_rope.hip
    ops/flash_decode.hip
    ops/rms_rope.hip
    ops/gemv_awq.hip
    ops/svdquant_w4a4.hip
    ops/w4a8_dequant.hip
    ops/w4a8_requant.hip
)

set(CPP_SOURCES
    dlpack_bindings.cpp
)

# Keep HIP compilation independent of the Python extension target. The kernels
# consume HIP and project headers only, so a compiler cache can reuse their
# objects across CPython wheel builds.
add_library(comfy_kitchen_hip_kernels OBJECT ${HIP_SOURCES})
target_compile_options(comfy_kitchen_hip_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
)
if(COMFY_ENABLE_LINEINFO OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(
        comfy_kitchen_hip_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>
    )
endif()
target_include_directories(comfy_kitchen_hip_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention
    ${_CK_HIP_GENERATED_INCLUDE_DIR}
)
target_link_libraries(comfy_kitchen_hip_kernels PRIVATE hip::host)

# NOMINSIZE: nanobind_add_module() otherwise puts -Os on the whole target, which the
# kernel object target already avoids. Keep it off the binding translation unit too.
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
    message(STATUS "Python ${Python_VERSION} detected - enabling stable ABI (abi3)")
    nanobind_add_module(_C NB_STATIC STABLE_ABI LTO NOMINSIZE ${CPP_SOURCES})
    if(WIN32)
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.pyd")
    else()
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
    endif()
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000)
else()
    message(STATUS "Python ${Python_VERSION} detected - building version-specific module")
    nanobind_add_module(_C NB_STATIC LTO NOMINSIZE ${CPP_SOURCES})
endif()

target_sources(_C PRIVATE $<TARGET_OBJECTS:comfy_kitchen_hip_kernels>)

target_compile_definitions(_C PRIVATE NB_STATIC)

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_CK_HIP_GENERATED_INCLUDE_DIR}
)

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
