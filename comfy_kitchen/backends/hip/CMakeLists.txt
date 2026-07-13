# 3.21 for the HIP language (CMAKE_HIP_ARCHITECTURES), 3.26 for the
# Development.SABIModule component the abi3 module below needs.
cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RDNA2/3/4. The GEMMs select a WMMA path per target in mma.h (gfx11 and gfx12
# have different fragment layouts); RDNA2 has no matrix cores and builds only the
# elementwise kernels. Override with --hip-archs / COMFY_HIP_ARCHS. Must precede
# project(), which fails HIP compiler detection when CMAKE_HIP_ARCHITECTURES is unset.
if(NOT DEFINED COMFY_HIP_ARCHS OR COMFY_HIP_ARCHS STREQUAL "")
    # set() already joins its arguments with ";", so these must not carry their own.
    set(COMFY_HIP_ARCHS
        "gfx1030" "gfx1031" "gfx1032" "gfx1033" "gfx1034" "gfx1035" "gfx1036"  # RDNA2
        "gfx1100" "gfx1101" "gfx1102" "gfx1103"                                # RDNA3
        "gfx1150" "gfx1151" "gfx1152" "gfx1153"                                # RDNA3.5
        "gfx1200" "gfx1201")                                                   # RDNA4
endif()
set(CMAKE_HIP_ARCHITECTURES "${COMFY_HIP_ARCHS}")

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

# Same components as the CUDA backend. A wheel only keeps the abi3 tag if every
# extension in it is abi3, so this has to match.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED Development.SABIModule)

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
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
    message(STATUS "Python ${Python_VERSION} detected - enabling stable ABI (abi3)")
    nanobind_add_module(_C NB_STATIC STABLE_ABI LTO NOMINSIZE ${CPP_SOURCES} ${HIP_SOURCES})
    if(WIN32)
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.pyd")
    else()
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
    endif()
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000)
else()
    message(STATUS "Python ${Python_VERSION} detected - building version-specific module")
    nanobind_add_module(_C NB_STATIC LTO NOMINSIZE ${CPP_SOURCES} ${HIP_SOURCES})
endif()

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
