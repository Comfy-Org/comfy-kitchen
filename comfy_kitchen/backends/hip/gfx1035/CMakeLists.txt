# gfx1035 (RDNA2) SageAttention port: a self-contained extension that only ever
# compiles for gfx103x targets.
#
# Why a separate target rather than sources in ../CMakeLists.txt: RDNA2 has no
# matrix cores, so the WMMA sage_attention/*.hip sources in the main HIP backend
# do not build for gfx103x at all. Keeping the port in its own CMake project
# means the main backend stays byte-identical for gfx11xx/gfx12xx, which is the
# part of the tree upstream owns; this directory can be deleted as a unit to
# undo the port.
#
# The kernels are ported from SageAttention (csrc/qattn/attn_gfx103x.cu) and are
# unmodified apart from the extension namespace, so a fresh upstream sync is a
# plain copy of that one file.
cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Only RDNA2 gfx103x targets are accepted. Anything else is a misconfiguration:
# this extension registers nothing else, and compiling it for gfx11xx would
# silently produce a library whose kernels are dead code.
set(_CK_GFX1035_VALID
    gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
)
if(NOT DEFINED COMFY_HIP_ARCHS OR COMFY_HIP_ARCHS STREQUAL "")
    set(COMFY_HIP_ARCHS gfx1035)
endif()
foreach(_arch IN LISTS COMFY_HIP_ARCHS)
    if(NOT _arch IN_LIST _CK_GFX1035_VALID)
        message(FATAL_ERROR
            "comfy_kitchen gfx1035 extension accepts only RDNA2 targets "
            "(${_CK_GFX1035_VALID}); got '${_arch}'. "
            "Unset COMFY_HIP_ARCHS for this target."
        )
    endif()
endforeach()
set(CMAKE_HIP_ARCHITECTURES "${COMFY_HIP_ARCHS}")

# A pip-installed rocm-sdk puts the device bitcode under lib/llvm/amdgcn/bitcode
# instead of <root>/amdgcn/bitcode. Must precede project(): CMake compiles a
# probe program when it identifies the HIP compiler.
if(DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
    set(_ck_devlib "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
    if(EXISTS "${_ck_devlib}" AND NOT EXISTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/amdgcn/bitcode")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --rocm-device-lib-path=\"${_ck_devlib}\"")
    endif()
    if(EXISTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/include/hip/hip_version.h")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --hip-path=\"${CMAKE_HIP_COMPILER_ROCM_ROOT}\"")
    endif()
endif()

project(comfy_kitchen_gfx1035 LANGUAGES CXX HIP)

message(STATUS "gfx1035 extension architectures: ${CMAKE_HIP_ARCHITECTURES}")

# AMDGPU scheduler hints the RDNA2 kernels were tuned with. These reach clang
# through -mllvm pairs, which is why they live in the free-form CMAKE_HIP_FLAGS
# string rather than as individual target_compile_options arguments.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -std=c++17 -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -mllvm -amdgpu-max-memory-clause=32 -mllvm -amdgpu-vgpr-index-mode=1")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fdiagnostics-color=always")

# ABI3 so the extension outlives a minor Python bump, matching the main HIP
# backend's wheel policy.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

# hip-config.cmake hard-errors when this is unset.
if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
endif()
find_package(hip REQUIRED CONFIG)

# Torch's stable tensor/ops API: header-only on the compile side, resolved
# through the C API at runtime, so no torch library is linked.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE _TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_RESULT
)
if(NOT _TORCH_RESULT EQUAL 0 OR NOT EXISTS "${_TORCH_DIR}/include/torch/csrc/stable/library.h")
    message(FATAL_ERROR
        "PyTorch stable headers not found; build with the same interpreter that "
        "ships this package."
    )
endif()
set(_TORCH_INCLUDE_DIR "${_TORCH_DIR}/include")
set(_TORCH_LIB_DIR "${_TORCH_DIR}/lib")
# The torch stable API is header-only, but the aoti symbols it calls (and the
# Stream handle they return) resolve at load time from the torch libraries, so
# the module must record them as a dependency. Python imports torch before this
# module, which is what makes the RTLD resolution succeed.
list(APPEND CMAKE_PREFIX_PATH "${_TORCH_DIR}")
# TorchConfig pulls in Caffe2's LoadHIP, which picks the build's GPU arch with
# torch_hip_get_arch_list(). That macro shells out to `rocm_agent_enumerator`
# (a bash script the Windows HIP SDK does not ship) whenever PYTORCH_ROCM_ARCH
# is unset, and turns the resulting spawn failure into a FATAL_ERROR. The arch
# list is already decided and validated above, so hand torch that instead of
# letting it probe a machine it cannot probe. Space-separated, as torch expects.
string(REPLACE ";" " " _CK_GFX1035_ARCH_STR "${COMFY_HIP_ARCHS}")
set(ENV{PYTORCH_ROCM_ARCH} "${_CK_GFX1035_ARCH_STR}")
find_package(Torch REQUIRED)
set(_TORCH_LINK_LIBS ${TORCH_LIBRARIES})

set(_CXX11_ABI 0)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE _CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _ABI_RESULT
)
if(NOT _ABI_RESULT EQUAL 0)
    set(_CXX11_ABI 0)
endif()

# Kernel compilation. Mirrors the upstream flag set: -std=c++17 plus the AMDGPU
# scheduler hints the RDNA2 kernels were tuned with.
add_library(comfy_kitchen_gfx1035_kernels OBJECT
    attn_gfx103x.hip
)
target_compile_options(comfy_kitchen_gfx1035_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-std=c++17>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fgpu-flush-denormals-to-zero>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-offload-uniform-block>
)
if(COMFY_ENABLE_LINEINFO OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(comfy_kitchen_gfx1035_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>
    )
endif()
target_include_directories(comfy_kitchen_gfx1035_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_TORCH_INCLUDE_DIR}
)
target_compile_definitions(comfy_kitchen_gfx1035_kernels PRIVATE
    __HIP_PLATFORM_AMD__=1
    USE_ROCM=1
    _GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI}
    SAGEATTN_VT_GLOBAL=1
)
target_link_libraries(comfy_kitchen_gfx1035_kernels PRIVATE hip::host)

# The python extension module. pybind_gfx1035.cpp supplies both the module init
# and the op registration; the kernel objects carry the device code. abi3 keeps
# the artifact loadable on any Python >= 3.12.
add_library(_qattn_gfx1035 SHARED
    pybind_gfx1035.cpp
    $<TARGET_OBJECTS:comfy_kitchen_gfx1035_kernels>
)
set_target_properties(_qattn_gfx1035 PROPERTIES
    PREFIX ""
    SUFFIX ".abi3.so"
)
if(WIN32)
    set_target_properties(_qattn_gfx1035 PROPERTIES SUFFIX ".abi3.pyd")
endif()
# On Windows a SHARED library is a RUNTIME artifact, so CMAKE_LIBRARY_OUTPUT_DIRECTORY
# -- the only output directory setup.py passes -- does not apply to it, and the .pyd is
# left behind in the build tree where the wheel never picks it up. The build succeeds
# and the import is missing, so route both artifact kinds explicitly, per-config
# included for the multi-config generators the CUDA backend still allows.
if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(_CK_OUT_PROPS
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
    foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND _CK_OUT_PROPS
            "LIBRARY_OUTPUT_DIRECTORY_${_cfg}" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
            "RUNTIME_OUTPUT_DIRECTORY_${_cfg}" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        )
    endforeach()
    set_target_properties(_qattn_gfx1035 PROPERTIES ${_CK_OUT_PROPS})
endif()
target_compile_options(_qattn_gfx1035 PRIVATE
    -O3
    -std=c++17
    -fvisibility=hidden
)
target_compile_definitions(_qattn_gfx1035 PRIVATE
    Py_LIMITED_API=0x030C0000
    _GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI}
)
target_include_directories(_qattn_gfx1035 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_TORCH_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)
if(WIN32)
    # Py_LIMITED_API makes Python.h emit `/DEFAULTLIB:python3.lib`, and the
    # aoti_* calls are dllimports, so lld-link needs an import library for
    # python3 plus one per torch DLL that owns them (aoti_torch_get_current_cuda_
    # stream is in torch_hip, the rest in torch_cpu). The imported targets are
    # unusable here: Python::SABIModule carries only the DLL with no implib, and
    # torch's targets drag in the whole rocblas/hipblaslt/MIOpen closure to
    # resolve a handful of stable-ABI symbols. Name the import libraries
    # directly, then check them, so a layout change reports itself here instead
    # of as a lld-link "could not open" after a full device compile.
    if(_Python_SABI_LIBRARY_RELEASE AND EXISTS "${_Python_SABI_LIBRARY_RELEASE}")
        set(_CK_PY_IMPLIB "${_Python_SABI_LIBRARY_RELEASE}")
    else()
        find_library(_CK_PY_IMPLIB
            NAMES python3
            HINTS ${Python_SABI_LIBRARY_DIRS} ${Python_LIBRARY_DIRS} ${Python_ROOT_DIR}
            PATH_SUFFIXES libs lib
        )
    endif()
    set(_CK_TORCH_IMPLIBS "")
    set(_CK_MISSING_IMPLIBS "")
    foreach(_implib IN ITEMS torch_cpu torch_hip c10 c10_hip)
        if(EXISTS "${_TORCH_LIB_DIR}/${_implib}.lib")
            list(APPEND _CK_TORCH_IMPLIBS "${_TORCH_LIB_DIR}/${_implib}.lib")
        else()
            list(APPEND _CK_MISSING_IMPLIBS "${_implib}.lib")
        endif()
    endforeach()
    if(NOT _CK_PY_IMPLIB OR _CK_MISSING_IMPLIBS)
        message(FATAL_ERROR
            "comfy_kitchen gfx1035 extension is missing Windows import "
            "libraries it links directly: "
            "python3=${_CK_PY_IMPLIB} (want python3.lib) "
            "torch=[${_CK_MISSING_IMPLIBS}] (want them in ${_TORCH_LIB_DIR}). "
            "Build with a torch wheel that ships its .lib files."
        )
    endif()
    target_link_libraries(_qattn_gfx1035 PRIVATE ${_CK_PY_IMPLIB} ${_CK_TORCH_IMPLIBS})
else()
    target_link_libraries(_qattn_gfx1035 PRIVATE ${Python_LIBRARIES} hip::host ${_TORCH_LINK_LIBS})
endif()

install(TARGETS _qattn_gfx1035
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
