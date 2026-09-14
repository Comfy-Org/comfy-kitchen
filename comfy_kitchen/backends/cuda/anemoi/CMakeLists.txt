# Anemoi has separate Ada and Blackwell instruction sets. The Ada (SM89)
# mma.sync kernel set builds for every requested architecture in [SM89, SM120)
# — Hopper (SM90) and SM100-class devices run the same kernels — while SM120
# and newer Blackwell-capability devices use the SM120a block-scaled kernels.
# Do not emit generic PTX for the SM120a block-scaled MMA kernels.
set(COMFY_ANEMOI_SM89 OFF)
set(COMFY_ANEMOI_SM120 OFF)
set(ANEMOI_ADA_ARCHS "")
set(ANEMOI_ADA_ARCH_NUMBERS "")
foreach(arch IN LISTS COMFY_CUDA_ARCHS)
    if(arch MATCHES "^([0-9]+)([af]?)(-real|-virtual)?$")
        set(num "${CMAKE_MATCH_1}")
        if(num GREATER_EQUAL 120)
            set(COMFY_ANEMOI_SM120 ON)
        elseif(num GREATER_EQUAL 89)
            list(APPEND ANEMOI_ADA_ARCHS "${arch}")
            list(APPEND ANEMOI_ADA_ARCH_NUMBERS "${num}")
            set(COMFY_ANEMOI_SM89 ON)
        endif()
    endif()
endforeach()

set(ANEMOI_OBJECTS "")
set(ANEMOI_BINDINGS "")
set(ANEMOI_DEFINITIONS "")
set(ANEMOI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/anemoi")

function(anemoi_object target)
    add_library(${target} OBJECT ${ARGN})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS};--pre-include=cassert>)
    set_target_properties(${target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(${target} PRIVATE ${ANEMOI_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
endfunction()

if(COMFY_ANEMOI_SM89 OR COMFY_ANEMOI_SM120)
    set(shared_archs "")
    if(COMFY_ANEMOI_SM89)
        # Same token list as the Ada attention kernels: the shared packing and
        # assembly code must offer a cubin for every architecture they serve.
        list(APPEND shared_archs ${ANEMOI_ADA_ARCHS})
    endif()
    if(COMFY_ANEMOI_SM120)
        list(APPEND shared_archs 120-real)
    endif()
    anemoi_object(comfy_kitchen_anemoi_shared
        ${ANEMOI_DIR}/preparation/raster_preprocess.cu
        ${ANEMOI_DIR}/preparation/value_preprocess.cu
        ${ANEMOI_DIR}/assembly/output_assembly.cu
    )
    set_target_properties(comfy_kitchen_anemoi_shared PROPERTIES CUDA_ARCHITECTURES "${shared_archs}")
    list(APPEND ANEMOI_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_anemoi_shared>)
    list(APPEND ANEMOI_BINDINGS
         ${ANEMOI_DIR}/anemoi_bindings.cpp ${ANEMOI_DIR}/anemoi_plan.cpp)
    list(APPEND ANEMOI_DEFINITIONS COMFY_ANEMOI_SHARED)

    # These replace torch FP32 statistics/softmax, not the donor's approximate
    # attention instructions. Preserve expf/division rounding here.
    set(anemoi_math_flags ${CUDA_NVCC_FLAGS})
    list(REMOVE_ITEM anemoi_math_flags --use_fast_math)
    add_library(comfy_kitchen_anemoi_math OBJECT
        ${ANEMOI_DIR}/anemoi_preprocess.cu ${ANEMOI_DIR}/anemoi_draft.cu)
    target_compile_options(comfy_kitchen_anemoi_math PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${anemoi_math_flags};--pre-include=cassert>)
    target_include_directories(comfy_kitchen_anemoi_math PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(comfy_kitchen_anemoi_math PROPERTIES CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${shared_archs}")
    list(APPEND ANEMOI_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_anemoi_math>)
endif()

if(COMFY_ANEMOI_SM89)
    set(sm89_sources
        k64_attention_host.cu
        instantiations/inst_k64_d128.cu
        instantiations/inst_k64_d128_int8_dense.cu
        instantiations/inst_k64_d64.cu
        instantiations/inst_k64_d64_int8_dense.cu
        instantiations/inst_q128_k64_d128.cu
        instantiations/inst_q128_k64_d128_fp16.cu
        instantiations/inst_q128_k64_d128_int8.cu
        instantiations/inst_q128_k64_d64.cu
        instantiations/inst_q128_k64_d64_fp16.cu
        instantiations/inst_q128_k64_d64_int8.cu
    )
    list(TRANSFORM sm89_sources PREPEND "${ANEMOI_DIR}/attention/ada/")
    list(APPEND sm89_sources ${ANEMOI_DIR}/draft/sm89_probability.cu
         ${ANEMOI_DIR}/route/sm89_precision.cu)
    anemoi_object(comfy_kitchen_anemoi_sm89 ${sm89_sources})
    set_target_properties(comfy_kitchen_anemoi_sm89 PROPERTIES
        CUDA_ARCHITECTURES "${ANEMOI_ADA_ARCHS}")
    # Capability numbers the Ada set covers; the bindings gate serves exactly
    # these, so unlisted (for example JIT-forward) architectures stay closed.
    list(REMOVE_DUPLICATES ANEMOI_ADA_ARCH_NUMBERS)
    string(JOIN "," ANEMOI_ADA_ARCH_NUMBERS_JOINED ${ANEMOI_ADA_ARCH_NUMBERS})
    list(APPEND ANEMOI_DEFINITIONS
         COMFY_ANEMOI_SM89_ARCHS="${ANEMOI_ADA_ARCH_NUMBERS_JOINED}")
    list(APPEND ANEMOI_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_anemoi_sm89>)
    list(APPEND ANEMOI_DEFINITIONS COMFY_ANEMOI_SM89)
    list(APPEND ANEMOI_BINDINGS ${ANEMOI_DIR}/anemoi_execute_sm89.cpp)
endif()

if(COMFY_ANEMOI_SM120)
    set(sm120_sources
        q64_attention_host.cu
        instantiations/inst_q64_k64_d128_fp16.cu
        instantiations/inst_q64_k64_d128_int8.cu
        instantiations/inst_q64_k64_d128_int8_fp16.cu
        instantiations/inst_q64_k64_d128_int8_dense.cu
        instantiations/inst_q64_k64_d128_nvfp4.cu
        instantiations/inst_q64_k64_d128_mxfp8.cu
        instantiations/inst_q64_k64_d128_nv_int8_fp16.cu
        instantiations/inst_q64_k64_d128_nv_mx_fp16.cu
        instantiations/inst_q128_k64_d128_fp16.cu
        instantiations/inst_q128_k64_d128_int8.cu
        instantiations/inst_q128_k64_d128_int8_dense.cu
        instantiations/inst_q128_k64_d128_nvfp4.cu
        instantiations/inst_q128_k64_d128_mxfp8.cu
        instantiations/inst_q128_k64_d128_mxfp8_compact.cu
        instantiations/inst_q128_k64_d128_nv_int8_fp16.cu
        instantiations/inst_q128_k64_d128_nv_mx_fp16.cu
    )
    list(TRANSFORM sm120_sources PREPEND "${ANEMOI_DIR}/attention/blackwell/")
    list(APPEND sm120_sources ${ANEMOI_DIR}/preparation/sm120_microscaling.cu
         ${ANEMOI_DIR}/draft/sm120_probability.cu ${ANEMOI_DIR}/route/sm120_precision.cu)
    anemoi_object(comfy_kitchen_anemoi_sm120 ${sm120_sources})
    set_target_properties(comfy_kitchen_anemoi_sm120 PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(comfy_kitchen_anemoi_sm120 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_120a,code=sm_120a>)
    list(APPEND ANEMOI_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_anemoi_sm120>)
    list(APPEND ANEMOI_DEFINITIONS COMFY_ANEMOI_SM120)
    list(APPEND ANEMOI_BINDINGS ${ANEMOI_DIR}/anemoi_execute_sm120.cpp)
endif()
