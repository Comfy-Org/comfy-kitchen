# Draft has separate Ada and Blackwell instruction sets. The Ada (SM89)
# mma.sync kernel set builds for every requested architecture in [SM89, SM120)
# — Hopper (SM90) and SM100-class devices run the same kernels — while SM120
# and newer Blackwell-capability devices use the SM120a block-scaled kernels.
# Do not emit generic PTX for the SM120a block-scaled MMA kernels.
set(COMFY_DRAFT_SM89 OFF)
set(COMFY_DRAFT_SM120 OFF)
set(DRAFT_ADA_ARCHS "")
set(DRAFT_ADA_ARCH_NUMBERS "")
foreach(arch IN LISTS COMFY_CUDA_ARCHS)
    if(arch MATCHES "^([0-9]+)([af]?)(-real|-virtual)?$")
        set(num "${CMAKE_MATCH_1}")
        if(num GREATER_EQUAL 120)
            set(COMFY_DRAFT_SM120 ON)
        elseif(num GREATER_EQUAL 89)
            list(APPEND DRAFT_ADA_ARCHS "${arch}")
            list(APPEND DRAFT_ADA_ARCH_NUMBERS "${num}")
            set(COMFY_DRAFT_SM89 ON)
        endif()
    endif()
endforeach()

set(DRAFT_OBJECTS "")
set(DRAFT_BINDINGS "")
set(DRAFT_DEFINITIONS "")
set(DRAFT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/draft")

function(draft_object target)
    add_library(${target} OBJECT ${ARGN})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS};--pre-include=cassert>)
    set_target_properties(${target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(${target} PRIVATE ${DRAFT_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
endfunction()

if(COMFY_DRAFT_SM89 OR COMFY_DRAFT_SM120)
    set(shared_archs "")
    if(COMFY_DRAFT_SM89)
        # Same token list as the Ada attention kernels: the shared packing and
        # assembly code must offer a cubin for every architecture they serve.
        list(APPEND shared_archs ${DRAFT_ADA_ARCHS})
    endif()
    if(COMFY_DRAFT_SM120)
        list(APPEND shared_archs 120-real)
    endif()
    draft_object(comfy_kitchen_draft_shared
        ${DRAFT_DIR}/preparation/raster_preprocess.cu
        ${DRAFT_DIR}/preparation/value_preprocess.cu
        ${DRAFT_DIR}/assembly/output_assembly.cu
    )
    set_target_properties(comfy_kitchen_draft_shared PROPERTIES CUDA_ARCHITECTURES "${shared_archs}")
    list(APPEND DRAFT_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_draft_shared>)
    list(APPEND DRAFT_BINDINGS
         ${DRAFT_DIR}/draft_bindings.cpp ${DRAFT_DIR}/draft_plan.cpp)
    list(APPEND DRAFT_DEFINITIONS COMFY_DRAFT_SHARED)

    # These replace torch FP32 statistics/softmax, not the donor's approximate
    # attention instructions. Preserve expf/division rounding here.
    set(draft_math_flags ${CUDA_NVCC_FLAGS})
    list(REMOVE_ITEM draft_math_flags --use_fast_math)
    add_library(comfy_kitchen_draft_math OBJECT
        ${DRAFT_DIR}/draft_preprocess.cu ${DRAFT_DIR}/draft_draft.cu)
    target_compile_options(comfy_kitchen_draft_math PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${draft_math_flags};--pre-include=cassert>)
    target_include_directories(comfy_kitchen_draft_math PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(comfy_kitchen_draft_math PROPERTIES CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${shared_archs}")
    list(APPEND DRAFT_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_draft_math>)
endif()

if(COMFY_DRAFT_SM89)
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
    list(TRANSFORM sm89_sources PREPEND "${DRAFT_DIR}/attention/ada/")
    list(APPEND sm89_sources ${DRAFT_DIR}/draft/sm89_probability.cu
         ${DRAFT_DIR}/route/sm89_precision.cu)
    draft_object(comfy_kitchen_draft_sm89 ${sm89_sources})
    set_target_properties(comfy_kitchen_draft_sm89 PROPERTIES
        CUDA_ARCHITECTURES "${DRAFT_ADA_ARCHS}")
    # Capability numbers the Ada set covers; the bindings gate serves exactly
    # these, so unlisted (for example JIT-forward) architectures stay closed.
    list(REMOVE_DUPLICATES DRAFT_ADA_ARCH_NUMBERS)
    string(JOIN "," DRAFT_ADA_ARCH_NUMBERS_JOINED ${DRAFT_ADA_ARCH_NUMBERS})
    list(APPEND DRAFT_DEFINITIONS
         COMFY_DRAFT_SM89_ARCHS="${DRAFT_ADA_ARCH_NUMBERS_JOINED}")
    list(APPEND DRAFT_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_draft_sm89>)
    list(APPEND DRAFT_DEFINITIONS COMFY_DRAFT_SM89)
    list(APPEND DRAFT_BINDINGS ${DRAFT_DIR}/draft_execute_sm89.cpp)
endif()

if(COMFY_DRAFT_SM120)
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
    list(TRANSFORM sm120_sources PREPEND "${DRAFT_DIR}/attention/blackwell/")
    list(APPEND sm120_sources ${DRAFT_DIR}/preparation/sm120_microscaling.cu
         ${DRAFT_DIR}/draft/sm120_probability.cu ${DRAFT_DIR}/route/sm120_precision.cu)
    draft_object(comfy_kitchen_draft_sm120 ${sm120_sources})
    set_target_properties(comfy_kitchen_draft_sm120 PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(comfy_kitchen_draft_sm120 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_120a,code=sm_120a>)
    list(APPEND DRAFT_OBJECTS $<TARGET_OBJECTS:comfy_kitchen_draft_sm120>)
    list(APPEND DRAFT_DEFINITIONS COMFY_DRAFT_SM120)
    list(APPEND DRAFT_BINDINGS ${DRAFT_DIR}/draft_execute_sm120.cpp)
endif()
