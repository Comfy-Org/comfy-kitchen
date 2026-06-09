cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(comfy_kitchen_hip LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fdiagnostics-color=always")

# Find Python (required by nanobind - must use "Python" not "Python3")
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(COMFY_HAS_PYTHON_SABI OFF)
if(Python_VERSION VERSION_GREATER_EQUAL "3.12" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.26")
    find_package(Python COMPONENTS Development.SABIModule QUIET)
    if(Python_Development.SABIModule_FOUND)
        set(COMFY_HAS_PYTHON_SABI ON)
    endif()
endif()

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

find_package(hip REQUIRED CONFIG)

if(DEFINED COMFY_HIP_ARCHS AND NOT COMFY_HIP_ARCHS STREQUAL "")
    message(STATUS "HIP architectures: ${COMFY_HIP_ARCHS}")
    set(CMAKE_HIP_ARCHITECTURES "${COMFY_HIP_ARCHS}")
else()
    message(STATUS "HIP architectures: auto")
endif()

set(HIP_SOURCES
    ops/per_tensor_fp8.hip
    ops/stochastic_round_fp8.hip
)

set(CPP_SOURCES
    dlpack_bindings.cpp
)

if(Python_VERSION VERSION_GREATER_EQUAL "3.12" AND COMFY_HAS_PYTHON_SABI)
    message(STATUS "Python ${Python_VERSION} detected - enabling stable ABI (abi3)")
    nanobind_add_module(_C NB_STATIC STABLE_ABI LTO ${CPP_SOURCES} ${HIP_SOURCES})
    set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000)
else()
    message(STATUS "Python ${Python_VERSION} detected - building version-specific wheel")
    nanobind_add_module(_C NB_STATIC LTO ${CPP_SOURCES} ${HIP_SOURCES})
endif()

target_compile_definitions(_C PRIVATE NB_STATIC)

target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
)

if(COMFY_ENABLE_LINEINFO OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling HIP line information for profiling")
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>)
endif()

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
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
