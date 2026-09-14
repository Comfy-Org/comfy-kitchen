// SPDX-License-Identifier: Apache-2.0
// Optional classic cuBLAS for Anemoi DraftMap. No cuBLAS import-library ABI.
#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <cublas_v2.h>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace comfy {

struct AnemoiCublasFunctions {
    // decltype preserves CUBLASWINAPI (__stdcall on Windows). These are
    // unevaluated declarations, not direct references to imported symbols.
    using Create = decltype(&::cublasCreate_v2);
    using Destroy = decltype(&::cublasDestroy_v2);
    using SetStream = decltype(&::cublasSetStream_v2);
    using SetPointerMode = decltype(&::cublasSetPointerMode_v2);
    using SetMathMode = decltype(&::cublasSetMathMode);
    // cublasGemmStridedBatchedEx has an additional C++ convenience overload;
    // spell the stable classic API explicitly to select its compute-type ABI.
    using GemmStridedBatchedEx = cublasStatus_t (CUBLASWINAPI *)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t,
        int, int, int, const void*, const void*, cudaDataType, int, long long,
        const void*, cudaDataType, int, long long, const void*, void*,
        cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);

    Create create = nullptr;
    Destroy destroy = nullptr;
    SetStream set_stream = nullptr;
    SetPointerMode set_pointer_mode = nullptr;
    SetMathMode set_math_mode = nullptr;
    GemmStridedBatchedEx gemm_strided_batched_ex = nullptr;
};

class AnemoiCublasRuntime final {
public:
    static const AnemoiCublasRuntime& instance() {
        // C++ local-static initialization is thread safe. Deliberately retain
        // the tiny table and successful library reference until process exit:
        // thread-local cuBLAS handles may be destroyed after other statics.
        // Never dlclose/FreeLibrary live function pointers during teardown.
        static const auto* runtime = new AnemoiCublasRuntime();
        return *runtime;
    }

    bool is_available() const noexcept { return library_ != nullptr; }
    const std::string& error_message() const noexcept { return error_; }

    const AnemoiCublasFunctions& functions() const {
        if (!is_available()) throw std::runtime_error(error_);
        return functions_;
    }

    AnemoiCublasRuntime(const AnemoiCublasRuntime&) = delete;
    AnemoiCublasRuntime& operator=(const AnemoiCublasRuntime&) = delete;

private:
#ifdef _WIN32
    using Library = HMODULE;
    static Library open(const char* name) {
        // Fixed ASCII basenames below only; avoid the unsafe legacy DLL search
        // through cwd/PATH. Includes os.add_dll_directory user directories and
        // already loaded PyTorch dependencies without changing process policy.
        const wchar_t* wide = std::strcmp(name, "cublas64_13.dll") == 0
                                ? L"cublas64_13.dll" : L"cublas64_12.dll";
        return LoadLibraryExW(wide, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
    static void close(Library library) noexcept { FreeLibrary(library); }
    static std::string load_error() {
        return "Windows loader error " + std::to_string(GetLastError());
    }
#else
    using Library = void*;
    static Library open(const char* name) {
        // Reuses a matching SONAME already loaded by PyTorch when available.
        // No environment overrides, directory scans, or unversioned ABI guesses.
        return dlopen(name, RTLD_NOW | RTLD_LOCAL);
    }
    static void close(Library library) noexcept { dlclose(library); }
    static std::string load_error() {
        const char* message = dlerror();
        return message ? message : "dynamic loader supplied no diagnostic";
    }
#endif

    template<class Function>
    static bool symbol(Library library, const char* name, Function& function,
                       std::string& error) {
#ifdef _WIN32
        const auto address = GetProcAddress(library, name);
        if (!address) {
            const std::string failure = load_error();
            error = std::string("missing ") + name + ": " + failure;
            return false;
        }
#else
        dlerror(); // Clear errors from earlier candidate/symbol lookups.
        void* address = dlsym(library, name);
        const char* failure = dlerror();
        if (failure || !address) {
            error = std::string("missing ") + name + ": " +
                    (failure ? failure : "null symbol address");
            return false;
        }
#endif
        function = reinterpret_cast<Function>(address);
        return true;
    }

    static bool symbols(Library library, AnemoiCublasFunctions& api,
                        std::string& error) {
        return symbol(library, "cublasCreate_v2", api.create, error) &&
               symbol(library, "cublasDestroy_v2", api.destroy, error) &&
               symbol(library, "cublasSetStream_v2", api.set_stream, error) &&
               symbol(library, "cublasSetPointerMode_v2", api.set_pointer_mode, error) &&
               symbol(library, "cublasSetMathMode", api.set_math_mode, error) &&
               symbol(library, "cublasGemmStridedBatchedEx", api.gemm_strided_batched_ex, error);
    }

    AnemoiCublasRuntime() {
#ifdef _WIN32
        const char* const candidates[] = {"cublas64_13.dll", "cublas64_12.dll"};
#else
        const char* const candidates[] = {"libcublas.so.13", "libcublas.so.12"};
#endif
        std::string failures;
        for (const char* name : candidates) {
            Library candidate = open(name);
            if (!candidate) {
                const std::string failure = load_error();
                failures += std::string("\n  ") + name + ": " + failure;
                continue;
            }
            AnemoiCublasFunctions api;
            std::string error;
            if (symbols(candidate, api, error)) {
                functions_ = api;
                library_ = candidate;
                return;
            }
            close(candidate);
            failures += std::string("\n  ") + name + ": " + error;
        }
        error_ = "Anemoi DraftMap requires an available classic cuBLAS 12.x or 13.x runtime "
                 "with its dependencies. Install the NVIDIA cuBLAS runtime compatible with "
                 "your CUDA/PyTorch environment, then restart the process. Other Kitchen "
                 "operators do not require this optional Anemoi dependency. Tried:" + failures;
    }

    Library library_ = nullptr;
    AnemoiCublasFunctions functions_;
    std::string error_;
};

// Call only from an Anemoi draft invocation, never module registration/static
// initialization. A missing library is isolated to that invocation. Successful
// tables are immutable; failed lookups are cached until process restart.
inline const AnemoiCublasFunctions& anemoi_cublas() {
    return AnemoiCublasRuntime::instance().functions();
}

} // namespace comfy
