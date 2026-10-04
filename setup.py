import importlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import ClassVar

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# Parse command-line early to check for --no-cuda flag
# This needs to happen before get_extensions() is called
# Usage: python setup.py install --no-cuda
BUILD_NO_CUDA = False
if "--no-cuda" in sys.argv:
    BUILD_NO_CUDA = True
    sys.argv.remove("--no-cuda")  # Remove so setuptools doesn't complain
    print("\n" + "=" * 80)
    print("CUDA backend disabled (--no-cuda flag)")
    print("=" * 80 + "\n")

# HIP is automatic on a ROCm-only build. --hip also adds it to a CUDA build and
# turns a missing compiler into an error; --no-hip suppresses it.
BUILD_HIP = os.getenv("COMFY_KITCHEN_BUILD_HIP") == "1"
if "--hip" in sys.argv:
    BUILD_HIP = True
    sys.argv.remove("--hip")

BUILD_NO_HIP = os.getenv("COMFY_KITCHEN_BUILD_NO_HIP") == "1"
if "--no-hip" in sys.argv:
    BUILD_NO_HIP = True
    sys.argv.remove("--no-hip")

# build_ext parses --hip-archs itself, but the extension list is built before its
# options are finalized, so the value has to be read here too. Left in argv for it.
HIP_ARCHS_CLI = ""
for _i, _arg in enumerate(sys.argv):
    if _arg.startswith("--hip-archs="):
        HIP_ARCHS_CLI = _arg.split("=", 1)[1]
    elif _arg == "--hip-archs" and _i + 1 < len(sys.argv):
        HIP_ARCHS_CLI = sys.argv[_i + 1]



def cmake_path(path: str | os.PathLike[str]) -> str:
    """Return a CMake-safe path with forward slashes on every platform."""
    return os.fspath(path).replace("\\", "/")


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = "", backend: str = "cuda",
                 hip_archs: str = ""):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir) if source_dir else ""
        self.backend = backend
        self.hip_archs = hip_archs


class CMakeBuildExt(build_ext):
    # Add custom command-line options
    user_options: ClassVar = [
        *build_ext.user_options,
        ('cuda-archs=', None, 'CUDA architectures to build for (semicolon-separated, e.g., "80;89;90a")'),
        ('hip-archs=', None, 'HIP architectures to build for (semicolon-separated, e.g., "gfx1200;gfx1201")'),
        ('debug-build', None, 'Build in debug mode with debug symbols'),
        ('lineinfo', None, 'Enable NVCC line information for profiling (adds -lineinfo flag)'),
    ]

    # Default values for options
    # Native cubins cover every named GPU. Keep compute_89 as the generic PTX
    # fallback and the two Blackwell family PTX images for future family members.
    # Older generic PTX and compute_90a are redundant when their cubins are present.
    DEFAULT_CUDA_ARCHS_WINDOWS = "75-real;80-real;89;120f"
    DEFAULT_CUDA_ARCHS_LINUX = "75-real;80-real;89;90a-real;100f;120f"

    def initialize_options(self):
        super().initialize_options()
        # Set defaults - can be overridden by command-line arguments
        self.cuda_archs = None  # Will use platform-specific default in finalize_options
        self.hip_archs = None  # None lets the HIP CMakeLists pick its default gfx list
        self.debug_build = False  # Default: Release build
        self.lineinfo = False  # Default: disabled

    def finalize_options(self):
        super().finalize_options()

        # An environment override also reaches build_ext when another command
        # (such as bdist_wheel) creates it internally.
        if self.cuda_archs is None:
            self.cuda_archs = os.environ.get("COMFY_CUDA_ARCHS") or (
                self.DEFAULT_CUDA_ARCHS_WINDOWS
                if os.name == "nt"
                else self.DEFAULT_CUDA_ARCHS_LINUX
            )


    def run(self):
        try:
            subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("CMake must be installed to build this package") from e

        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        regular_extensions = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]

        for ext in cmake_extensions:
            self.build_cmake(ext)

        if regular_extensions:
            original_extensions = self.extensions
            self.extensions = regular_extensions
            super().run()
            self.extensions = original_extensions

    def build_cmake(self, ext: CMakeExtension):
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        ext_dir = ext_fullpath.parent
        ext_dir.mkdir(parents=True, exist_ok=True)

        # Each backend gets its own build directory: the CUDA and HIP extensions
        # share self.build_temp, and CMake refuses to reuse a cache generated for a
        # different source dir ("does not match the source ... used to generate
        # cache"), so configuring the second one into the first one's directory fails.
        build_temp = pathlib.Path(self.build_temp).resolve() / ext.backend
        build_temp.mkdir(parents=True, exist_ok=True)

        # All options have been set in finalize_options with proper defaults
        config = "Debug" if self.debug_build else "Release"
        cuda_archs = self.cuda_archs
        enable_lineinfo = self.lineinfo

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={cmake_path(ext_dir)}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython_EXECUTABLE={cmake_path(sys.executable)}",
            f"-DCOMFY_ENABLE_LINEINFO={'ON' if enable_lineinfo else 'OFF'}",
        ]

        if ext.backend == "hip":
            # CMake's default generator on Windows is Visual Studio, which does
            # not support the HIP language.
            cmake_args.extend(["-G", "Ninja"])

            rocm_home, hip_compiler = get_rocm_path()
            if hip_compiler is None:
                raise RuntimeError(
                    "HIP extension build requested, but no ROCm compiler could be found. "
                    "Set ROCM_HOME to a valid ROCm install or build with --no-hip."
                )
            cmake_args.append(f"-DCMAKE_HIP_COMPILER={hip_compiler.as_posix()}")

            # CMake refuses to mix a GNU-like clang with a CL-compatible C/C++
            # compiler, and would otherwise pick MSVC for C/C++ while using clang
            # for HIP. Pin all three languages to the ROCm driver. The C driver
            # drops the ++ (clang++ -> clang, amdclang++ -> amdclang); hipcc
            # compiles both.
            c_name = hip_compiler.stem.removesuffix("++")
            c_compiler = hip_compiler.with_name(c_name + hip_compiler.suffix)
            if c_compiler.is_file():
                cmake_args.append(f"-DCMAKE_C_COMPILER={c_compiler.as_posix()}")
            cmake_args.append(f"-DCMAKE_CXX_COMPILER={hip_compiler.as_posix()}")

            # Enabling CXX pulls in the RC language, and CMake looks for the
            # resource compiler on PATH alone. ROCm ships neither rc nor llvm-rc,
            # so without this the configure dies in project() on any machine that
            # is not a Visual Studio developer prompt. An explicit RC wins.
            if os.name == "nt" and not os.environ.get("RC"):
                rc_compiler = find_rc_compiler(hip_compiler)
                if rc_compiler:
                    cmake_args.append(f"-DCMAKE_RC_COMPILER={rc_compiler.as_posix()}")

            if rocm_home:
                rocm_posix = pathlib.Path(rocm_home).as_posix()
                cmake_args.append(f"-DCMAKE_PREFIX_PATH={rocm_posix}")
                cmake_args.append(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rocm_posix}")

            # --hip-archs beats the environment, which beats what setup_hip_extension
            # resolved from the visible devices. The CLI value is raw, so normalize it
            # the way ext.hip_archs already was: CMake splits its arch list on ";", and
            # an unnormalized "gfx1100,gfx1200" would reach it as a single bad target.
            cli_archs = ";".join(normalize_archs(self.hip_archs)) if self.hip_archs else ""
            hip_archs = cli_archs or ext.hip_archs
            if hip_archs:
                cmake_args.append(f"-DCOMFY_HIP_ARCHS={hip_archs}")

            # HIP is the same clang++ driver as CXX here, so the C++ launcher
            # covers both unless a HIP-specific one is set.
            hip_launcher = os.environ.get("COMFY_HIP_COMPILER_LAUNCHER")
            if hip_launcher:
                cmake_args.append(f"-DCOMFY_HIP_COMPILER_LAUNCHER={hip_launcher}")
            cxx_launcher = os.environ.get("COMFY_CXX_COMPILER_LAUNCHER")
            if cxx_launcher:
                cmake_args.append(f"-DCOMFY_CXX_COMPILER_LAUNCHER={cxx_launcher}")
        else:
            cmake_args.append(f"-DCOMFY_CUDA_ARCHS={cuda_archs}")

            # Let CMake manage its own configuration cache. Reconfiguring with the
            # explicit arguments above updates changed settings without throwing
            # away cached compiler checks and the generated build graph.
            generator = os.environ.get("CMAKE_GENERATOR")
            if generator:
                cmake_args.extend(["-G", generator])

            # Compiler caching is opt-in. Pass project-specific variables so CMake
            # enables the launchers after compiler identification; wrapping the
            # identification probes is unreliable with NVCC + MSVC on Windows.
            cuda_launcher = os.environ.get("COMFY_CUDA_COMPILER_LAUNCHER")
            if cuda_launcher:
                cmake_args.append(f"-DCOMFY_CUDA_COMPILER_LAUNCHER={cuda_launcher}")
            cxx_launcher = os.environ.get("COMFY_CXX_COMPILER_LAUNCHER")
            if cxx_launcher:
                cmake_args.append(f"-DCOMFY_CXX_COMPILER_LAUNCHER={cxx_launcher}")

            cuda_paths = get_cuda_path()
            if cuda_paths is None:
                raise RuntimeError(
                    "CUDA extension build requested, but nvcc could not be found. "
                    "Set CUDA_HOME to a valid CUDA toolkit or build with --no-cuda."
                )
            cuda_home, nvcc_bin = cuda_paths
            cmake_args.append(f"-DCUDAToolkit_ROOT={cmake_path(cuda_home)}")
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cmake_path(nvcc_bin)}")

            # FindCUDAToolkit only learned the Windows ARM64 library layout in
            # CMake 4.4. Help older CMake releases find cudart under lib/arm64;
            # once CUDA_CUDART is known, the module uses its directory for the
            # remaining CUDA imported targets as well.
            if os.name == "nt" and platform.machine().lower() in {"arm64", "aarch64"}:
                arm64_cudart = pathlib.Path(cuda_home) / "lib" / "arm64" / "cudart.lib"
                if not arm64_cudart.is_file():
                    raise RuntimeError(
                        f"Windows ARM64 CUDA runtime library not found: {arm64_cudart}"
                    )
                cmake_args.append(f"-DCUDA_CUDART={cmake_path(arm64_cudart)}")
                cmake_args.append("-DCOMFY_MSVC_PERMISSIVE=ON")

        build_args = ["--config", config]

        max_jobs = os.cpu_count() or 1
        build_args.extend(["--parallel", str(max_jobs)])

        # Run CMake configure
        source_dir = cmake_path(ext.source_dir if ext.source_dir else os.path.dirname(os.path.abspath(__file__)))

        print(f"Configuring CMake for {ext.name}...")
        print(f"  Source directory: {source_dir}")
        print(f"  Build directory: {build_temp}")
        print(f"  Config: {config}")
        print(f"  CUDA architectures: {cuda_archs}")
        print(f"  Line info: {'enabled' if enable_lineinfo else 'disabled'}")

        configure_cmd = ["cmake", source_dir, *cmake_args]
        try:
            subprocess.run(
                configure_cmd,
                cwd=build_temp,
                check=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake configuration failed for {ext.name}") from e

        # Run CMake build
        print(f"Building {ext.name} with CMake...")
        build_cmd = ["cmake", "--build", ".", *build_args]
        try:
            subprocess.run(
                build_cmd,
                cwd=build_temp,
                check=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake build failed for {ext.name}") from e

        print(f"Successfully built {ext.name}")

def get_cuda_path() -> tuple[pathlib.Path, pathlib.Path] | None:
    nvcc_bin = None
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        nvcc_bin = pathlib.Path(cuda_home) / "bin" / "nvcc"

    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            nvcc_bin = pathlib.Path(nvcc_path)

    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_bin = pathlib.Path("/usr/local/cuda/bin/nvcc")

    if not nvcc_bin.is_file():
        return None

    if cuda_home is None:
        cuda_home = nvcc_bin.parent.parent

    return pathlib.Path(cuda_home), nvcc_bin

# Keep build-time, CMake, and runtime architecture policy in one package resource.
# Exact membership is intentional: accepting an unreviewed gfx11xx/gfx12xx target
# can compile the no-WMMA trap stubs into an otherwise successful wheel.
HIP_ARCH_GROUP_NAMES = ("elementwise_only", "wmma_gfx11", "wmma_gfx117", "wmma_gfx12")
HIP_ARCH_MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "comfy_kitchen"
    / "backends"
    / "hip"
    / "architectures.json"
)
HIP_ARCH_GROUPS = json.loads(HIP_ARCH_MANIFEST_PATH.read_text(encoding="utf-8"))
if tuple(HIP_ARCH_GROUPS) != HIP_ARCH_GROUP_NAMES:
    raise RuntimeError(
        f"{HIP_ARCH_MANIFEST_PATH} must contain these groups in order: "
        f"{', '.join(HIP_ARCH_GROUP_NAMES)}"
    )

SUPPORTED_HIP_ARCHS = tuple(
    arch
    for group_name in HIP_ARCH_GROUP_NAMES
    for arch in HIP_ARCH_GROUPS[group_name]
)
if not SUPPORTED_HIP_ARCHS or len(SUPPORTED_HIP_ARCHS) != len(set(SUPPORTED_HIP_ARCHS)):
    raise RuntimeError(f"{HIP_ARCH_MANIFEST_PATH} is empty or contains duplicate targets")

DEFAULT_HIP_ARCHS = ";".join(SUPPORTED_HIP_ARCHS)


def hip_arch_supported(arch: str) -> bool:
    return arch in SUPPORTED_HIP_ARCHS


def normalize_archs(value: str) -> list[str]:
    """Split an arch list on , or ; and drop the :xnack+/-like feature suffixes."""
    archs = []
    for part in value.replace(";", ",").split(","):
        arch = part.strip().split(":", 1)[0]
        if arch and arch not in archs:
            archs.append(arch)
    return archs


def get_hip_archs_override() -> list[str]:
    if HIP_ARCHS_CLI:
        return normalize_archs(HIP_ARCHS_CLI)
    # PYTORCH_ROCM_ARCH and GPU_ARCHS are the conventional ROCm spellings.
    for var in ("COMFY_HIP_ARCHS", "PYTORCH_ROCM_ARCH", "GPU_ARCHS"):
        value = os.getenv(var)
        if value:
            return normalize_archs(value)
    return []


def detect_hip_archs() -> list[str]:
    """gfx names of the visible AMD devices, empty when none can be enumerated."""
    # PyTorch is intentionally not a build dependency. Defer this optional,
    # heavyweight import so ordinary metadata and CUDA-only builds do not load it.
    try:
        torch = importlib.import_module("torch")
        if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
            return []
        names = [
            torch.cuda.get_device_properties(i).gcnArchName
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []
    return normalize_archs(";".join(n for n in names if n))


def get_torch_gpu_runtime() -> str | None:
    """Return the PyTorch GPU runtime when PyTorch is available to the build.

    PyTorch is absent from an isolated build environment, so this is a guard
    against selecting a useless HIP-only extension under an installed CUDA
    PyTorch rather than a requirement for all builds.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return None


def rocm_sdk_root() -> str | None:
    """Ask the pip rocm-sdk wheel for its root, if it is installed."""
    try:
        root = subprocess.run(
            [sys.executable, "-m", "rocm_sdk", "path", "--root"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    return root if root and pathlib.Path(root).exists() else None


def find_rc_compiler(hip_compiler: pathlib.Path) -> pathlib.Path | None:
    """Locate a Windows resource compiler, newest Windows SDK last.

    CMake searches PATH for rc then llvm-rc, which finds neither outside a
    developer prompt. The SDK is already a build requirement: clang links
    against it, and its rc.exe sits in a directory nothing puts on PATH.
    """
    beside = hip_compiler.with_name("llvm-rc.exe")
    if beside.is_file():
        return beside

    found = shutil.which("rc") or shutil.which("llvm-rc")
    if found:
        return pathlib.Path(found)

    host = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    candidates = []
    for var in ("ProgramFiles(x86)", "ProgramFiles"):
        program_files = os.environ.get(var)
        if program_files:
            base = pathlib.Path(program_files) / "Windows Kits" / "10" / "bin"
            candidates.extend(base.glob(f"*/{host}/rc.exe"))
    if not candidates:
        return None

    def sdk_version(path: pathlib.Path) -> list[int]:
        return [int(p) if p.isdigit() else 0 for p in path.parent.parent.name.split(".")]

    return max(candidates, key=sdk_version)


def get_rocm_path() -> tuple[str | None, pathlib.Path | None]:
    """Locate a ROCm root and its clang driver.

    Handles the pip ``rocm-sdk`` layout (site-packages/_rocm_sdk_devel) as well
    as a system ROCm install. In precedence order: an explicit ROCM_HOME or
    ROCM_PATH, the SDK of the interpreter running the build, then a system
    install found through HIP_PATH, /opt/rocm or PATH.
    """
    # An explicit ROCM_HOME/ROCM_PATH wins, but only while it still points at a
    # directory, so a stale value does not shadow an installed toolchain.
    rocm_home = None
    for var in ("ROCM_HOME", "ROCM_PATH"):
        value = os.getenv(var)
        if value and pathlib.Path(value).is_dir():
            rocm_home = value
            break

    if rocm_home is None:
        rocm_home = rocm_sdk_root()

    if rocm_home is None:
        sdk = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "_rocm_sdk_devel"
        if not sdk.exists():
            sdk = (
                pathlib.Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "_rocm_sdk_devel"
            )
        if sdk.exists():
            rocm_home = str(sdk)

    if rocm_home is None:
        # The Windows HIP SDK installer sets HIP_PATH itself, so it is not the
        # deliberate choice ROCM_HOME is and ranks below the SDK installed in the
        # interpreter running the build.
        hip_path = os.getenv("HIP_PATH")
        if hip_path and pathlib.Path(hip_path).is_dir():
            rocm_home = hip_path

    if rocm_home is None and pathlib.Path("/opt/rocm").exists():
        rocm_home = "/opt/rocm"

    compiler = None
    if rocm_home:
        root = pathlib.Path(rocm_home)
        for candidate in (
            root / "lib" / "llvm" / "bin" / "clang++",
            root / "lib" / "llvm" / "bin" / "clang++.exe",
            root / "bin" / "amdclang++",
            # The Windows HIP SDK ships its driver as %HIP_PATH%\bin\clang++.exe.
            root / "bin" / "clang++.exe",
            root / "bin" / "hipcc",
        ):
            if candidate.is_file():
                compiler = candidate
                break

    if compiler is None:
        for name in ("amdclang++", "hipcc"):
            found = shutil.which(name)
            if found:
                compiler = pathlib.Path(found)
                if rocm_home is None:
                    rocm_home = str(compiler.parent.parent)
                break

    if compiler is None and os.name == "nt":
        # A HIP SDK whose bin is on PATH without HIP_PATH set is reachable only
        # through its plain clang++, a name unrelated LLVM installs answer to as
        # well. Walk PATH rather than taking shutil.which's first hit, and keep
        # the entry that has the SDK headers beside it.
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            found = shutil.which("clang++", path=entry) if entry else None
            if found is None:
                continue
            root = pathlib.Path(found).parent.parent
            if (root / "include" / "hip" / "hip_runtime.h").is_file():
                compiler = pathlib.Path(found)
                if rocm_home is None:
                    rocm_home = str(root)
                break

    return rocm_home, compiler


def setup_hip_extension() -> CMakeExtension | None:
    print("=" * 80)
    print("Checking for HIP/ROCm availability...")
    print("=" * 80)

    if BUILD_NO_HIP:
        print("HIP extension disabled by --no-hip flag")
        return None

    rocm_home, hip_compiler = get_rocm_path()
    if hip_compiler is None:
        if BUILD_HIP:
            raise RuntimeError(
                "ERROR: --hip requested but no ROCm compiler was found "
                "(looked for clang++/amdclang++/hipcc). Install ROCm or the rocm-sdk wheel."
            )
        print("No ROCm compiler detected; skipping HIP backend")
        return None

    print(f"Found ROCm root: {rocm_home or 'auto'}")
    print(f"Found HIP compiler: {hip_compiler}")

    # RDNA2 has no matrix cores, so it gets the elementwise kernels only; the GEMMs
    # need the gfx11 or gfx12 WMMA intrinsics. Everything below RDNA2 (and CDNA,
    # which uses MFMA rather than WMMA) has no path through these sources at all.
    archs = get_hip_archs_override()
    if archs:
        unsupported = [arch for arch in archs if not hip_arch_supported(arch)]
        if unsupported:
            raise RuntimeError(
                f"ERROR: unsupported HIP architecture target(s): {';'.join(unsupported)}. "
                f"Validated targets: {';'.join(SUPPORTED_HIP_ARCHS)}"
            )
        print(f"HIP architectures from the override: {';'.join(archs)}")
    else:
        detected = detect_hip_archs()
        if detected:
            archs = [arch for arch in detected if hip_arch_supported(arch)]
            if not archs:
                message = (
                    f"Visible AMD GPUs ({';'.join(detected)}) are not RDNA2/3/4; "
                    "these kernels would not run on them."
                )
                if BUILD_HIP:
                    raise RuntimeError(f"ERROR: --hip requested but {message}")
                print(f"{message} Skipping the HIP backend.")
                print("Set COMFY_HIP_ARCHS to build for a target anyway.")
                return None
            print(f"Detected supported devices: {';'.join(archs)}")
        else:
            archs = normalize_archs(DEFAULT_HIP_ARCHS)
            print(f"No AMD GPU visible; building for the default {';'.join(archs)}")

    root_dir = pathlib.Path(__file__).resolve().parent
    hip_backend_dir = root_dir / "comfy_kitchen" / "backends" / "hip"
    if not hip_backend_dir.exists():
        raise RuntimeError(f"HIP backend directory not found: {hip_backend_dir}")

    print("Building HIP extension with CMake + nanobind: comfy_kitchen.backends.hip._C")
    return CMakeExtension(
        name="comfy_kitchen.backends.hip._C",
        source_dir=str(hip_backend_dir),
        backend="hip",
        hip_archs=";".join(archs),
    )


def get_cuda_version() -> tuple[int, ...] | None:
    # get_cuda_path() returns None rather than a pair when nvcc is absent.
    cuda_paths = get_cuda_path()
    if cuda_paths is None:
        return None

    _cuda_home, nvcc_bin = cuda_paths
    # A toolkit was found: absence is get_cuda_path()'s None above. A present but
    # broken nvcc (unrunnable, or a failing -V) is a real error and must not be
    # laundered into "no CUDA", which would silently ship a HIP-only wheel.
    try:
        output = subprocess.run(
            [nvcc_bin, "-V"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"CUDA toolkit was found, but `{nvcc_bin} -V` failed") from exc

    match = re.search(r"release\s*([\d.]+)", output.stdout)
    if not match:
        return None

    version = tuple(map(int, match.group(1).split(".")))
    return version


class CudaToolkitNotFoundError(RuntimeError):
    """No CUDA toolkit on this machine.

    The only CUDA failure a ROCm box is allowed to shrug off. Everything else (a
    toolkit too old to use, a missing backend directory, a broken nanobind) means
    a CUDA build was intended and went wrong, and must not silently degrade the
    combined wheel to HIP-only.
    """


def assert_cuda_version(version: tuple[int, ...]) -> None:
    lowest_cuda_version = (12, 8)
    if version < lowest_cuda_version:
        raise RuntimeError(
            f"ComfyKitchen CUDA backend requires CUDA {lowest_cuda_version} or newer. "
            f"Got {version}. Install will continue without CUDA backend."
        )


def setup_cuda_extension() -> CMakeExtension | None:
    print("=" * 80)
    print("Checking for CUDA availability...")
    print("=" * 80)

    if BUILD_NO_CUDA:
        print("CUDA extension disabled by --no-cuda flag")
        return None

    try:
        import nanobind  # noqa: F401
    except ImportError as e:
        raise ImportError("ERROR: nanobind not found. Install with: pip install nanobind") from e

    cuda_version = get_cuda_version()
    if cuda_version is None:
        raise CudaToolkitNotFoundError(
            "ERROR: Could not detect CUDA toolkit (nvcc not found). Install CUDA toolkit and try again."
        )

    print(f"Found CUDA version: {'.'.join(map(str, cuda_version))}")

    try:
        assert_cuda_version(cuda_version)
    except RuntimeError as e:
        raise RuntimeError(f"ERROR: {e}") from e

    root_dir = pathlib.Path(__file__).resolve().parent
    cuda_backend_dir = root_dir / "comfy_kitchen" / "backends" / "cuda"

    if not cuda_backend_dir.exists():
        raise RuntimeError(f"WARNING: CUDA backend directory not found: {cuda_backend_dir}")

    print("Building CUDA extension with CMake + nanobind: comfy_kitchen.backends.cuda._C")

    # Create CMake extension pointing to the CUDA backend directory
    ext_module = CMakeExtension(
        name="comfy_kitchen.backends.cuda._C",
        source_dir=str(cuda_backend_dir),
    )

    print("CUDA extension configured successfully (will be built with CMake)")
    return ext_module


def get_extensions() -> list[setuptools.Extension]:
    extensions = []

    if not BUILD_NO_CUDA:
        try:
            cuda_ext = setup_cuda_extension()
            if cuda_ext is not None:
                extensions.append(cuda_ext)
        except CudaToolkitNotFoundError as e:
            # An absent toolkit is survivable on a ROCm host, but must not turn an
            # NVIDIA source build into a successfully installed HIP-only package.
            _rocm_home, hip_compiler = get_rocm_path()
            if hip_compiler is None or BUILD_NO_HIP:
                raise
            if get_torch_gpu_runtime() == "cuda" and not BUILD_HIP:
                raise CudaToolkitNotFoundError(
                    f"{e} A ROCm compiler was also found, but CUDA PyTorch is installed; "
                    "refusing to replace the missing CUDA backend with a HIP-only build. "
                    "Install nvcc, use --no-cuda for a Python-only build, or explicitly "
                    "request a cross-build with --hip."
                ) from e
            print(f"\nNo CUDA toolkit ({e})")
            print("A ROCm toolchain was found, so building the HIP backend instead.\n")

    # A CUDA source build stays CUDA-only unless HIP was explicitly requested.
    # This prevents an incidental ROCm SDK from multiplying build time and failure
    # surface on NVIDIA workstations. ROCm-only hosts still auto-select HIP after
    # the missing-CUDA path above, and combined-wheel CI opts in explicitly.
    build_hip_extension = BUILD_HIP or (not BUILD_NO_CUDA and not extensions)
    if build_hip_extension:
        hip_ext = setup_hip_extension()
        if hip_ext is not None:
            extensions.append(hip_ext)

    if not extensions:
        print("\n" + "=" * 80)
        print("Installing comfy_kitchen without a native backend")
        print("Available backends: eager, triton (if installed)")
        print("=" * 80 + "\n")

    return extensions


def get_cmdclass(has_extensions):
    cmdclass = {}

    if has_extensions:
        cmdclass["build_ext"] = CMakeBuildExt

    try:
        from wheel.bdist_wheel import bdist_wheel

        class ComfyBdistWheel(bdist_wheel):
            def finalize_options(self):
                super().finalize_options()
                # Stable ABI on 3.12+ only (nanobind's floor); 3.10/3.11 stay
                # version-specific. Both CMakeLists build against the limited API
                # there, so a wheel carrying either or both keeps the abi3 tag and
                # the HIP backend does not multiply the wheel matrix.
                if has_extensions and sys.version_info >= (3, 12):
                    self.py_limited_api = "cp312"

        cmdclass["bdist_wheel"] = ComfyBdistWheel
    except ImportError as e:
        print(f"Warning: Could not import wheel.bdist_wheel: {e}")

    return cmdclass


def get_packages():
    if BUILD_NO_CUDA:
        cuda_dir = pathlib.Path("comfy_kitchen/backends/cuda")
        cuda_backup = pathlib.Path("cuda_backup_temp_build")

        if cuda_dir.exists():
            shutil.move(str(cuda_dir), str(cuda_backup))

        try:
            all_packages = setuptools.find_packages(where=".")
            packages = [pkg for pkg in all_packages if not pkg.startswith(("tests", "cuda_backup"))]
            return packages
        finally:
            if cuda_backup.exists():
                shutil.move(str(cuda_backup), str(cuda_dir))

    return setuptools.find_packages(where=".", exclude=["tests*"])


extensions = get_extensions()

setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": get_cmdclass(has_extensions=bool(extensions)),
}

if BUILD_NO_CUDA and not extensions:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    project_meta = pyproject.get("project", {})
    version = project_meta["version"]
    description = project_meta.get("description", "")

    setup_kwargs.update({
        "packages": get_packages(),
        "name": "comfy-kitchen",
        "version": version,
        "description": f"{description} (CPU-only)",
        "include_package_data": False,
        "install_requires": [
            "torch>=2.5.0",
        ],
    })

    readme_path = pathlib.Path("README.md")
    if readme_path.exists():
        setup_kwargs.update({
            "long_description": readme_path.read_text(),
            "long_description_content_type": "text/markdown",
        })

setuptools.setup(**setup_kwargs)
