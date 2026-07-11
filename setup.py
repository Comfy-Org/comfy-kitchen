import os
import pathlib
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
#    or: pip install . --no-cuda
BUILD_NO_CUDA = False
if "--no-cuda" in sys.argv:
    BUILD_NO_CUDA = True
    sys.argv.remove("--no-cuda")  # Remove so setuptools doesn't complain
    print("\n" + "=" * 80)
    print("Building CPU-only variant (--no-cuda flag)")
    print("CUDA backend excluded - only eager, triton backends")
    print("=" * 80 + "\n")

# The HIP backend is built whenever a ROCm compiler is present. --hip turns a
# missing compiler into an error rather than a skip; --no-hip suppresses it.
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
for _arg in sys.argv:
    if _arg.startswith("--hip-archs="):
        HIP_ARCHS_CLI = _arg.split("=", 1)[1]


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
    DEFAULT_CUDA_ARCHS_WINDOWS = "75-real;75-virtual;80;89;120f"  # No need for Datacenter GPUs
    DEFAULT_CUDA_ARCHS_LINUX = "75-real;75-virtual;80;89;90a;100f;120f"  # + H100, B100

    def initialize_options(self):
        super().initialize_options()
        # Set defaults - can be overridden by command-line arguments
        self.cuda_archs = None  # Will use platform-specific default in finalize_options
        self.hip_archs = None  # None lets the HIP CMakeLists pick its gfx12 default
        self.debug_build = False  # Default: Release build
        self.lineinfo = False  # Default: disabled

    def finalize_options(self):
        super().finalize_options()

        # Apply platform-specific default for CUDA architectures if not specified
        if self.cuda_archs is None:
            self.cuda_archs = (
                self.DEFAULT_CUDA_ARCHS_WINDOWS if os.name == "nt"
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

        build_temp = pathlib.Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Clean CMake cache if it exists (to avoid stale configuration)
        cmake_cache = build_temp / "CMakeCache.txt"
        if cmake_cache.exists():
            cmake_cache.unlink()
            print(f"Cleaned stale CMake cache: {cmake_cache}")

        # All options have been set in finalize_options with proper defaults
        config = "Debug" if self.debug_build else "Release"
        cuda_archs = self.cuda_archs
        enable_lineinfo = self.lineinfo

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCOMFY_ENABLE_LINEINFO={'ON' if enable_lineinfo else 'OFF'}",
        ]

        if ext.backend == "hip":
            # CMake's default generator on Windows is Visual Studio, which does
            # not support the HIP language.
            cmake_args.extend(["-G", "Ninja"])

            rocm_home, hip_compiler = get_rocm_path()
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

            if rocm_home:
                rocm_posix = pathlib.Path(rocm_home).as_posix()
                cmake_args.append(f"-DCMAKE_PREFIX_PATH={rocm_posix}")
                cmake_args.append(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rocm_posix}")

            # --hip-archs beats the environment, which beats what setup_hip_extension
            # resolved from the visible devices.
            hip_archs = self.hip_archs or ext.hip_archs
            if hip_archs:
                cmake_args.append(f"-DCOMFY_HIP_ARCHS={hip_archs}")
        else:
            cmake_args.append(f"-DCOMFY_CUDA_ARCHS={cuda_archs}")
            cuda_home, nvcc_bin = get_cuda_path()
            cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_home}")
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_bin}")

        build_args = ["--config", config]

        max_jobs = os.cpu_count() or 1
        # Use appropriate parallel build syntax for the generator
        if os.name == "nt" and ext.backend != "hip":
            # Windows MSBuild uses /m:N for parallel builds
            build_args.extend(["--", f"/m:{max_jobs}"])
        else:
            # Ninja and Unix make both take -jN
            build_args.extend(["--", f"-j{max_jobs}"])

        # Run CMake configure
        source_dir = ext.source_dir if ext.source_dir else os.path.dirname(os.path.abspath(__file__))

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

def get_cuda_path():
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
        cuda_home = str(nvcc_bin.parent.parent)

    return cuda_home, nvcc_bin

DEFAULT_HIP_ARCHS = "gfx1200;gfx1201"


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
    try:
        import torch

        if torch.version.hip is None or not torch.cuda.is_available():
            return []
        names = [
            torch.cuda.get_device_properties(i).gcnArchName
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []
    return normalize_archs(";".join(n for n in names if n))


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


def get_rocm_path() -> tuple[str | None, pathlib.Path | None]:
    """Locate a ROCm root and its clang driver.

    Handles the pip ``rocm-sdk`` layout (site-packages/_rocm_sdk_devel) as well
    as a system ROCm install.
    """
    rocm_home = os.getenv("ROCM_HOME") or os.getenv("ROCM_PATH") or os.getenv("HIP_PATH")

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

    if rocm_home is None and pathlib.Path("/opt/rocm").exists():
        rocm_home = "/opt/rocm"

    compiler = None
    if rocm_home:
        root = pathlib.Path(rocm_home)
        for candidate in (
            root / "lib" / "llvm" / "bin" / "clang++",
            root / "lib" / "llvm" / "bin" / "clang++.exe",
            root / "bin" / "amdclang++",
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

    # The kernels use gfx12-only WMMA intrinsics, so an arch list that names any
    # other target does not compile. An explicit override is taken as given; a
    # detected non-gfx12 GPU means there is nothing worth building here.
    archs = get_hip_archs_override()
    if archs:
        # An override skips the device gate below, so it is the only other place a
        # non-gfx12 target can get through to the gfx12-only sources.
        unsupported = [arch for arch in archs if not arch.startswith("gfx12")]
        if unsupported:
            raise RuntimeError(
                f"ERROR: the HIP backend's WMMA kernels are gfx12-only (RDNA4); "
                f"cannot build for {';'.join(unsupported)}"
            )
        print(f"HIP architectures from the override: {';'.join(archs)}")
    else:
        detected = detect_hip_archs()
        if detected:
            archs = [arch for arch in detected if arch.startswith("gfx12")]
            if not archs:
                message = (
                    f"Visible AMD GPUs ({';'.join(detected)}) are not gfx12/RDNA4; "
                    "the WMMA kernels would not run on them."
                )
                if BUILD_HIP:
                    raise RuntimeError(f"ERROR: --hip requested but {message}")
                print(f"{message} Skipping the HIP backend.")
                print("Set COMFY_HIP_ARCHS to build for a target anyway.")
                return None
            print(f"Detected gfx12 devices: {';'.join(archs)}")
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
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None

    _cuda_home, nvcc_bin = cuda_path
    try:
        output = subprocess.run(
            [nvcc_bin, "-V"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None

    match = re.search(r"release\s*([\d.]+)", output.stdout)
    if not match:
        return None

    version = tuple(map(int, match.group(1).split(".")))
    return version


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
        raise RuntimeError("ERROR: Could not detect CUDA toolkit (nvcc not found). Install CUDA toolkit and try again.")

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
        except RuntimeError as e:
            # A machine without nvcc is only fatal when there is no ROCm
            # toolchain to fall back on.
            _rocm_home, hip_compiler = get_rocm_path()
            if hip_compiler is None or BUILD_NO_HIP:
                raise
            print(f"\nCUDA toolkit not usable ({e})")
            print("A ROCm toolchain was found, so building the HIP backend instead.\n")

    hip_ext = setup_hip_extension()
    if hip_ext is not None:
        extensions.append(hip_ext)

    if not extensions:
        print("\n" + "=" * 80)
        print("Installing comfy_kitchen without a native backend")
        print("Available backends: eager, triton (if installed)")
        print("=" * 80 + "\n")

    return extensions


def get_cmdclass(has_extensions, has_hip_extension=False):
    cmdclass = {}

    if has_extensions:
        cmdclass["build_ext"] = CMakeBuildExt

    try:
        from wheel.bdist_wheel import bdist_wheel

        class CUDABdistWheel(bdist_wheel):
            def finalize_options(self):
                super().finalize_options()
                # Set stable ABI tag only for Python 3.12+ (nanobind requirement)
                # For 3.10/3.11, leave as version-specific (cpXXX-cpXXX)
                # The HIP extension is not built against the stable ABI, so a
                # wheel containing it must stay version-specific.
                if not BUILD_NO_CUDA and not has_hip_extension and sys.version_info >= (3, 12):
                    self.py_limited_api = "cp312"

        cmdclass["bdist_wheel"] = CUDABdistWheel
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

has_hip_extension = any(
    isinstance(ext, CMakeExtension) and ext.backend == "hip" for ext in extensions
)

setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": get_cmdclass(
        has_extensions=bool(extensions),
        has_hip_extension=has_hip_extension,
    ),
}

if BUILD_NO_CUDA:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    project_meta = pyproject.get("project", {})
    version = project_meta.get("version", "0.1.0")
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
