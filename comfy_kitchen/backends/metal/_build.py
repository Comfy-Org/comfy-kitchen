# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile Metal shaders + Torch MPS ObjC++ extension for the metal backend."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

_logger = logging.getLogger("comfy_kitchen.metal")

_PACKAGE_DIR = Path(__file__).resolve().parent
_CSRC = _PACKAGE_DIR / "csrc"
_BUILD = _PACKAGE_DIR / "build"
# kernels / fused_quant: Metal 3.1. mxfp8_matmul: TensorOps (toolchain default lang).
_METAL_SOURCES = (
    ("kernels.metal", ["-std=metal3.1"]),
    ("fused_quant_mm.metal", ["-std=metal3.1"]),
    ("convrot_fused.metal", ["-std=metal3.1"]),
    ("mxfp8_matmul.metal", []),
)
_METALLIB_NAME = "comfy_kitchen_metal.metallib"
_EXT_NAME = "comfy_kitchen_metal_ops"
_MPP_INCLUDE = (
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/"
    "MetalPerformancePrimitives.framework/Headers"
)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _find_metal_bin() -> str | None:
    """Locate ``metal`` / ``metallib``, including Xcode 27 MetalToolchain cryptex."""
    override = os.environ.get("COMFY_KITCHEN_METAL_BIN")
    if override and Path(override).is_file():
        return override

    try:
        out = subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "-f", "metal"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out and Path(out).is_file():
            # Probe that it actually runs (missing toolchain stubs fail here).
            probe = subprocess.run(
                [out, "-v"], capture_output=True, text=True, timeout=10
            )
            if probe.returncode == 0:
                return out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass

    search_roots = [
        Path("/Volumes/MetalToolchainCryptex"),
        Path("/System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for cand in root.rglob("bin/metal"):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return None


def _metal_env(metal_bin: str) -> dict[str, str]:
    env = os.environ.copy()
    # Prefer the toolchain that owns this metal binary for metallib too.
    bin_dir = str(Path(metal_bin).resolve().parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def compile_metallib(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or _BUILD
    out_dir.mkdir(parents=True, exist_ok=True)
    metal_bin = _find_metal_bin()
    if metal_bin is None:
        raise RuntimeError(
            "Metal compiler not found. Install Xcode Metal Toolchain "
            "(xcodebuild -downloadComponent MetalToolchain) or set COMFY_KITCHEN_METAL_BIN."
        )
    env = _metal_env(metal_bin)
    metallib_bin = str(Path(metal_bin).parent / "metallib")
    if not Path(metallib_bin).is_file():
        metallib_bin = "metallib"

    air_files: list[Path] = []
    for name, extra_flags in _METAL_SOURCES:
        src = _CSRC / name
        air = out_dir / (Path(name).stem + ".air")
        cmd = [metal_bin, "-c", str(src), "-o", str(air), *extra_flags]
        if name == "mxfp8_matmul.metal" and Path(_MPP_INCLUDE).is_dir():
            cmd.extend(["-I", _MPP_INCLUDE])
        _run(cmd, env=env)
        air_files.append(air)
    metallib = out_dir / _METALLIB_NAME
    # Write via a temp path — metallib refuses some in-place overwrites / broken symlinks.
    tmp = out_dir / (_METALLIB_NAME + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    _run([metallib_bin, *[str(a) for a in air_files], "-o", str(tmp)], env=env)
    if metallib.exists() or metallib.is_symlink():
        metallib.unlink()
    tmp.rename(metallib)
    return metallib


def load_extension(force_rebuild: bool = False):
    import torch
    from torch.utils.cpp_extension import load

    if sys.platform != "darwin":
        raise RuntimeError("metal backend requires macOS")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    _BUILD.mkdir(parents=True, exist_ok=True)
    metallib = _BUILD / _METALLIB_NAME
    metal_src_mtime = max((_CSRC / name).stat().st_mtime for name, _ in _METAL_SOURCES)
    need_metallib = (
        force_rebuild
        or not metallib.is_file()
        or metallib.stat().st_mtime < metal_src_mtime
    )
    if need_metallib:
        metallib = compile_metallib(_BUILD)
    os.environ["COMFY_KITCHEN_METALLIB"] = str(metallib.resolve())

    module = load(
        name=_EXT_NAME,
        sources=[str(_CSRC / "metal_ops.mm")],
        extra_cflags=["-O3", "-std=c++20", "-fobjc-arc", "-Wno-unused-parameter"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        build_directory=str(_BUILD),
        verbose=bool(os.environ.get("COMFY_KITCHEN_METAL_VERBOSE")),
        with_cuda=False,
    )
    return module


_ops = None
_load_error: Exception | None = None


def get_ops(force_rebuild: bool = False):
    global _ops, _load_error
    if _ops is not None and not force_rebuild:
        return _ops
    if _load_error is not None and not force_rebuild:
        raise _load_error
    try:
        _ops = load_extension(force_rebuild=force_rebuild)
        _load_error = None
        return _ops
    except Exception as e:
        _load_error = e
        _logger.warning("metal extension unavailable: %s", e)
        raise
