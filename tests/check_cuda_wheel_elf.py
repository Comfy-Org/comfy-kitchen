"""Reject CUDA runtime ABI and absolute RPATH regressions in built Linux wheels."""

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        extensions = [
            name for name in archive.namelist()
            if name.startswith("comfy_kitchen/backends/cuda/_C") and name.endswith(".so")
        ]
        if len(extensions) != 1:
            raise RuntimeError(f"{wheel}: expected exactly one native CUDA extension, got {extensions}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(archive.extract(extensions[0], tmp))
            completed = subprocess.run(
                ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
            )
            dynamic_section = completed.stdout

    needed = re.findall(
        r"\(NEEDED\).*?Shared library: \[([^\]]+)\]", dynamic_section
    )
    cuda_runtime = [name for name in needed if name.startswith("libcudart.so.")]
    if cuda_runtime != ["libcudart.so.13"]:
        raise RuntimeError(
            f"{wheel}: expected libcudart.so.13, got {cuda_runtime}; DT_NEEDED={needed}"
        )

    search_paths = re.findall(
        r"\((?:RPATH|RUNPATH)\).*?\[([^\]]+)\]", dynamic_section
    )
    for group in search_paths:
        for path in group.split(":"):
            if path.startswith("/"):
                raise RuntimeError(f"{wheel}: absolute CUDA/build-machine RPATH: {path}")

    print(f"{wheel.name}: libcudart.so.13 dependency and relocatable RPATH verified")


if __name__ == "__main__":
    wheels = [Path(path) for path in sys.argv[1:]]
    if not wheels:
        raise SystemExit("Pass the Linux wheel paths to verify")
    for wheel_path in wheels:
        check_wheel(wheel_path)
