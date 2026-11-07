"""Validate compiler and Python interpreter cache switches without running CUDA."""

import ast
import os
import pathlib
import shutil


NVCC = pathlib.Path("/opt/conda/env/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc")
PYTHON = pathlib.Path("/opt/conda/env/bin/python3.12")
INCLUDE = pathlib.Path("/opt/conda/env/include/python3.12")


def _cache_reset():
    setup_path = pathlib.Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reset_cuda_cmake_cache_on_toolchain_switch"
    ]
    assert len(matches) == 1
    scope = {"pathlib": pathlib, "os": os, "shutil": shutil}
    exec(compile(ast.Module(body=matches, type_ignores=[]), str(setup_path), "exec"), scope)
    return scope[matches[0].name]


def test_cuda_library_cached_from_other_toolkit_triggers_reset(tmp_path):
    """Compiler identity alone cannot qualify a stale FindCUDAToolkit library."""
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={NVCC}\n"
        f"Python_EXECUTABLE:FILEPATH={PYTHON}\n"
        f"_Python_EXECUTABLE:INTERNAL={PYTHON}\n"
        f"Python_INCLUDE_DIR:FILEPATH={INCLUDE}\n"
        "CUDA_cudart_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcudart.so.11.0\n"
        "CUDA_cudart_static_LIBRARY:FILEPATH=/usr/local/cuda-11.5/lib64/libcudart_static.a\n",
        encoding="utf-8",
    )
    assert reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert not cache.exists()


def test_cuda_library_cached_inside_selected_toolkit_is_unchanged(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    root = NVCC.parent.parent
    cache.write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={NVCC}\n"
        f"CUDAToolkit_ROOT:UNINITIALIZED={root}\n"
        f"CUDA_cudart_LIBRARY:FILEPATH={root}/lib/libcudart.so.13\n"
        f"CUDA_cudart_static_LIBRARY:FILEPATH={root}/lib/libcudart_static.a\n",
        encoding="utf-8",
    )
    assert not reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert cache.exists()


def test_cuda_root_change_resets_stale_cache_without_compiler_change(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={NVCC}\n"
        "CUDAToolkit_ROOT:UNINITIALIZED=/usr/local/cuda-11.5\n",
        encoding="utf-8",
    )
    assert reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert not cache.exists()


def test_cuda_compiler_switch_removes_only_generated_backend_cache(tmp_path):
    reset = _cache_reset()
    root = tmp_path / "build" / "temp.linux-x86_64-cpython-312"
    cuda, hip = root / "cuda", root / "hip"
    cuda.mkdir(parents=True)
    hip.mkdir()
    (hip / "HIP_BUILD_SENTINEL").write_text("keep", encoding="utf-8")
    (cuda / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_COMPILER:FILEPATH=/usr/bin/nvcc\n"
        "COMFY_CUDA_ARCHS:UNINITIALIZED=75-real;80-real;120f\n",
        encoding="utf-8",
    )
    (cuda / "CMakeFiles").mkdir()
    (cuda / "CMakeFiles" / "old.txt").write_text("old", encoding="utf-8")
    assert reset(cuda, NVCC, PYTHON, INCLUDE)
    assert cuda.is_dir() and list(cuda.iterdir()) == []
    assert (hip / "HIP_BUILD_SENTINEL").read_text(encoding="utf-8") == "keep"


def test_stale_internal_findpython_overrides_correct_public_python_argument(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={NVCC}\n"
        f"Python_EXECUTABLE:UNINITIALIZED={PYTHON}\n"
        "_Python_EXECUTABLE:INTERNAL=/usr/bin/python3.10\n"
        "_Python_INCLUDE_DIR:INTERNAL=/usr/include/python3.10\n",
        encoding="utf-8",
    )
    assert reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert tmp_path.is_dir() and list(tmp_path.iterdir()) == []


def test_stale_internal_python_include_invalidates_even_if_executables_match(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={NVCC}\n"
        f"Python_EXECUTABLE:FILEPATH={PYTHON}\n"
        f"_Python_EXECUTABLE:INTERNAL={PYTHON}\n"
        "_Python_INCLUDE_DIR:INTERNAL=/usr/include/python3.10\n",
        encoding="utf-8",
    )
    assert reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert not cache.exists()


def test_unchanged_compiler_and_interpreter_keep_incremental_state(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={NVCC}\n"
        f"Python_EXECUTABLE:UNINITIALIZED={PYTHON}\n"
        f"_Python_EXECUTABLE:INTERNAL={PYTHON}\n"
        f"Python_INCLUDE_DIR:UNINITIALIZED={INCLUDE}\n"
        f"_Python_INCLUDE_DIR:INTERNAL={INCLUDE}\n",
        encoding="utf-8",
    )
    (tmp_path / "CMakeFiles").mkdir()
    assert not reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert cache.exists() and (tmp_path / "CMakeFiles").is_dir()


def test_partial_cache_invalidation_removes_residual_cmakefiles(tmp_path):
    reset = _cache_reset()
    (tmp_path / "CMakeFiles").mkdir()
    (tmp_path / "CMakeFiles" / "stale.txt").write_text("old", encoding="utf-8")
    assert not reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert not (tmp_path / "CMakeFiles").exists()


def test_cache_without_qualified_toolchain_artifacts_is_preserved(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("COMFY_CUDA_ARCHS:UNINITIALIZED=120f\n", encoding="utf-8")
    assert not reset(tmp_path, NVCC, PYTHON, INCLUDE)
    assert cache.exists()
