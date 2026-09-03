"""HIP dispatch and gating logic. Needs no GPU and no compiled extension.

These cover which architectures the backend registers for and which ops it
advertises on each. They are deliberately not gated on registry.is_available("hip"):
on a CPU-only runner the kernel suite in test_hip_wmma.py skips in full, and these
routing rules would otherwise go untested behind a green tick.
"""
import ast
import json
import pathlib
import re
import subprocess
import sys

import pytest
import torch

import comfy_kitchen.scaled_mm_v2 as scaled_mm_module
from comfy_kitchen.backends import hip as hip_backend

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HIP_DIR = _ROOT / "comfy_kitchen" / "backends" / "hip"
_HIP_CMAKE = _HIP_DIR / "CMakeLists.txt"
_HIP_ARCH_MANIFEST = _HIP_DIR / "architectures.json"
_HIP_ARCH_GROUP_NAMES = (
    "elementwise_only",
    "wmma_gfx11",
    "wmma_gfx12",
)


def _architecture_groups() -> dict[str, list[str]]:
    return json.loads(_HIP_ARCH_MANIFEST.read_text(encoding="utf-8"))


def _manifest_archs() -> list[str]:
    groups = _architecture_groups()
    return [arch for group_name in _HIP_ARCH_GROUP_NAMES for arch in groups[group_name]]


@pytest.mark.parametrize("value", [False, True])
def test_hip_attention_scale_rejects_boolean_tensor(value):
    q = torch.empty(1, 1, 1, 128)
    with pytest.raises(TypeError, match="scale"):
        hip_backend.hip_int8_attention(q, None, None, torch.tensor(value))


def test_non_rocm_runtime_does_not_import_hip_backend():
    """A combined wheel must not load the ROCm runtime in CUDA/CPU processes."""
    if getattr(torch.version, "hip", None):
        pytest.skip("requires a non-ROCm PyTorch runtime")

    code = """
import sys
import comfy_kitchen as ck

assert "comfy_kitchen.backends.hip" not in sys.modules
status = ck.list_backends()["hip"]
assert not status["available"]
assert status["unavailable_reason"] == "PyTorch ROCm/HIP runtime not available"
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_scaled_mm_does_not_probe_hip_on_non_rocm_runtime(monkeypatch):
    """Keep the HIP routing check off NVIDIA's latency-sensitive FP8 path."""
    if getattr(torch.version, "hip", None):
        pytest.skip("requires a non-ROCm PyTorch runtime")

    sentinel = object()

    def unexpected_hip_probe(*args, **kwargs):
        raise AssertionError("HIP was probed without a ROCm PyTorch runtime")

    monkeypatch.setattr(scaled_mm_module, "_hip_fp8_gemm", unexpected_hip_probe)
    monkeypatch.setattr(scaled_mm_module, "has_scaled_mm_v2", lambda: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_mm", lambda *args, **kwargs: sentinel)

    result = scaled_mm_module.scaled_mm_v2(
        object(),
        object(),
        object(),
        object(),
    )
    assert result is sentinel


@pytest.mark.parametrize(
    "arches",
    [
        [],
        ["gfx90a"],   # CDNA: MFMA, not WMMA
        ["gfx1010"],  # RDNA1: neither matrix cores nor the dot-product paths
        ["gfx1201", "gfx90a"],
    ],
)
def test_hip_declines_unsupported_arch(arches):
    """The backend registers for RDNA2/3/4 and nothing else."""
    assert hip_backend._unsupported_arch_reason(arches) is not None


@pytest.mark.parametrize("arch", _manifest_archs())
def test_hip_accepts_every_manifest_target(arch):
    assert hip_backend._unsupported_arch_reason([arch]) is None


@pytest.mark.parametrize(
    "arch",
    [
        "gfx1037",
        "gfx1104",
        "gfx1154",
        "gfx1170",
        "gfx1171",
        "gfx1172",
        "gfx1202",
        "gfx1250",
        "gfx1251",
        "gfx11-generic",
        "gfx12-generic",
        "gfx12-5-generic",
    ],
)
def test_hip_rejects_unreviewed_near_prefix_targets(arch):
    """A future target must not silently inherit a possibly incompatible WMMA policy."""
    assert hip_backend._unsupported_arch_reason([arch]) is not None
    assert not hip_backend._has_wmma([arch])


def test_hip_declines_when_an_arch_cannot_be_read():
    """A device whose architecture is unknown cannot be shown to be supported."""
    assert hip_backend._unsupported_arch_reason([None]) is not None
    assert hip_backend._unsupported_arch_reason(["gfx1200", None]) is not None


@pytest.mark.parametrize(
    ("arches", "expected"),
    [
        (["gfx1200"], True),
        (["gfx1201", "gfx1100"], True),
        (["gfx1151"], True),
        (["gfx1030"], False),             # RDNA2 has no matrix cores
        (["gfx1200", "gfx1030"], False),  # kernels launch on the tensor's own device
        ([None], False),
    ],
)
def test_hip_wmma_capability(arches, expected):
    """Only an all-matrix-core process may advertise the GEMMs."""
    assert hip_backend._has_wmma(arches) is expected


def test_hip_drops_gemms_without_matrix_cores():
    """RDNA2 keeps the elementwise kernels and hands the GEMMs back to triton/eager."""
    with_wmma = hip_backend._build_constraints(has_wmma=True)
    without = hip_backend._build_constraints(has_wmma=False)

    # Every WMMA-only op must be advertised with matrix cores; an intersection
    # would pass while any single one was missing from the constraints.
    assert set(with_wmma) >= hip_backend._WMMA_ONLY_OPS
    assert not (hip_backend._WMMA_ONLY_OPS & set(without))
    # na3d is a matrix-core kernel: without one it traps rather than answers.
    assert "na3d" in hip_backend._WMMA_ONLY_OPS
    # The elementwise kernels need no matrix cores and must survive.
    for op in ("apply_rope", "apply_rope_", "rms_rope", "rms_rope_split_half1_", "adaln",
               "rms_adaln", "quantize_per_tensor_fp8", "gemv_awq_w4a16",
               "dequantize_int8_simple_dtype",
               "dequantize_int8_convrot_weight_dtype"):
        assert op in without
    # The fused W4A8 requantize is elementwise too: it packs weights and never
    # reaches a matrix core, so RDNA2 must keep it.
    assert "quantize_w4a8_int8_weight" in without


def test_hip_advertises_every_inplace_rope_entry():
    """A missing entry routes the in-place call to eager while the functional one
    stays on HIP: silently half the coverage rather than a failure."""
    constraints = hip_backend._build_constraints(has_wmma=True)
    for functional in ("apply_rope", "apply_rope1", "apply_rope_split_half",
                       "apply_rope_split_half1", "rms_rope", "rms_rope1",
                       "rms_rope_split_half", "rms_rope_split_half1"):
        assert constraints[f"{functional}_"] is constraints[functional]


def _setup_namespace() -> dict:
    """Load setup.py definitions without executing its final setuptools.setup()."""
    path = _ROOT / "setup.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "extensions"
            for target in node.targets
        ):
            break
        body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    namespace = {"__file__": str(path), "__name__": "comfy_kitchen_setup_test"}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_setup_keeps_cuda_build_cuda_only_by_default():
    namespace = _setup_namespace()
    cuda_extension = object()

    namespace["setup_cuda_extension"] = lambda: cuda_extension
    namespace["setup_hip_extension"] = lambda: pytest.fail(
        "an incidental ROCm compiler must not add HIP to a CUDA source build"
    )

    assert namespace["get_extensions"]() == [cuda_extension]


def test_setup_builds_both_backends_only_when_hip_is_requested():
    namespace = _setup_namespace()
    cuda_extension = object()
    hip_extension = object()

    namespace["BUILD_HIP"] = True
    namespace["setup_cuda_extension"] = lambda: cuda_extension
    namespace["setup_hip_extension"] = lambda: hip_extension

    assert namespace["get_extensions"]() == [cuda_extension, hip_extension]


def test_setup_refuses_hip_only_fallback_under_cuda_pytorch():
    namespace = _setup_namespace()
    missing_cuda = namespace["CudaToolkitNotFoundError"]

    def raise_missing_cuda():
        raise missing_cuda("nvcc missing")

    namespace["setup_cuda_extension"] = raise_missing_cuda
    namespace["get_rocm_path"] = lambda: ("/opt/rocm", object())
    namespace["get_torch_gpu_runtime"] = lambda: "cuda"
    namespace["setup_hip_extension"] = lambda: pytest.fail(
        "CUDA PyTorch must not silently receive a HIP-only native build"
    )

    with pytest.raises(missing_cuda, match="refusing to replace"):
        namespace["get_extensions"]()


def test_no_cuda_means_python_only_unless_hip_is_explicit():
    namespace = _setup_namespace()
    namespace["BUILD_NO_CUDA"] = True
    namespace["setup_cuda_extension"] = lambda: pytest.fail("CUDA was not disabled")
    namespace["setup_hip_extension"] = lambda: pytest.fail("HIP was not requested")

    assert namespace["get_extensions"]() == []


def test_rocm_only_build_still_auto_selects_hip():
    namespace = _setup_namespace()
    missing_cuda = namespace["CudaToolkitNotFoundError"]
    hip_extension = object()

    def raise_missing_cuda():
        raise missing_cuda("nvcc missing")

    namespace["setup_cuda_extension"] = raise_missing_cuda
    namespace["get_rocm_path"] = lambda: ("/opt/rocm", object())
    namespace["get_torch_gpu_runtime"] = lambda: "hip"
    namespace["setup_hip_extension"] = lambda: hip_extension

    assert namespace["get_extensions"]() == [hip_extension]


def test_architecture_manifest_is_unique_and_shared_by_setup_and_runtime():
    groups = _architecture_groups()
    manifest_archs = _manifest_archs()
    namespace = _setup_namespace()

    assert tuple(groups) == _HIP_ARCH_GROUP_NAMES
    assert len(manifest_archs) == len(set(manifest_archs))
    assert tuple(manifest_archs) == namespace["SUPPORTED_HIP_ARCHS"]
    assert manifest_archs == namespace["DEFAULT_HIP_ARCHS"].split(";")
    assert set(groups["elementwise_only"]) == hip_backend._ARCH_ELEMENTWISE_ONLY
    assert set(groups["wmma_gfx11"]) == hip_backend._ARCH_WMMA_GFX11
    assert set(groups["wmma_gfx12"]) == hip_backend._ARCH_WMMA_GFX12
    assert "gfx1200" in groups["wmma_gfx12"]
    assert "gfx1201" in groups["wmma_gfx12"]
    assert set(manifest_archs) == hip_backend._ARCH_SUPPORTED


def test_setup_architecture_check_is_exact_and_fail_closed():
    namespace = _setup_namespace()
    supported = namespace["hip_arch_supported"]

    assert all(supported(arch) for arch in _manifest_archs())
    for arch in ("gfx1037", "gfx1104", "gfx1170", "gfx1202", "gfx1250"):
        assert not supported(arch)


def test_cmake_reads_and_validates_the_shared_architecture_manifest():
    text = _HIP_CMAKE.read_text(encoding="utf-8")

    assert 'file(READ "${_CK_HIP_ARCH_MANIFEST}"' in text
    assert "string(JSON" in text
    assert "IN_LIST COMFY_HIP_SUPPORTED_ARCHS" in text
    assert "configure_file(" in text
    assert not re.search(r'"gfx\d', text)

    read_groups = re.findall(r"_ck_read_arch_group\(\w+ (\w+)\)", text)
    assert tuple(read_groups) == _HIP_ARCH_GROUP_NAMES


def test_cmake_uses_only_gfx12_group_for_gfx12_wmma_policy():
    text = _HIP_CMAKE.read_text(encoding="utf-8")

    gfx12_condition = re.search(
        r"_ck_make_arch_condition\(COMFY_HIP_GFX12_CONDITION(.*?)\)", text, re.DOTALL
    )
    assert gfx12_condition is not None
    assert "COMFY_HIP_WMMA_GFX12_ARCHS" in gfx12_condition.group(1)


def test_mma_architecture_macros_are_generated_from_the_manifest():
    mma = (_HIP_DIR / "mma.h").read_text(encoding="utf-8")
    template = (_HIP_DIR / "architecture_config.h.in").read_text(encoding="utf-8")

    assert '#include "architecture_config.h"' in mma
    assert "defined(__gfx" not in mma
    assert "@COMFY_HIP_GFX11_CONDITION@" in template
    assert "@COMFY_HIP_GFX12_CONDITION@" in template


def test_sdist_rules_include_every_hip_build_input():
    manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "include comfy_kitchen/backends/hip/CMakeLists.txt" in manifest
    assert "recursive-include comfy_kitchen/backends/hip *.cpp *.h *.hip *.in *.json" in manifest
    assert "include-package-data = false" in pyproject
    assert (
        '"comfy_kitchen.backends.hip" = '
        '["architectures.json"]'
        in pyproject
    )


def test_hip_kernels_are_independent_of_the_python_extension_target():
    """Keep expensive HIP objects reusable across the CPython wheel matrix."""
    text = _HIP_CMAKE.read_text(encoding="utf-8")

    assert "add_library(comfy_kitchen_hip_kernels OBJECT ${HIP_SOURCES})" in text
    assert "target_sources(_C PRIVATE $<TARGET_OBJECTS:comfy_kitchen_hip_kernels>)" in text

    module_calls = re.findall(r"nanobind_add_module\((.*?)\)", text, re.DOTALL)
    assert module_calls
    assert all("${HIP_SOURCES}" not in call for call in module_calls)


def test_combined_wheel_cache_keys_include_hip_sources():
    """HIP-only changes must produce a cache key that Actions can save."""
    workflow = (_ROOT / ".github" / "workflows" / "build-wheels.yml").read_text(
        encoding="utf-8"
    )
    combined_cache_keys = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("key: ccache-")
        and ("linux-x86_64" in line or "windows-x86_64" in line)
    ]

    assert len(combined_cache_keys) == 2
    assert all("'comfy_kitchen/backends/hip/**'" in key for key in combined_cache_keys)
    windows_key = next(key for key in combined_cache_keys if "windows-x86_64" in key)
    assert "py${{ matrix.python-version }}" in windows_key
    versioned_windows_prefix = (
        "ccache-windows-x86_64-cuda13-py${{ matrix.python-version }}-"
    )
    assert workflow.count(versioned_windows_prefix) == 2  # primary key + restore prefix
