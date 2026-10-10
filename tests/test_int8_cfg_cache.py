"""The per-shape INT8 cfg table loader: which picks it serves, to which device, and
that a malformed table can never raise into the dispatch path. No GPU needed: the
device queries are faked."""

import json
import types

import pytest
import torch

from comfy_kitchen import _int8_cfg_cache as cache

TABLE = {
    "device": "NVIDIA RTX A6000",
    "sm_version": "8.6",
    "multiprocessor_count": 84,
    "m_min_swept": "274",  # a hand-edited string must not reach the int comparison
    "m_max_swept": None,
    "shapes": {
        "1024x4096x4096": {"m": 1024, "n": 4096, "k": 4096, "best_cfg": 12},
        "53730x21504x5376": {"m": 53730, "n": 21504, "k": 5376, "best_cfg": 13},
        "274x2048x2048": {"m": 274, "n": 2048, "k": 2048, "best_cfg": 2},
        "bad-key": {"best_cfg": 1},
        "8x8x8": {"best_cfg": "twelve"},
        "16x16x16": {"best_cfg": -3},
        "32x32x32": "not an object",
        "64x64x64": {"best_ms": 1.0},  # no best_cfg: ignored, not counted as malformed
    },
}


@pytest.fixture
def table(tmp_path, monkeypatch):
    def write(**overrides):
        data = {**TABLE, **overrides}
        path = tmp_path / "table.json"
        path.write_text(json.dumps(data))
        monkeypatch.setenv("COMFY_KITCHEN_INT8_CFG_CACHE", str(path))
        cache.reset()
        return path

    yield write
    cache.reset()


@pytest.fixture
def fake_device(monkeypatch):
    def install(capability=(8, 6), sms=84):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=0: capability)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx=0: types.SimpleNamespace(multi_processor_count=sms),
        )

    return install


def test_serves_exact_shapes_and_skips_malformed_entries(table, fake_device):
    table()
    fake_device()
    assert cache.get_cfg(1024, 4096, 4096, 2) == 12
    assert cache.get_cfg(53730, 21504, 5376, 2) == 13
    assert cache.get_cfg(1024, 4096, 4096, 1) is None  # only the swept bf16 epilogue
    assert cache.get_cfg(9999, 4096, 4096, 2) is None
    assert len(cache._loaded) == 3


def test_swept_range_comes_from_the_entries(table, fake_device):
    table()
    fake_device()
    cache.get_cfg(1024, 4096, 4096, 2)
    assert (cache._m_min_swept, cache._m_max_swept) == (274, 53730)
    cache.check_m_in_swept_range(100)  # a string bound here would have raised TypeError
    cache.check_m_in_swept_range(100000)


@pytest.mark.parametrize(
    ("capability", "sms", "served"),
    [
        ((8, 6), 84, True),
        ((8, 6), 82, True),  # RTX 3090: within 10%
        ((8, 6), 28, False),  # RTX 3060: same SM version, a third of the SMs
        ((8, 9), 84, False),  # other SM version
    ],
)
def test_table_only_serves_like_silicon(table, fake_device, capability, sms, served):
    table()
    fake_device(capability, sms)
    assert (cache.get_cfg(1024, 4096, 4096, 2) == 12) is served


def test_table_without_device_fields_serves_anyone(table, fake_device):
    table(sm_version=None, multiprocessor_count="84")  # non-int count is ignored
    fake_device((12, 0), 170)
    assert cache.get_cfg(1024, 4096, 4096, 2) == 12


def test_device_fit_is_decided_once_per_device(table, fake_device, monkeypatch):
    table()
    fake_device()
    calls = []
    monkeypatch.setattr(cache, "_table_fits_device", lambda idx: calls.append(idx) or True)
    for _ in range(3):
        cache.get_cfg(1024, 4096, 4096, 2, device_index=1)
    assert calls == [1]


def test_missing_override_falls_through_to_packaged_table(tmp_path, monkeypatch, fake_device):
    monkeypatch.setenv("COMFY_KITCHEN_INT8_CFG_CACHE", str(tmp_path / "absent.json"))
    cache.reset()
    fake_device()
    cache.get_cfg(1024, 4096, 4096, 2)
    assert cache._loaded_from is not None and cache._loaded_from.name == "a6000_int8_cfg_table.json"
    cache.reset()
