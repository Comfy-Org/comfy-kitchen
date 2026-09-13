from __future__ import annotations

import ast
from pathlib import Path


_LEGACY_TAIL = [
    "rope_eps",
    "tail",
    "block_len",
    "coarse_gate",
    "token_aug",
]


def _chunked_signature(path: str) -> tuple[list[str], list[str], list[ast.expr | None]]:
    tree = ast.parse(Path(path).read_text())
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "sol_attn_chunked"
    ]
    assert len(matches) == 1
    fn = matches[0]
    positional = [arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args)]
    keyword_only = [arg.arg for arg in fn.args.kwonlyargs]
    return positional, keyword_only, fn.args.kw_defaults


def test_chunked_key_bias_is_optional_keyword_only_after_legacy_positional_tail():
    for path in (
        "comfy_kitchen/backends/cuda/__init__.py",
        "comfy_kitchen/backends/hip/__init__.py",
    ):
        positional, keyword_only, keyword_defaults = _chunked_signature(path)
        assert positional[-5:] == _LEGACY_TAIL
        assert "key_bias" not in positional
        index = keyword_only.index("key_bias")
        default = keyword_defaults[index]
        assert isinstance(default, ast.Constant)
        assert default.value is None
