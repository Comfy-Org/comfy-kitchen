"""Compiled weighted Sol-Attn interaction coverage for token routing.

This file stays separate from the general Sol-Attn suite so the weighted
Mixed-Grid acceptance case is easy to target on a supported CUDA/HIP host.
"""

import math

import pytest
import torch

from tests.test_sol_attn import _chunked_case, _cos, backend


pytestmark = pytest.mark.skipif(
    backend is None, reason="a compiled CUDA or HIP backend with sol_attn is required"
)


def test_chunked_key_bias_with_token_aug_preserves_exact_sink_and_calibration():
    """Weighted sink blocks stay exact while token routing operates on the tail.

    The direct fused path and the fused-QKV producer share the sparse algorithm
    but reach it through independent preprocessing paths.  Enabling token
    routing here exercises the interaction that the zero-token-budget weighted
    parity case does not: the biased sink is exact, unrelated unrouted blocks
    may enter token routing, and producer calibration must remain bit-identical
    to the unweighted activation statistics.
    """
    c = _chunked_case(seed=31, rot=96, v_scale=0.02)
    bias = torch.zeros(c["t"], device="cuda")
    bias[64:192] = math.log(0.37)
    sinks = [1, 3]
    common = {"tau": 1.4, "token_aug": 64, "sink_blocks": sinks}

    ref = backend.sol_attn(
        c["q"], c["k"], c["v"], key_bias=bias, **common
    )
    cold, kmean, vscale = backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"],
        key_bias=bias, **common
    )
    primed, _, _ = backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"],
        kmean=kmean, vscale=vscale, key_bias=bias, **common
    )
    plain, plain_kmean, plain_vscale = backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], **common
    )

    assert torch.equal(kmean, plain_kmean)
    assert torch.equal(vscale, plain_vscale)
    assert torch.isfinite(cold.float()).all()
    assert torch.isfinite(primed.float()).all()
    assert _cos(cold, ref) > 0.995
    assert _cos(primed, ref) > 0.995
    assert not torch.equal(primed, plain)
