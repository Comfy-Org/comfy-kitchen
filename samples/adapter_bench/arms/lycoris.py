# ruff: noqa: N803  -- LyCORIS's kernel signature, kept verbatim.
"""LyCORIS's register-resident LoKr bypass kernel (lycoris/kernels/triton/lokr/bypass.py),
copied verbatim, with its own plan space (tokens per program, warps, stages).

It holds whole factors in registers, padded to powers of two up to 128;
larger factors are outside its scope (LyCORIS hands those to torch). Like
LyCORIS's tuner, the arm times its plan space once per case and keeps the
best; the base add is a torch add, as LyCORIS's bypass composes it.
"""

import torch
import triton
import triton.language as tl

from ..core.timing import device_ms

MAX_KRON_FACTOR = 128


@triton.jit
def _lokr_bypass_fwd_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    T,
    A,
    B,
    C,
    D,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    syt,
    syo,
    gamma,
    PREC: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    PC: tl.constexpr,
    PD: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BT * PB)
    ti = rows // PB
    bi = rows % PB
    tok = pid * BT + ti
    rd = tl.arange(0, PD)
    xmask = (tok < T)[:, None] & (bi < B)[:, None] & (rd < D)[None, :]
    x2 = tl.load(
        x_ptr + tok[:, None] * sxt + (bi[:, None] * D + rd[None, :]) * sxi,
        mask=xmask,
        other=0.0,
    )
    rc = tl.arange(0, PC)
    w2t = tl.load(
        w2_ptr + rc[None, :] * s2c + rd[:, None] * s2d,
        mask=(rc < C)[None, :] & (rd < D)[:, None],
        other=0.0,
    )
    m = tl.dot(x2, w2t, input_precision=PREC)
    m2 = tl.reshape(tl.permute(tl.reshape(m, (BT, PB, PC)), (0, 2, 1)), (BT * PC, PB))
    ra = tl.arange(0, PA)
    rb = tl.arange(0, PB)
    w1t = tl.load(
        w1_ptr + ra[None, :] * s1a + rb[:, None] * s1b,
        mask=(ra < A)[None, :] & (rb < B)[:, None],
        other=0.0,
    )
    y2 = tl.dot(m2.to(w1t.dtype), w1t, input_precision=PREC)
    rows2 = tl.arange(0, BT * PC)
    ti2 = rows2 // PC
    ci = rows2 % PC
    tok2 = pid * BT + ti2
    ymask = (tok2 < T)[:, None] & (ci < C)[:, None] & (ra < A)[None, :]
    tl.store(
        y_ptr + tok2[:, None] * syt + (ra[None, :] * C + ci[:, None]) * syo,
        (y2 * gamma).to(y_ptr.dtype.element_ty),
        mask=ymask,
    )


def _pad(v: int) -> int:
    return max(16, triton.next_power_of_2(v))


def supported(a: int, b: int, c: int, d: int) -> bool:
    return max(_pad(v) for v in (a, b, c, d)) <= MAX_KRON_FACTOR


def _space(a, b, c, d):
    pa, pb, pc, pd = (_pad(v) for v in (a, b, c, d))
    out = []
    for bt in (1, 2, 4, 8, 16, 32):
        if max(bt * pb * pd, bt * pc * pa, bt * pb * pc) > 16384:
            continue
        for w in (2, 4, 8):
            for s in (1, 2, 3):
                out.append((bt, w, s))
    return out


def _launch(cfg, x, w1, w2, gamma, y):
    bt, w, s = cfg
    t = x.shape[0]
    a, b = w1.shape
    c, d = w2.shape
    _lokr_bypass_fwd_kernel[(triton.cdiv(t, bt),)](
        x,
        w1,
        w2,
        y,
        t,
        a,
        b,
        c,
        d,
        *x.stride(),
        *w1.stride(),
        *w2.stride(),
        *y.stride(),
        gamma,
        PREC="tf32",
        BT=bt,
        PA=_pad(a),
        PB=_pad(b),
        PC=_pad(c),
        PD=_pad(d),
        num_warps=w,
        num_stages=s,
    )


def tuned_bypass(x, w1, w2, base, scale):
    """base + LyCORIS lokr_bypass_fwd(x) with its best plan for this case."""
    flat = x.reshape(-1, x.shape[-1])
    a, c = w1.shape[0], w2.shape[0]
    y = torch.empty(flat.shape[0], a * c, device=x.device, dtype=x.dtype)
    best, best_ms = None, float("inf")
    for cfg in _space(*w1.shape, *w2.shape):
        try:
            ms, _ = device_ms(
                lambda cfg=cfg: _launch(cfg, flat, w1, w2, scale, y), warmup=1, iters=5
            )
        except Exception:
            continue
        if ms < best_ms:
            best, best_ms = cfg, ms
    if best is None:
        raise RuntimeError("no LyCORIS plan compiled for this case")

    def run(x, w1, w2, base, scale, **_):
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty(flat.shape[0], a * c, device=x.device, dtype=x.dtype)
        _launch(best, flat, w1, w2, scale, out)
        return base + out.reshape(*x.shape[:-1], a * c)

    return run, best
