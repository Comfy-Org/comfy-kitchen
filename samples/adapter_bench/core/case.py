"""One benchmark case: every arm timed forward and forward+backward, with its
peak memory and its error against an fp64 reference, forward and gradients.

A Case names its inputs once; each arm is a function of those inputs by
keyword. The reference is the operation's definition evaluated in fp64,
differentiated by autograd for the gradient error.
"""

import dataclasses
from collections.abc import Callable

import torch

from .accuracy import rel_err, ulp_err
from .timing import interleave, peak_extra_bytes


@dataclasses.dataclass
class Arm:
    fn: Callable
    backward: bool = True
    note: str = ""


@dataclasses.dataclass
class Case:
    family: str
    op: str
    sweep: str
    x: float
    case: str
    params: dict
    inputs: dict
    grad_names: tuple
    ref: Callable
    arms: dict
    logical_bytes: float
    logical_flops: float
    ulp_mode: str = "rms"


def _upstream(like: torch.Tensor) -> torch.Tensor:
    """The backward's incoming gradient, drawn on a CPU generator: the default
    CUDA generator is left unusable by an arm whose graph capture failed."""
    gen = torch.Generator().manual_seed(1)
    return torch.randn(like.shape, generator=gen).to(like.device, like.dtype)


def _reference(case: Case, upstream: torch.Tensor | None):
    leaves = {}
    inputs64 = {}
    for k, v in case.inputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            t = v.detach().double()
            if upstream is not None and k in case.grad_names:
                t.requires_grad_(True)
                leaves[k] = t
            inputs64[k] = t
        else:
            inputs64[k] = v
    out = case.ref(**inputs64)
    grads = {}
    if upstream is not None and leaves:
        got = torch.autograd.grad(out, list(leaves.values()), upstream.double())
        grads = {k: g.detach() for k, g in zip(leaves, got, strict=True)}
    return out.detach(), grads


def _fwdbwd_closure(arm: Arm, case: Case, upstream: torch.Tensor):
    leaves = {
        k: case.inputs[k].detach().clone().requires_grad_(True) for k in case.grad_names
    }
    kwargs = {**case.inputs, **leaves}

    def run():
        for t in leaves.values():
            t.grad = None
        arm.fn(**kwargs).backward(upstream)

    return run, leaves


def measure_case(case: Case, dtype: torch.dtype, meta: dict, log=print) -> list[dict]:
    upstream = None
    probe_out = None
    for arm in case.arms.values():
        try:
            with torch.no_grad():
                probe_out = arm.fn(**case.inputs)
            break
        except Exception:
            continue
    if probe_out is None:
        return []
    if any(a.backward for a in case.arms.values()) and case.grad_names:
        upstream = _upstream(probe_out)
    ref_out, ref_grads = _reference(case, upstream)

    fwd, fwdbwd, outs, grads, errors = {}, {}, {}, {}, {}
    for name, arm in case.arms.items():

        def fwd_fn(arm=arm):
            with torch.no_grad():
                return arm.fn(**case.inputs)

        try:
            outs[name] = fwd_fn()
            fwd[name] = fwd_fn
        except Exception as e:  # recorded on the row: a failing arm is a result
            errors[name] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        if arm.backward and upstream is not None:
            try:
                run, leaves = _fwdbwd_closure(arm, case, upstream)
                run()
                missing = [
                    k
                    for k, t in leaves.items()
                    if t.grad is None or not bool(t.grad.abs().sum() > 0)
                ]
                if missing:
                    raise RuntimeError(f"backward did not reach {missing}")
                grads[name] = {k: t.grad.detach().clone() for k, t in leaves.items()}
                fwdbwd[name] = run
            except Exception as e:
                errors[name] = f"backward {type(e).__name__}: {str(e)[:200]}"

    t_fwd = interleave(fwd)
    t_bwd = interleave(fwdbwd) if fwdbwd else {}
    rows = []
    for name, arm in case.arms.items():
        row = {
            "family": case.family,
            "op": case.op,
            "sweep": case.sweep,
            "x": case.x,
            "case": case.case,
            "arm": name,
            **case.params,
            "note": arm.note,
            "error": errors.get(name),
        }
        if name in fwd:
            ms, hms, dms, graphed = t_fwd[name]
            row.update(
                fwd_ms=ms,
                fwd_host_ms=hms,
                fwd_dev_ms=dms,
                fwd_graphed=graphed,
                eff_gbps=case.logical_bytes / (dms * 1e-3) / 1e9,
                pct_bw=100
                * case.logical_bytes
                / (dms * 1e-3)
                / 1e9
                / meta["dram_gbps"],
                eff_tflops=case.logical_flops / (dms * 1e-3) / 1e12,
                vram_fwd=peak_extra_bytes(fwd[name]),
                rel_err=rel_err(outs[name], ref_out),
                ulp=ulp_err(outs[name], ref_out, dtype, case.ulp_mode),
                host_share=hms / ms if ms else float("nan"),
                host_bound=bool(hms >= ms / 2),
            )
        if name in fwdbwd:
            ms, hms, dms, graphed = t_bwd[name]
            g = grads[name]
            row.update(
                fwdbwd_ms=ms,
                fwdbwd_host_ms=hms,
                fwdbwd_dev_ms=dms,
                fwdbwd_graphed=graphed,
                vram_fwdbwd=peak_extra_bytes(fwdbwd[name]),
                ulp_grad=max(ulp_err(g[k], ref_grads[k], dtype, "rms") for k in g),
                rel_err_grad=max(rel_err(g[k], ref_grads[k]) for k in g),
            )
        rows.append(row)
        log(
            f"{case.op:18s} {case.case:26s} {name:8s} "
            + (
                f"dev {row['fwd_dev_ms'] * 1e3:8.1f}us wall {row['fwd_ms'] * 1e3:8.1f}us "
                f"bw {row['pct_bw']:5.1f}% ulp {row['ulp']:7.2f}"
                if name in fwd
                else ""
            )
            + (
                f" | fb dev {row['fwdbwd_dev_ms'] * 1e3:8.1f}us ulp_g {row['ulp_grad']:7.2f}"
                if name in fwdbwd
                else ""
            )
            + (f" ERROR {row['error']}" if row["error"] else ""),
        )
    return rows
