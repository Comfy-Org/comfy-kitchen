"""Line figures for one op and one sweep, drawn from the measure JSON.

Panels: latency (device time bold, wall faint behind it: the gap is host
dispatch), speedup over ComfyUI's code on device time (forward solid,
forward+backward dashed), % of the measured DRAM ceiling, peak extra memory,
and error in ULP against fp64 (forward solid, gradients dashed). Arms keep one
Okabe-Ito color in every figure. Rows whose wall is mostly host dispatch get a
ring: their rate is a floor, not a measurement.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLORS = {
    "comfy": "#7F7F7F",
    "comfy_fixed": "#000000",
    "torch": "#7F7F7F",
    "compile": "#0072B2",
    "eager": "#CC79A7",
    "kitchen": "#D55E00",
    "lycoris": "#009E73",
}
ARM_MARK = {
    "comfy": "o",
    "comfy_fixed": "P",
    "torch": "o",
    "compile": "s",
    "eager": "v",
    "kitchen": "^",
    "lycoris": "D",
}
CEIL = "#C0392B"
RING = "#8E44AD"
GEOMETRIC = {
    "tokens",
    "batch",
    "rank",
    "block",
    "conv_spatial",
    "conv_batch",
    "merge_rank",
    "merge_block",
}
CATEGORICAL = {"factor", "merge_factor"}
SWEEP_LABEL = {
    "tokens": "tokens (batch 1)",
    "batch": "batch (x 4096 tokens)",
    "dim": "layer width (square)",
    "rank": "rank",
    "factor": "LoKr factor",
    "block": "OFT block size",
    "conv_spatial": "feature map side (2 x 640 ch)",
    "conv_batch": "batch (640 ch, 64x64)",
    "merge_dim": "weight width (square)",
    "merge_rank": "rank",
    "merge_block": "OFT block size",
    "merge_factor": "LoKr factor",
}

_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
}


def _baseline(rows):
    """ComfyUI's code where it ran; its minimally fixed form where it raised (OFT
    in 16-bit); plain torch where ComfyUI has no such adapter (IA3)."""
    ran = {r["arm"] for r in rows if r.get("fwd_dev_ms")}
    for arm in ("comfy", "comfy_fixed", "torch"):
        if arm in ran:
            return arm
    return "comfy"


def _arms(rows):
    order = list(ARM_COLORS)
    return sorted(
        {r["arm"] for r in rows}, key=lambda a: order.index(a) if a in order else 99
    )


def _series(rows, arm, key):
    pts = sorted((r["x"], r.get(key)) for r in rows if r["arm"] == arm)
    return [p[0] for p in pts], [float("nan") if p[1] is None else p[1] for p in pts]


def _xaxis(ax, rows, sweep):
    xs = sorted({r["x"] for r in rows})
    labels = {r["x"]: r["case"] for r in rows}
    if sweep in GEOMETRIC or sweep in ("dim", "merge_dim"):
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [labels[x] if sweep in CATEGORICAL else f"{x:g}" for x in xs], fontsize=7
    )
    ax.set_xlabel(SWEEP_LABEL.get(sweep, sweep))


def _positive(values):
    return any(v is not None and not math.isnan(v) and v > 0 for v in values)


def draw(rows, meta, sweep, out_png):
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.ravel()
    arms = _arms(rows)
    base = _baseline(rows)
    op = rows[0]["op"]

    ax = axes[0]
    drawn = []
    for arm in arms:
        c, m = ARM_COLORS.get(arm, "#000"), ARM_MARK.get(arm, "o")
        xs, wall = _series(rows, arm, "fwd_ms")
        _, dev = _series(rows, arm, "fwd_dev_ms")
        ax.plot(xs, [v * 1e3 for v in wall], color=c, alpha=0.25, linewidth=1.1)
        ax.plot(xs, [v * 1e3 for v in dev], color=c, marker=m, label=arm)
        drawn += dev
        hb = [
            (x, w * 1e3)
            for x, w, r in zip(
                xs,
                wall,
                sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["x"]),
                strict=False,
            )
            if r.get("host_bound")
        ]
        if hb:
            ax.scatter(
                [p[0] for p in hb],
                [p[1] for p in hb],
                s=110,
                facecolors="none",
                edgecolors=RING,
                linewidths=1.3,
                zorder=5,
            )
    if _positive(drawn):
        ax.set_yscale("log")
    ax.set_ylabel("forward µs (bold device, faint wall)")
    ax.set_title("Forward latency")
    ax.legend()

    ax = axes[1]
    drawn = []
    for arm in arms:
        c, m = ARM_COLORS.get(arm, "#000"), ARM_MARK.get(arm, "o")
        xs, wall = _series(rows, arm, "fwdbwd_ms")
        _, dev = _series(rows, arm, "fwdbwd_dev_ms")
        if not _positive(dev):
            continue
        ax.plot(xs, [v * 1e3 for v in wall], color=c, alpha=0.25, linewidth=1.1)
        ax.plot(xs, [v * 1e3 for v in dev], color=c, marker=m, label=arm)
        drawn += dev
    if _positive(drawn):
        ax.set_yscale("log")
        ax.legend()
    ax.set_ylabel("forward + backward µs (bold device, faint wall)")
    ax.set_title("Forward + backward latency")

    ax = axes[2]
    for phase, style in (("fwd", "-"), ("fwdbwd", "--")):
        ref = {r["x"]: r.get(f"{phase}_dev_ms") for r in rows if r["arm"] == base}
        for arm in arms:
            if arm == base:
                continue
            pts = sorted(
                (r["x"], r.get(f"{phase}_dev_ms")) for r in rows if r["arm"] == arm
            )
            ys = [ref.get(x) / v if v and ref.get(x) else float("nan") for x, v in pts]
            if not _positive(ys):
                continue
            ax.plot(
                [p[0] for p in pts],
                ys,
                style,
                color=ARM_COLORS.get(arm, "#000"),
                marker=ARM_MARK.get(arm, "o"),
                label=f"{arm} {'fwd' if phase == 'fwd' else 'fwd+bwd'}",
            )
    ax.axhline(1.0, color=CEIL, linestyle="--", linewidth=1.1)
    ax.set_ylabel(f"speedup over {base} (device time, >1 faster)")
    base_name = {
        "comfy": "ComfyUI",
        "comfy_fixed": "ComfyUI (dtype fix; as shipped it raises)",
        "torch": "plain torch",
    }
    ax.set_title(f"Speedup over {base_name.get(base, base)}")
    ax.legend()

    ax = axes[3]
    for arm in arms:
        xs, ys = _series(rows, arm, "pct_bw")
        ax.plot(
            xs,
            ys,
            color=ARM_COLORS.get(arm, "#000"),
            marker=ARM_MARK.get(arm, "o"),
            label=arm,
        )
    ax.axhline(100.0, color=CEIL, linestyle="--", linewidth=1.1)
    ax.annotate(
        f"measured DRAM {meta['dram_gbps']:.0f} GB/s",
        (0.01, 0.96),
        xycoords="axes fraction",
        color=CEIL,
        fontsize=8,
    )
    ax.set_ylabel("% of measured DRAM bandwidth (minimum bytes / device time)")
    ax.set_title("Bandwidth efficiency (above 100% = L2-resident)")
    ax.legend()

    ax = axes[4]
    for arm in arms:
        c, m = ARM_COLORS.get(arm, "#000"), ARM_MARK.get(arm, "o")
        xs, f = _series(rows, arm, "vram_fwd")
        _, fb = _series(rows, arm, "vram_fwdbwd")
        ax.plot(xs, [v / 2**20 for v in f], color=c, marker=m, label=f"{arm} fwd")
        if _positive(fb):
            ax.plot(
                xs,
                [v / 2**20 for v in fb],
                "--",
                color=c,
                marker=m,
                alpha=0.8,
                label=f"{arm} fwd+bwd",
            )
    ax.set_ylabel("peak memory above inputs (MiB)")
    ax.set_title("Peak memory")
    ax.legend(ncol=2)

    ax = axes[5]
    drawn = []
    for arm in arms:
        c, m = ARM_COLORS.get(arm, "#000"), ARM_MARK.get(arm, "o")
        xs, u = _series(rows, arm, "ulp")
        _, g = _series(rows, arm, "ulp_grad")
        ax.plot(xs, u, color=c, marker=m, label=f"{arm} output")
        drawn += u
        if _positive(g):
            ax.plot(
                xs, g, "--", color=c, marker=m, alpha=0.8, label=f"{arm} grads (worst)"
            )
            drawn += g
    ax.axhline(1.0, color=CEIL, linestyle="--", linewidth=1.1)
    if _positive(drawn):
        ax.set_yscale("log")
    ax.set_ylabel(f"max error, ULP of {meta['dtype']} vs fp64")
    ax.set_title("Precision (0.5 ULP = correctly rounded)")
    ax.legend(ncol=2)

    for a in axes:
        _xaxis(a, rows, sweep)
    params = rows[0]
    fixed = ", ".join(
        f"{k}={params[k]}"
        for k in ("dim", "rank", "factor", "block", "tokens")
        if k in params and k not in sweep
    )
    fig.suptitle(
        f"{op} — sweep: {SWEEP_LABEL.get(sweep, sweep)} ({fixed}) — {meta['device']} ({meta['arch']}, {meta['dtype']})\n"
        f"ceilings measured this run: DRAM {meta['dram_gbps']:.0f} GB/s (best of copy/read/triad), "
        f"matmul {meta['mamf_tflops']:.0f} TFLOP/s; L2 flushed before every timed call; "
        "rings = host dispatch over half the wall",
        fontweight="bold",
        fontsize=10,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(os.path.splitext(out_png)[0] + ".svg")
    plt.close(fig)
    print(f"figure -> {out_png}")
