"""Draw every op x sweep figure from the measure JSON; no GPU needed.

python -m samples.adapter_bench.plot.plot_all --dir samples/adapter_bench/out/sm_120
"""

import argparse
from pathlib import Path

from ..core.io import load_rows
from .figures import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", type=Path, required=True, help="one device's output directory"
    )
    args = ap.parse_args()
    for path in sorted(args.dir.glob("*.json")):
        meta, rows = load_rows(path)
        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r["op"], r["sweep"]), []).append(r)
        for (op, sweep), sub in groups.items():
            draw(sub, meta, sweep, str(args.dir / "figs" / f"{op}__{sweep}.png"))


if __name__ == "__main__":
    main()
