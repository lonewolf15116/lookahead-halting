#!/usr/bin/env python3
"""Regenerate Figure 1 (rule-dependent operating points) from results/c1/test_report.json only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
NAME = "fig1_rule_dependent_operating_points"
BENCHMARKS = [("openbookqa", "OpenBookQA"), ("commonsense_qa", "CommonsenseQA"), ("hellaswag", "HellaSwag")]
METHODS = [("pi1-x", "π1-x", "o", "#1f5fa8"), ("native-pb", "native-pb", "s", "#c4571a")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(ROOT / "results" / "c1" / "test_report.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "figures"))
    args = ap.parse_args()

    with open(args.report) as f:
        rep = json.load(f)
    screened = rep["per_benchmark"]
    point = rep["secondary_point_rule"]["per_benchmark"]

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 10.5, "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, (bk, title) in zip(axes, BENCHMARKS):
        n = screened["pi1-x"][bk]["n"]
        ax.set_title(f"{title} (n = {n:,})", fontsize=10.5)
        ax.axvline(1.0, color="0.35", ls="--", lw=1.3, zorder=1)
        ax.grid(True, color="0.9", lw=0.8)
        for m, _, mk, col in METHODS:
            s, p = screened[m][bk], point[m][bk]
            ax.plot([s["loss_pp"], p["loss_pp"]], [s["saving"], p["saving"]], ls=":", lw=1.2,
                    color=col, alpha=0.5, zorder=2)
            ax.plot(s["loss_pp"], s["saving"], marker=mk, ms=8.5, color=col, ls="none", zorder=3)
            ax.plot(p["loss_pp"], p["saving"], marker=mk, ms=8.5, mfc="white", mec=col, mew=2,
                    ls="none", zorder=3)
        ax.set_xlim(-0.4, 1.9)
        ax.set_ylim(0, 0.45)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_xlabel("accuracy loss vs loop 4 (pp)", fontsize=10.5)
    axes[0].set_ylabel("gross loop saving (fraction of 4 loops)", fontsize=10.5)

    handles, labels = [], []
    for m, lab, mk, col in METHODS:
        handles.append(Line2D([], [], marker=mk, ms=8.5, color=col, ls="none"))
        labels.append(f"{lab}, bootstrap-screened")
        handles.append(Line2D([], [], marker=mk, ms=8.5, mfc="white", mec=col, mew=2, ls="none"))
        labels.append(f"{lab}, point-only")
    # order: pi1 screened, pi1 point, native screened, native point
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{NAME}.png", dpi=200, metadata={"Software": None})
    fig.savefig(out / f"{NAME}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})


if __name__ == "__main__":
    main()
