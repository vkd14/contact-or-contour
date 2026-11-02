#!/usr/bin/env python3
"""Fig. 2: error profile on BBBC039 test (ten seeds, frozen stratified protocol).

Grouped bars per method for touching-nucleus merges, small-nucleus misses and
splits (percent of reference nuclei in each stratum), with seed-SD error bars.
Source: results/bbbc_stratified_comparison.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/fig_error_profile.pdf"
METHODS = [("semantic_watershed", "Semantic +\nwatershed"), ("full_contour", "Full\ncontour"),
           ("contact", "Contact"), ("cellpose", "Cellpose 3")]
PANELS = [("touching_merged_pct", "Touching merged (%)"), ("small_missed_pct", "Small unmatched (%)"),
          ("all_split_pct", "Split (%)")]


def main():
    data = json.loads((ROOT / "results/bbbc_stratified_comparison.json").read_text())["methods"]
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.7))
    ink = "#2a2a2a"
    colors = {"semantic_watershed": "#c9d0da", "full_contour": "#8a98ab", "contact": "#1f5fa8", "cellpose": "#2a2a2a"}
    labels = {"semantic_watershed": "Semantic + watershed", "full_contour": "Full contour",
              "contact": "Contact", "cellpose": "Cellpose 3"}
    for ax, (key, title) in zip(axes, PANELS):
        means = [data[m][key]["mean"] for m, _ in METHODS]
        sds = [data[m][key]["seed_sd"] for m, _ in METHODS]
        x = np.arange(len(METHODS))
        bars = ax.bar(x, means, yerr=sds, color=[colors[m] for m, _ in METHODS], width=0.78, linewidth=0,
                      error_kw={"lw": 0.6, "capsize": 1.5, "ecolor": "#555555"})
        top = max(m + s for m, s in zip(means, sds))
        for xi, v, sd in zip(x, means, sds):
            ax.text(xi, v + sd + top * 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=5.8, color=ink)
        ax.set_title(title, fontsize=6.8, color=ink, pad=2)
        ax.set_xticks([])
        ax.set_ylim(0, top * 1.22)
        ax.tick_params(width=0.5, length=2, colors=ink, labelsize=5.5)
        ax.grid(axis="y", color="#e3e3e3", lw=0.4); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m, _ in METHODS]
    fig.legend(handles, [labels[m] for m, _ in METHODS], loc="lower center", ncol=4, fontsize=6.2,
               frameon=False, handlelength=1.0, columnspacing=0.9)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.88, wspace=0.32)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
