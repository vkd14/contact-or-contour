#!/usr/bin/env python3
"""Forest plot of design-choice effects on AJI (results/design_effects.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/fig_effects.pdf"
SHORT = {
    "Decoder: tuned watershed vs. connected components": "Decoder: watershed vs. CC",
    "Representation: Cellpose vs. contact model": "Cellpose vs. base contact$^\\dagger$",
    "Representation: Cellpose vs. best contact model (searched recipe, native + TTA)": "Cellpose vs. best contact$^\\dagger$",
    "Boundary supervision: full contour vs. semantic + watershed": "Contour head vs. none$^\\dagger$",
    "Target: contact vs. full contour": "Contact vs. contour target",
    "Target: contact vs. full contour (TTA, same decoder grid)": "Contact vs. contour target (TTA)",
    "Resolution: native vs. 512^2 (both with TTA)": "Native vs. $512^2$ resolution$^\\dagger$",
    "Inference: TTA + selected decoder": "TTA + decoder ($512^2$ contact)$^\\dagger$",
    "Foreground: intact vs. contacts carved out": "Intact vs. carved foreground$^\\dagger$",
    "Contact head added to Cellpose": "Contact head added to Cellpose",
    "Target: contact vs. full contour (NucleusSegData Huh7)": "Contact vs. contour (NSD Huh7)",
    "Target: contact vs. full contour (NucleusSegData HepG2)": "Contact vs. contour (NSD HepG2)",
    "Target: contact vs. full contour (NucleusSegData Huh7, widened grid + TTA)": "Contact vs. contour (NSD Huh7$^\\ast$)",
    "Target: contact vs. full contour (NucleusSegData HepG2, widened grid + TTA)": "Contact vs. contour (NSD HepG2$^\\ast$)",
    "Boundary supervision: full contour vs. semantic + watershed (DSB external)": "Contour head vs. none (external)$^\\dagger$",
    "Target: contact vs. full contour (DSB external)": "Contact vs. contour (external)",
    "Decoder: tuned watershed vs. connected components (DSB external)": "Decoder: watershed vs. CC (external)",
}
ORDER = [
    "Target: contact vs. full contour",
    "Target: contact vs. full contour (TTA, same decoder grid)",
    "Foreground: intact vs. contacts carved out",
    "Contact head added to Cellpose",
    "Decoder: tuned watershed vs. connected components",
    "Boundary supervision: full contour vs. semantic + watershed",
    "Inference: TTA + selected decoder",
    "Resolution: native vs. 512^2 (both with TTA)",
    "Representation: Cellpose vs. contact model",
    "Representation: Cellpose vs. best contact model (searched recipe, native + TTA)",
    "Target: contact vs. full contour (NucleusSegData Huh7)",
    "Target: contact vs. full contour (NucleusSegData HepG2)",
    "Target: contact vs. full contour (NucleusSegData Huh7, widened grid + TTA)",
    "Target: contact vs. full contour (NucleusSegData HepG2, widened grid + TTA)",
    "Target: contact vs. full contour (DSB external)",
    "Decoder: tuned watershed vs. connected components (DSB external)",
    "Boundary supervision: full contour vs. semantic + watershed (DSB external)",
]
assert set(ORDER) == set(SHORT)


def main():
    effects = {e["label"]: e for e in json.loads((ROOT / "results/design_effects.json").read_text())["effects"]}
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ink = "#2a2a2a"
    y = list(range(len(ORDER)))[::-1]
    for yi, label in zip(y, ORDER):
        e = effects[label]
        d, (lo, hi) = e["difference"] * 100, [v * 100 for v in e["ci95"]]
        nsd = "NucleusSegData" in label or "DSB" in label
        color = "#8a98ab" if nsd else ("#b5443b" if d < 0 and lo < 0 and hi < 0 else "#1f5fa8")
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.2, solid_capstyle="butt")
        composite = "dagger" in SHORT[label]
        ax.plot(d, yi, "o", ms=3.6, color="white" if composite else color,
                markeredgecolor=color, markeredgewidth=0.9 if composite else 0.4)
        ax.text(max(hi, 0) + 0.4, yi, f"{d:+.1f}", va="center", fontsize=6.0, color=ink)
    ax.axvline(0, color=ink, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[l] for l in ORDER], fontsize=6.2)
    ax.set_xscale("symlog", linthresh=2.0)
    ax.set_xticks([-2, -1, 0, 1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["-2", "-1", "0", "1", "2", "5", "10", "20", "40"])
    ax.set_xlim(-3, 60)
    ax.set_xlabel("$\\Delta$AJI (points), 95% CI", color=ink)
    ax.axhline(6.5, color="#cccccc", lw=0.5, ls="--")
    ax.tick_params(width=0.5, length=2, colors=ink)
    ax.grid(axis="x", color="#ececec", lw=0.4); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.50, right=0.99, bottom=0.135, top=0.99)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
