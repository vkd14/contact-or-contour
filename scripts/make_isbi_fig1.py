#!/usr/bin/env python3
"""Fig. 1 of the ISBI paper: one BBBC039 test crop, five segmentations with matched colours.

Selection rule (fixed, mechanical): the contact model's upper-median seed by mean test
object F1 under the component-preserving decoder (6th of 10, ascending), the test image
whose per-image object F1 for that seed is nearest that seed's median, and the 208-px
window with the most reference contact pixels.  The same seed index is shown for every
model.  Semantic + watershed is the Table 1 family (bbbc_semantic_ws_10seed); two-head
models use the component-preserving decoder; Cellpose uses its validation-selected
thresholds.  Each predicted object takes the colour of the reference nucleus holding
its majority; objects holding the majority of two or more reference nuclei (merges) are
red; objects without a majority reference are grey.
"""
import csv
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_bbbc_merge_split import majority_owner  # noqa: E402
from analyze_component_preserving import component_preserving  # noqa: E402
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import bbbc039_reference_instances, published_metrics  # noqa: E402
from make_isbi_4page_figure import best_window, norm01  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "paper/fig_contact_examples.pdf"
MERGE = np.array([0.86, 0.12, 0.10])
GREY = np.array([0.62, 0.62, 0.62])


def ref_colour(i):
    hue = 0.10 + 0.78 * ((i * 0.61803398875) % 1)       # avoid red hues
    return matplotlib.colors.hsv_to_rgb((hue, .55, .95))


def paint(pred, truth):
    pred = np.asarray(pred, np.int64); truth = np.asarray(truth, np.int64)
    out = np.zeros((*pred.shape, 3))
    owner = majority_owner(pred, truth)                   # pred -> reference
    ref_owner = majority_owner(truth, pred)               # reference -> pred
    held = np.bincount(ref_owner[ref_owner > 0], minlength=int(pred.max()) + 1)
    n_merge = 0
    for j in np.unique(pred)[1:]:
        if held[j] >= 2:
            colour = MERGE; n_merge += 1
        elif owner[j] > 0:
            colour = ref_colour(owner[j])
        else:
            colour = GREY
        out[pred == j] = colour
    out[find_boundaries(pred, mode="inner")] = 0.0
    return out, n_merge


def main():
    rows = [r for r in csv.DictReader(open(RES / "bbbc_component_preserving/per_image.csv")) if r["method"] == "contact"]
    f1 = {s: np.mean([float(r["caicedo_af1"]) for r in rows if int(r["seed"]) == s]) for s in range(10)}
    seed = sorted(f1, key=f1.get)[5]
    per = {r["id"]: float(r["caicedo_af1"]) for r in rows if int(r["seed"]) == seed}
    med = float(np.median(list(per.values())))
    case_id = min(per, key=lambda i: abs(per[i] - med))
    case = next(c for c in load_bbbc039(ROOT / "data/bbbc039", ("test",)) if c["id"] == case_id)
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        raw = cv2.imdecode(np.frombuffer(archive.read(f"masks/{case_id}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    truth = np.asarray(bbbc039_reference_instances(raw), np.int64)
    preds = {"Semantic + watershed": cv2.imread(str(RES / f"bbbc_semantic_ws_10seed_predictions/s{seed}/{case_id}_instances.tif"),
                                                cv2.IMREAD_UNCHANGED)}
    for name, study in (("Full contour", "bbbc_contour_10seed/vanilla_contour_b7"),
                        ("Contact", "bbbc_contact_10seed/vanilla_contact_b7")):
        d = RES / f"{study}_s{seed}" / "predictions"
        preds[name] = component_preserving(cv2.imread(str(d / f"{case_id}_foreground.png"), 0) > 0,
                                           cv2.imread(str(d / f"{case_id}_contact.png"), 0) > 0)
    preds["Cellpose"] = cv2.imread(str(RES / f"cellpose_bbbc_scratch/cellpose3_scratch_s{seed}/predictions/{case_id}_instances.tif"),
                                   cv2.IMREAD_UNCHANGED)
    size = 208
    y, x = best_window(truth, size)
    crop = (slice(y, y + size), slice(x, x + size))
    ref = np.zeros((*truth.shape, 3))
    for i in np.unique(truth)[1:]:
        ref[truth == i] = ref_colour(i)
    ref[find_boundaries(truth, mode="inner")] = 0
    panels = [(np.repeat(norm01(case["image"])[crop][..., None], 3, 2), "Input (BBBC039 test)"),
              (ref[crop], "Reference")]
    for name, p in preds.items():
        p = np.asarray(p, np.int64)
        m = published_metrics(truth, p)
        img, _ = paint(p[crop], truth[crop])
        _, merges = paint(p, truth)
        panels.append((img, f"{name}\nAJI {m['aji']:.3f}, {merges} merges"))
        print(name, round(m["aji"], 4), round(m["caicedo_af1"], 4), "merges", merges)
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest"); ax.set_title(title, fontsize=6.2, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.80, bottom=0.01, wspace=0.03)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("seed", seed, "image", case_id, "crop", (y, x), "wrote", OUT)


if __name__ == "__main__":
    main()
