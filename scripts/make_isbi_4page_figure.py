#!/usr/bin/env python3
"""Qualitative figure for the four-page ISBI manuscript.

Row 1 (BBBC039, seed 4 = median contact-aware seed): the strongest semantic
baseline (distance-transform watershed with h and smoothing selected on
validation for that seed) versus contact-aware decoding of the same image.
Row 2 (expanded NucleusSegData Huh7, seed 1 = median Standard-Contact seed):
Standard-Contact versus Gabor-Contact.

Images are those already fixed by the analysis scripts (nearest the median
per-image score of the median seed).  The displayed crop is chosen
mechanically as the window containing the most reference nucleus-nucleus
contact pixels, i.e. the region the method is designed for; whole-image
scores are printed in the panel titles.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breastseg.data import load_bbbc039, load_nucleusseg_extended  # noqa: E402
from breastseg.instance_metrics import bbbc039_reference_instances  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from posthoc_bbbc_decoding import semantic_instances  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "paper/fig_contact_examples.pdf"


def norm01(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99.5))
    return np.clip((image - low) / (high - low + 1e-7), 0, 1)


def colorize(labels):
    labels = np.asarray(labels, np.int64)
    out = np.zeros((*labels.shape, 3), float)
    for identifier in np.unique(labels)[1:]:
        hue = (identifier * 0.61803398875) % 1
        out[labels == identifier] = matplotlib.colors.hsv_to_rgb((hue, .70, 1.0))
    out[find_boundaries(labels, mode="inner")] = 0.0
    return out


def contact_map(labels):
    """Pixels whose 8-neighbourhood contains a different nonzero instance."""
    labels = np.asarray(labels, np.int64)
    contact = np.zeros(labels.shape, bool)
    padded = np.pad(labels, 1)
    h, w = labels.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            contact |= (labels > 0) & (nb > 0) & (nb != labels)
    return contact


def best_window(labels, size, stride=16):
    contact = contact_map(labels).astype(np.int64)
    integral = np.pad(contact.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_yx = -1, (0, 0)
    for y in range(0, labels.shape[0] - size + 1, stride):
        for x in range(0, labels.shape[1] - size + 1, stride):
            s = (integral[y + size, x + size] - integral[y, x + size]
                 - integral[y + size, x] + integral[y, x])
            if s > best:
                best, best_yx = s, (y, x)
    return best_yx


def rows_by_id(path):
    with open(path) as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def bbbc_row():
    seed = 4
    case_id = "IXMtest_A09_s1_w1CE70AD49-290D-4312-82E6-CDC717F32637"
    case = next(c for c in load_bbbc039(ROOT / "data/bbbc039", ("test",)) if c["id"] == case_id)
    import zipfile
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        raw_mask = cv2.imdecode(np.frombuffer(archive.read(f"masks/{case_id}.png"), np.uint8),
                                cv2.IMREAD_UNCHANGED)
    truth = bbbc039_reference_instances(raw_mask)
    semantic_run = RES / f"bbbc_primary_10seed_additional/vanilla_b7_s{seed}"
    semantic = cv2.imread(str(semantic_run / "predictions" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
    post = RES / "posthoc_bbbc_decoding"
    stem = f"bbbc_primary_10seed_additional__vanilla_b7_s{seed}"
    config = json.loads((post / f"{stem}.json").read_text())["settings"]["plain"]["selected"]["config"]
    assert config["decoder"] == "distance_watershed", config
    watershed = semantic_instances(semantic, "watershed", config["h"], config["sigma"])
    contact_run = RES / f"bbbc_contact_10seed/vanilla_contact_b7_s{seed}"
    contact = cv2.imread(str(contact_run / "predictions" / f"{case_id}_instances.tif"), cv2.IMREAD_UNCHANGED)
    f1_ws = float(rows_by_id(post / f"{stem}__plain__selected_test_per_image.csv")[case_id]["caicedo_af1"])
    f1_ct = float(rows_by_id(contact_run / "test_per_image.csv")[case_id]["caicedo_af1"])
    return dict(image=case["image"], truth=truth, a=watershed, b=contact,
                a_title=f"Semantic + watershed (obj.-F1 {f1_ws:.3f})",
                b_title=f"Contact-aware (obj.-F1 {f1_ct:.3f})",
                label="BBBC039", size=208)


def nucleusseg_row():
    aggregate = json.loads((RES / "nucleusseg_extended_contact_10seed/aggregate.json").read_text())
    seed = aggregate["qualitative_figure"]["display_seed"]
    case_id = aggregate["qualitative_figure"]["cases"]["Huh7"]
    case = next(c for c in load_nucleusseg_extended(ROOT / "data/nucleusseg_extended") if c["id"] == case_id)
    preds, scores = {}, {}
    for model in ("standard_contact_b7", "gabor_contact_b7"):
        run = RES / f"nucleusseg_extended_contact_10seed/{model}_s{seed}"
        preds[model] = cv2.imread(str(run / "predictions" / f"{case_id}_instances.tif"), cv2.IMREAD_UNCHANGED)
        scores[model] = float(rows_by_id(run / "test_per_image.csv")[case_id]["bilkent_nucleus_f1"])
    return dict(image=case["image"], truth=case["instances"], a=preds["standard_contact_b7"],
                b=preds["gabor_contact_b7"],
                a_title=f"Standard-Contact (F1 {scores['standard_contact_b7']:.3f})",
                b_title=f"Gabor-Contact (F1 {scores['gabor_contact_b7']:.3f})",
                label="NucleusSegData Huh7", size=320)


def bbbc_controls_row():
    """BBBC039, seed 4: watershed vs full-contour vs contact, two-head models decoded component-preserving."""
    from analyze_component_preserving import component_preserving
    seed = 4
    case_id = "IXMtest_A09_s1_w1CE70AD49-290D-4312-82E6-CDC717F32637"
    base = bbbc_row()
    from breastseg.instance_metrics import published_metrics as _pm
    _m = _pm(base["truth"], base["a"])
    panels = {"watershed": (base["a"], f"Semantic + watershed\nAJI {_m['aji']:.3f}, F1 {_m['caicedo_af1']:.3f}")}
    for name, study in (("contour", "bbbc_contour_10seed/vanilla_contour_b7"),
                        ("contact", "bbbc_contact_10seed/vanilla_contact_b7")):
        pred_dir = RES / f"{study}_s{seed}" / "predictions"
        fg = cv2.imread(str(pred_dir / f"{case_id}_foreground.png"), cv2.IMREAD_GRAYSCALE) > 0
        bd = cv2.imread(str(pred_dir / f"{case_id}_contact.png"), cv2.IMREAD_GRAYSCALE) > 0
        labels = component_preserving(fg, bd)
        from breastseg.instance_metrics import published_metrics
        met = published_metrics(base["truth"], labels)
        panels[name] = (labels, f"{'Full contour' if name == 'contour' else 'Contact'}\nAJI {met['aji']:.3f}, F1 {met['caicedo_af1']:.3f}")
    return base, panels


def main_controls():
    base, panels = bbbc_controls_row()
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.68))
    y, x = best_window(base["truth"], base["size"])
    s = base["size"]
    crop = (slice(y, y + s), slice(x, x + s))
    items = [(np.repeat(norm01(base["image"])[crop][..., None], 3, 2), "BBBC039 input"),
             (colorize(base["truth"][crop]), "Reference"),
             (colorize(panels["watershed"][0][crop]), panels["watershed"][1]),
             (colorize(panels["contour"][0][crop]), panels["contour"][1]),
             (colorize(panels["contact"][0][crop]), panels["contact"][1])]
    for ax, (panel, title) in zip(axes, items):
        ax.imshow(panel, interpolation="nearest"); ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.01, wspace=0.03)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print("wrote", OUT)


def main():
    rows = [bbbc_row(), nucleusseg_row()]
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.62))
    for r, row in enumerate(rows):
        y, x = best_window(row["truth"], row["size"])
        s = row["size"]
        crop = (slice(y, y + s), slice(x, x + s))
        panels = [(np.repeat(norm01(row["image"])[crop][..., None], 3, 2), f"{row['label']} input"),
                  (colorize(row["truth"][crop]), "Reference"),
                  (colorize(row["a"][crop]), row["a_title"]),
                  (colorize(row["b"][crop]), row["b_title"])]
        for c, (panel, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, interpolation="nearest")
            ax.set_title(title, fontsize=7, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        print(row["label"], "crop", (y, x, s))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.03, hspace=0.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main_controls() if "--controls" in sys.argv else main()
