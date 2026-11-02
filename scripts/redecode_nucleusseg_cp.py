#!/usr/bin/env python3
"""Re-decode the expanded NucleusSegData contact and contour models with the component-preserving decoder.

The original runs (train_nucleusseg_contact.py) decoded with the global
nearest-marker decoder, which hands marker-less components to a neighbour.
Here each final-epoch checkpoint is decoded with the component-preserving decoder;
contact threshold and minimum size are chosen on the official training images with
the same grid and criterion as the original runs (pooled Bilkent nucleus F1, then
pixel F1); test images are decoded once (no TTA).  Writes
results/nucleusseg_cp/<model>_s<seed>.csv and summary.json with the paired
contact-vs-contour comparison per subset (seed-averaged per-image AJI and pooled
nucleus F1, 10,000 image bootstrap draws).
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_component_preserving import component_preserving  # noqa: E402
from breastseg.core import Net, features  # noqa: E402
from breastseg.data import load_nucleusseg_extended  # noqa: E402
from breastseg.instance_metrics import bilkent_metrics, published_metrics  # noqa: E402
from train_nucleusseg_contact import aggregate_bilkent  # noqa: E402

OUT = ROOT / "results/nucleusseg_cp"
RUNS = {"contact": ("nucleusseg_extended_contact_10seed", "standard_contact_b7"),
        "contour": ("nucleusseg_extended_contour_10seed", "standard_contour_b7")}
CONFIG = json.loads((ROOT / "configs/nucleusseg_extended_contour_10seed.json").read_text())


@torch.inference_mode()
def probabilities(model, cases):
    out = {}
    for c in cases:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(c["x"][None].cuda()).float().sigmoid()[0].cpu().numpy()
        h, w = c["instances"].shape
        out[c["id"]] = np.stack([cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR) for ch in p])
    return out


def rows_for(probs, cases, tau, min_size):
    rows = []
    for c in cases:
        p = probs[c["id"]]
        pred = component_preserving(p[0] > CONFIG["foreground_threshold"], p[1] > tau, min_size=min_size)
        b = bilkent_metrics(c["instances"], pred)
        rows.append({"id": c["id"], "subset": c["subset"], **published_metrics(c["instances"], pred),
                     **{k: b[k] for k in ("n_reference", "n_prediction", "n_one_to_one", "pixel_true_positive",
                                          "n_reference_pixels", "n_prediction_pixels")}})
    return rows


def main():
    # optional: <target> <seed> [<seed> ...] to run a subset; "summarize" only summarizes
    if sys.argv[1:2] == ["summarize"]:
        return summarize()
    only_target = sys.argv[1] if len(sys.argv) > 1 else None
    only_seeds = {int(x) for x in sys.argv[2:]} or set(range(10))
    cv2.setNumThreads(1)
    cases = load_nucleusseg_extended(ROOT / "data/nucleusseg_extended")
    for c in cases:
        c["x"] = features(c["image"], edge=False, size=CONFIG["image_size"])
    train = [c for c in cases if c["split"] == "train"]
    test = [c for c in cases if c["split"] == "test"]
    OUT.mkdir(parents=True, exist_ok=True)
    for target, (study, model_name) in RUNS.items():
        if only_target and target != only_target:
            continue
        for seed in sorted(only_seeds):
            dest = OUT / f"{target}_s{seed}.csv"
            if dest.exists():
                continue
            state = torch.load(ROOT / "checkpoints" / study / f"{model_name}_s{seed}" / "final.pt",
                               map_location="cpu", weights_only=True)
            model = Net("vanilla_b7", pretrained=False, classes=2).cuda().eval()
            model.load_state_dict(state["model_state_dict"], strict=True)
            ptrain, ptest = probabilities(model, train), probabilities(model, test)
            del model; torch.cuda.empty_cache()
            grid = []
            for tau in CONFIG["contact_thresholds"]:
                for ms in CONFIG["min_sizes"]:
                    s = aggregate_bilkent(rows_for(ptrain, train, tau, ms))
                    grid.append((s["nucleus_f1"], s["pixel_f1"], tau, ms))
            _, _, tau, ms = max(grid)
            rows = rows_for(ptest, test, tau, ms)
            with open(dest, "w", newline="") as handle:
                w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader(); w.writerows(rows)
            f1 = {sub: aggregate_bilkent([r for r in rows if r["subset"] == sub])["nucleus_f1"] for sub in ("Huh7", "HepG2")}
            print(target, seed, "tau", tau, "min", ms, {k: round(v, 4) for k, v in f1.items()}, flush=True)
    if not only_target:
        summarize()


def summarize():
    data = {t: [list(csv.DictReader(open(OUT / f"{t}_s{s}.csv"))) for s in range(10)] for t in RUNS}
    rng = np.random.default_rng(0)
    result = {"decoder": "component_preserving", "selection": "training images, pooled Bilkent nucleus F1",
              "subsets": {}}
    for sub in ("Huh7", "HepG2"):
        ids = [r["id"] for r in data["contact"][0] if r["subset"] == sub]
        n = len(ids)
        def mat(t, key):
            by = [{r["id"]: float(r[key]) for r in runs} for runs in data[t]]
            return np.array([[b[i] for i in ids] for b in by])
        entry = {"n_images": n}
        tp = {t: mat(t, "n_one_to_one") for t in RUNS}
        nr = {t: mat(t, "n_reference") for t in RUNS}
        npd = {t: mat(t, "n_prediction") for t in RUNS}
        def pooled(t, idx):
            return 2 * tp[t][:, idx].sum(1) / (nr[t][:, idx].sum(1) + npd[t][:, idx].sum(1))
        for t in RUNS:
            f = pooled(t, np.arange(n)); a = mat(t, "aji").mean(1)
            entry[t] = {"nucleus_f1": [float(f.mean()), float(f.std(ddof=1))],
                        "aji": [float(a.mean()), float(a.std(ddof=1))]}
        d = mat("contact", "aji").mean(0) - mat("contour", "aji").mean(0)
        draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
        entry["aji_diff"] = [float(d.mean()), [float(x) for x in np.percentile(draws, [2.5, 97.5])],
                             int((d > 0).sum()), n]
        fd = []
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            fd.append(pooled("contact", idx).mean() - pooled("contour", idx).mean())
        point = float(pooled("contact", np.arange(n)).mean() - pooled("contour", np.arange(n)).mean())
        entry["f1_diff"] = [point, [float(x) for x in np.percentile(fd, [2.5, 97.5])]]
        result["subsets"][sub] = entry
        print(sub, json.dumps(entry))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
