#!/usr/bin/env python3
"""Re-decode the TTA ("final setting") two-head models without the global nearest-marker decoder.

The original TTA grid (posthoc_bbbc_decoding.py) let validation pick the global
nearest-marker decoder, which hands marker-less foreground components to a
neighbour's label (the artifact the base rows already avoid).  Here the grid is the
component-preserving decoder plus contact-watershed (with marker-less components
kept), each at the same five contact thresholds.  Per seed: eight-fold TTA on the 50
validation and 50 test images, decoder chosen by mean validation object F1, test
decoded once.  Writes results/posthoc_tta_cp/<study>__s<seed>.csv/.json.
"""
import csv
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import posthoc_bbbc_decoding as phd  # noqa: E402
from analyze_component_preserving import component_preserving  # noqa: E402
from breastseg.core import Net  # noqa: E402
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import published_metrics  # noqa: E402
from train_bbbc_contact import instance_annotations  # noqa: E402

OUT = ROOT / "results/posthoc_tta_cp"
THRESHOLDS = phd.CONTACT_THRESHOLDS
GRID = ([{"decoder": "component_preserving", "threshold": t} for t in THRESHOLDS]
        + [{"decoder": "contact_watershed", "threshold": t} for t in THRESHOLDS])
STUDIES = [("bbbc_contour_10seed", "vanilla_contour_b7", None),
           ("bbbc_contact_10seed", "vanilla_contact_b7", None),
           ("bbbc_contact_native_10seed", "vanilla_contact_b7", (544, 704))]


def decode(p, config):
    if config["decoder"] == "component_preserving":
        return component_preserving(p[0] > .5, p[1] > config["threshold"])
    return phd.decode(p, config)


def score(job):
    truth, p, config = job
    return published_metrics(truth, decode(p, config))


def main():
    cv2.setNumThreads(1)
    cases = load_bbbc039(ROOT / "data/bbbc039", ("val", "test"))
    ann = instance_annotations(ROOT / "data/bbbc039")
    for c in cases:
        c["instances"] = ann[c["id"]]
    val = [c for c in cases if c["split"] == "val"]
    test = [c for c in cases if c["split"] == "test"]
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(16) as pool:
        for study, stem, pad in STUDIES:
            phd.NATIVE_PAD = pad
            for seed in range(10):
                dest = OUT / f"{study}__s{seed}"
                if dest.with_suffix(".json").exists():
                    print("SKIP", study, seed, flush=True)
                    continue
                run = f"{study}/{stem}_s{seed}"
                state = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location="cpu", weights_only=True)
                model = Net("vanilla_b7", pretrained=False, classes=2).cuda()
                model.load_state_dict(state["model_state_dict"], strict=True)
                vp = phd.probabilities(model, val, False, True)
                tp = phd.probabilities(model, test, False, True)
                del model; torch.cuda.empty_cache()
                grid = []
                for config in GRID:
                    rows = pool.map(score, [(c["instances"], vp[c["id"]], config) for c in val], chunksize=2)
                    grid.append({**config, "val_caicedo_af1": float(np.mean([r["caicedo_af1"] for r in rows]))})
                best = max(grid, key=lambda g: g["val_caicedo_af1"])
                config = {k: v for k, v in best.items() if not k.startswith("val_")}
                rows = pool.map(score, [(c["instances"], tp[c["id"]], config) for c in test], chunksize=2)
                with open(dest.with_suffix(".csv"), "w", newline="") as handle:
                    w = csv.DictWriter(handle, fieldnames=["id"] + list(rows[0]), lineterminator="\n")
                    w.writeheader()
                    for c, r in zip(test, rows):
                        w.writerow({"id": c["id"], **r})
                summary = {"run": run, "selected": config, "validation_grid": grid,
                           "test_mean": {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}}
                dest.with_suffix(".json").write_text(json.dumps(summary, indent=2))
                print("DONE", study, seed, config, {m: round(summary["test_mean"][m], 4)
                      for m in ("aji", "pq", "caicedo_af1", "dice1")}, flush=True)


if __name__ == "__main__":
    main()
