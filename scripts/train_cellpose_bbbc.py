#!/usr/bin/env python3
"""Same-split Cellpose 3 baseline on BBBC039, trained from scratch.

Cellpose's released nucleus models were trained on collections that include
2018 Data Science Bowl images, which overlap BBBC039's source, so we do not use
them.  Instead a Cellpose 3 network (random initialisation, nuclei diameter
convention) is trained on the 100 official training images, its flow and
cell-probability thresholds are selected on the 50 validation images by mean
Caicedo object F1, and the 50 test images are decoded once.  Metrics use the
same evaluator as every other row.

Run with PYTHONPATH=.tools/cellpose_pkgs:src:scripts.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import models, train, utils

from breastseg.data import load_bbbc039
from breastseg.instance_metrics import published_metrics
from train_bbbc_contact import instance_annotations

ROOT = Path(__file__).resolve().parents[1]
FLOW = (0.4, 0.6, 0.8)
CELLPROB = (-1.0, 0.0, 1.0)
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    cases = load_bbbc039(ROOT / "data/bbbc039")
    annotations = instance_annotations(ROOT / "data/bbbc039")
    split = {s: [c for c in cases if c["split"] == s] for s in ("train", "val", "test")}
    images = {c["id"]: c["image"].astype(np.float32) for c in cases}
    labels = {c["id"]: annotations[c["id"]].astype(np.int32) for c in cases}
    train_x = [images[c["id"]] for c in split["train"]]
    train_y = [labels[c["id"]] for c in split["train"]]
    diameter = float(np.median(np.concatenate([utils.diameters(y)[1] for y in train_y if y.max() > 0])))
    study = ROOT / "results/cellpose_bbbc_scratch"
    study.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        out = study / f"cellpose3_scratch_s{seed}"
        if (out / "summary.json").exists():
            print("SKIP", out, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = models.CellposeModel(gpu=True, pretrained_model=False, model_type=None, diam_mean=17.0, nchan=2)
        start = time.time()
        path = train.train_seg(model.net, train_data=train_x, train_labels=train_y, channels=[0, 0],
                               normalize=True, n_epochs=args.epochs, save_path=str(out),
                               model_name=f"cellpose3_scratch_s{seed}", min_train_masks=1)
        path = path[0] if isinstance(path, (tuple, list)) else path
        model = models.CellposeModel(gpu=True, pretrained_model=str(path))
        train_seconds = time.time() - start

        def run(cases_, flow, cellprob):
            masks, _, _ = model.eval([images[c["id"]] for c in cases_], channels=[0, 0], diameter=diameter,
                                     flow_threshold=flow, cellprob_threshold=cellprob)
            return [published_metrics(labels[c["id"]], m) for c, m in zip(cases_, masks)]

        grid = []
        for flow in FLOW:
            for cellprob in CELLPROB:
                rows = run(split["val"], flow, cellprob)
                grid.append({"flow_threshold": flow, "cellprob_threshold": cellprob,
                             "val_caicedo_af1": float(np.mean([r["caicedo_af1"] for r in rows]))})
        best = max(grid, key=lambda g: g["val_caicedo_af1"])
        t0 = time.time()
        rows = run(split["test"], best["flow_threshold"], best["cellprob_threshold"])
        test_seconds = time.time() - t0
        with open(out / "test_per_image.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id"] + list(rows[0]), lineterminator="\n")
            writer.writeheader()
            for c, r in zip(split["test"], rows):
                writer.writerow({"id": c["id"], **r})
        summary = {"model": "Cellpose 3, random init", "seed": seed, "epochs": args.epochs,
                   "train_diameter_px": diameter, "validation_grid": grid, "selected": best,
                   "train_seconds": train_seconds, "test_seconds_50_images": test_seconds,
                   "mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS}}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "validation_grid"}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
