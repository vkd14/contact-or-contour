#!/usr/bin/env python3
"""Cellpose 3 from scratch on expanded NucleusSegData (agreed plan, 5 seeds).

Same protocol as our contact model on this benchmark: train on the 25 official
training images; select flow/cell-probability thresholds by pooled Bilkent
nucleus F1 on the training images (no validation set exists; AttentionBoost
also selected on training images); decode the 11 Huh7 and 25 HepG2 test
images once.  Run with PYTHONPATH=.tools/cellpose_pkgs:src:scripts.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import models, train, utils

from breastseg.data import load_nucleusseg_extended
from breastseg.instance_metrics import bilkent_metrics, published_metrics
from train_nucleusseg_contact import aggregate_bilkent

ROOT = Path(__file__).resolve().parents[1]
FLOW = (0.4, 0.6, 0.8)
CELLPROB = (-1.0, 0.0, 1.0)
ADDITIVE = ("n_reference", "n_prediction", "n_one_to_one", "pixel_true_positive",
            "n_reference_pixels", "n_prediction_pixels")


def gray(image):
    image = np.asarray(image)
    return (image[..., :3].mean(-1) if image.ndim == 3 else image).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    cases = load_nucleusseg_extended(ROOT / "data/nucleusseg_extended")
    train_cases = [c for c in cases if c["split"] == "train"]
    test_cases = [c for c in cases if c["split"] == "test"]
    tx = [gray(c["image"]) for c in train_cases]
    ty = [c["instances"].astype(np.int32) for c in train_cases]
    diameter = float(np.median(np.concatenate([utils.diameters(y)[1] for y in ty if y.max() > 0])))
    study = ROOT / "results/cellpose_nucleusseg_scratch"
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
        path = train.train_seg(model.net, train_data=tx, train_labels=ty, channels=[0, 0], normalize=True,
                               n_epochs=args.epochs, save_path=str(out), model_name=f"cellpose3_scratch_s{seed}",
                               min_train_masks=1)
        path = path[0] if isinstance(path, (tuple, list)) else path
        model = models.CellposeModel(gpu=True, pretrained_model=str(path))

        def run(cs, flow, cellprob):
            masks, _, _ = model.eval([gray(c["image"]) for c in cs], channels=[0, 0], diameter=diameter,
                                     flow_threshold=flow, cellprob_threshold=cellprob)
            return masks

        grid = []
        for flow in FLOW:
            for cellprob in CELLPROB:
                masks = run(train_cases, flow, cellprob)
                rows = [bilkent_metrics(c["instances"], m) for c, m in zip(train_cases, masks)]
                grid.append({"flow_threshold": flow, "cellprob_threshold": cellprob,
                             "train_nucleus_f1": aggregate_bilkent(rows)["nucleus_f1"]})
        best = max(grid, key=lambda g: g["train_nucleus_f1"])
        masks = run(test_cases, best["flow_threshold"], best["cellprob_threshold"])
        rows = []
        for c, m in zip(test_cases, masks):
            b = bilkent_metrics(c["instances"], m)
            rows.append({"id": c["id"], "subset": c["subset"], **published_metrics(c["instances"], m),
                         **{f"bilkent_{k}": v for k, v in b.items()}})
        with open(out / "test_per_image.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        subsets = {}
        for subset in ("Huh7", "HepG2"):
            sub = [r for r in rows if r["subset"] == subset]
            pooled = aggregate_bilkent([{k: r[f"bilkent_{k}"] for k in ADDITIVE} for r in sub])
            subsets[subset] = {"nucleus_f1": pooled["nucleus_f1"], "aji": float(np.mean([r["aji"] for r in sub])),
                               "pq": float(np.mean([r["pq"] for r in sub]))}
        summary = {"model": "Cellpose 3, random init", "seed": seed, "epochs": args.epochs,
                   "train_diameter_px": diameter, "training_grid": grid, "selected": best,
                   "seconds": time.time() - start, "test": subsets}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "training_grid"}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
