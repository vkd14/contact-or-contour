#!/usr/bin/env python3
"""Interface-weighted Cellpose (IW-Cellpose) on BBBC039.

Identical to the from-scratch Cellpose 3 baseline (scripts/train_cellpose_bbbc.py:
same network, data, augmentation, schedule, optimiser, epochs, decoder grid and
validation selection), except that Cellpose's own flow MSE and cell-probability
BCE are weighted per pixel by w = 1 + W * contact, where contact marks
nucleus-nucleus interfaces (1-px dilation inside the foreground; carried through
Cellpose's augmentation as an extra label channel placed before the two flow
channels).  No extra output and no change at inference.

  --mode search : train seeds on the 100 training images, report VALIDATION
                  object F1 only (test images are never decoded)
  --mode final  : 10 seeds with the validation-selected W, test decoded once
Run with PYTHONPATH=.tools/cellpose_pkgs:src:scripts.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import cellpose.dynamics as cp_dynamics
import cellpose.train as cp_train
from cellpose import models, train, utils

from breastseg.data import load_bbbc039
from breastseg.instance_metrics import published_metrics
from train_bbbc_contact import instance_annotations
from skimage import morphology, segmentation
import cellpose.dynamics as _dyn
from train_contact_cellpose_bbbc import labels_to_flows_with_contact

_orig_l2f = _dyn.labels_to_flows


def labels_to_flows_with_contour(labels, *args, **kwargs):
    """Control: full-contour channel (inner boundary dilated 1 px, inside foreground) in the contact slot."""
    flows = _orig_l2f(labels, *args, **kwargs)
    out = []
    for flow in flows:
        lab = flow[0].astype(np.int64)
        contour = morphology.dilation(segmentation.find_boundaries(lab, mode="inner", connectivity=1),
                                      morphology.disk(1)) & (lab > 0)
        out.append(np.concatenate([flow[:2], contour.astype(np.float32)[None], flow[2:]], 0))
    return out

ROOT = Path(__file__).resolve().parents[1]
FLOW = (0.4, 0.6, 0.8)
CELLPROB = (-1.0, 0.0, 1.0)
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")
WEIGHT = {"W": 0.0, "target": "contact", "rng": None}


def interface_weighted_loss(lbl, y, device):
    # lbl: [prob, contact, flowY, flowX]; y: [flowY, flowX, cellprob]
    fg = torch.from_numpy(lbl[:, 0] > .5).to(device)
    region = torch.from_numpy(lbl[:, 1] > .5).to(device) & fg  # keep weights inside foreground after augmentation
    if WEIGHT["target"] == "random":
        # same number of weighted pixels as the contact target, drawn uniformly from foreground (separate RNG)
        contact = region
        n = int(contact.sum())
        idx = torch.nonzero(fg.flatten(), as_tuple=False).flatten()
        region = torch.zeros_like(fg.flatten())
        if n > 0 and idx.numel() > 0:
            pick = torch.from_numpy(WEIGHT["rng"].choice(idx.numel(), size=min(n, idx.numel()), replace=False)).to(device)
            region[idx[pick]] = True
        region = region.view_as(fg)
    scale = WEIGHT["W"]
    if WEIGHT["target"] == "contour" and region.any():
        # match the excess weight mass of the contact weighting in this batch
        scale = WEIGHT["W"] * float(WEIGHT["contact_count"](lbl)) / float(region.sum())
    w = 1.0 + scale * region.float()
    veci = 5. * torch.from_numpy(lbl[:, -2:]).to(device)
    flow = ((y[:, :2] - veci) ** 2).mean(1)
    loss = (w * flow).sum() / w.sum() / 2.
    bce = F.binary_cross_entropy_with_logits(y[:, -1], torch.from_numpy(lbl[:, 0] > .5).to(device).float(),
                                             reduction="none")
    return loss + (w * bce).sum() / w.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("search", "final"), required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--target", choices=("contact", "random", "contour"), default="contact")
    args = parser.parse_args()
    WEIGHT["W"] = args.weight
    WEIGHT["target"] = args.target
    WEIGHT["rng"] = np.random.default_rng(12345)
    l2f = labels_to_flows_with_contour if args.target == "contour" else labels_to_flows_with_contact
    if args.target == "contour":
        # contact pixel count of the same augmented batch is not available, so match the expected excess mass
        # using the dataset-level contact/contour pixel ratio computed on the training labels
        WEIGHT["contact_count"] = None
    cp_dynamics.labels_to_flows = l2f
    cp_train.dynamics.labels_to_flows = l2f
    cp_train._loss_fn_seg = interface_weighted_loss
    splits = ("train", "val") if args.mode == "search" else ("train", "val", "test")
    cases = load_bbbc039(ROOT / "data/bbbc039", splits)
    annotations = instance_annotations(ROOT / "data/bbbc039", case_ids={c["id"] for c in cases})
    split = {s: [c for c in cases if c["split"] == s] for s in splits}
    images = {c["id"]: c["image"].astype(np.float32) for c in cases}
    labels = {c["id"]: annotations[c["id"]].astype(np.int32) for c in cases}
    train_x = [images[c["id"]] for c in split["train"]]
    train_y = [labels[c["id"]] for c in split["train"]]
    diameter = float(np.median(np.concatenate([utils.diameters(y)[1] for y in train_y if y.max() > 0])))
    if args.target == "contour":
        from train_bbbc_contact import touching_contact_target, full_contour_target
        n_contact = sum(int(touching_contact_target(y.astype(np.int64), radius=1).sum()) for y in train_y)
        n_contour = sum(int(full_contour_target(y.astype(np.int64), radius=1).sum()) for y in train_y)
        ratio = n_contact / max(n_contour, 1)
        WEIGHT["contact_count"] = lambda lbl, r=ratio: r * float((lbl[:, 1] > .5).sum())
        print(f"contour control: contact/contour pixel ratio {ratio:.3f}", flush=True)
    tag = f"W{args.weight:g}" + ("" if args.target == "contact" else f"_{args.target}")
    study = ROOT / ("results/iw_cellpose_search" if args.mode == "search" else "results/iw_cellpose_final")
    study.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        out = study / f"iw_cellpose3_{tag}_s{seed}"
        if (out / "summary.json").exists():
            print("SKIP", out, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = models.CellposeModel(gpu=True, pretrained_model=False, model_type=None, diam_mean=17.0, nchan=2)
        start = time.time()
        path = train.train_seg(model.net, train_data=train_x, train_labels=train_y, channels=[0, 0],
                               normalize=True, n_epochs=args.epochs, save_path=str(out),
                               model_name=f"iw_cellpose3_{tag}_s{seed}", min_train_masks=1)
        path = path[0] if isinstance(path, (tuple, list)) else path
        train_seconds = time.time() - start
        model = models.CellposeModel(gpu=True, pretrained_model=str(path))

        saved = {}

        def run(cs, flow, cellprob, keep=False):
            masks, _, _ = model.eval([images[c["id"]] for c in cs], channels=[0, 0], diameter=diameter,
                                     flow_threshold=flow, cellprob_threshold=cellprob)
            if keep:
                saved.update({c["id"]: m for c, m in zip(cs, masks)})
            return [published_metrics(labels[c["id"]], m) for c, m in zip(cs, masks)]

        grid = []
        for flow in FLOW:
            for cellprob in CELLPROB:
                rows = run(split["val"], flow, cellprob)
                grid.append({"flow_threshold": flow, "cellprob_threshold": cellprob,
                             **{f"val_{m}": float(np.mean([r[m] for r in rows])) for m in METRICS}})
        best = max(grid, key=lambda g: g["val_caicedo_af1"])
        summary = {"model": "IW-Cellpose (interface-weighted flow/cellprob loss), random init", "W": args.weight,
                   "seed": seed, "epochs": args.epochs, "train_diameter_px": diameter, "validation_grid": grid,
                   "selected": best, "target": args.target, "train_seconds": train_seconds}
        if args.mode == "final":
            rows = run(split["test"], best["flow_threshold"], best["cellprob_threshold"], keep=True)
            import cv2
            (out / "predictions").mkdir(exist_ok=True)
            for cid, m in saved.items():
                cv2.imwrite(str(out / "predictions" / f"{cid}_instances.tif"), np.asarray(m).astype(np.uint16))
            with open(out / "test_per_image.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id"] + list(rows[0]), lineterminator="\n")
                writer.writeheader()
                for c, r in zip(split["test"], rows):
                    writer.writerow({"id": c["id"], **r})
            summary["mean"] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "validation_grid"}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
