#!/usr/bin/env python3
"""Contact-aware Cellpose: Cellpose 3 with an auxiliary nucleus-nucleus contact head.

Everything is Cellpose 3's own train_seg (data handling, augmentation, flows,
schedule, optimiser, epochs) with exactly three additions, applied by patching:
  1. labels_to_flows appends the contact target (interfaces between different
     instances, 1-px dilation inside the foreground) BEFORE the two flow
     channels, because Cellpose's augmentation rotates the last two channels;
  2. the network has four outputs (flowY, flowX, cellprob, contact) instead of
     three; Cellpose's standard decoder reads only the first three;
  3. the loss adds 0.5 x [0.5 soft Dice + 0.5 positive-weighted BCE] on the
     contact channel, the same contact term as our UNet++ model.
At inference, Cellpose masks are decoded as usual; optionally, a Cellpose object
is split when removing predicted contacts (p_c > tau) leaves >= 2 markers of
>= 16 px, assigning its pixels to the nearest marker.  Flow/cell-probability
thresholds and tau (including "no split") are selected on the 50 validation
images by mean Caicedo object F1; the 50 test images are decoded once.
Run with PYTHONPATH=.tools/cellpose_pkgs:src:scripts.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import measure, morphology
from torch.nn import functional as F

import cellpose.dynamics as cp_dynamics
import cellpose.models as cp_models
import cellpose.train as cp_train
from cellpose import models, train, utils
from cellpose.resnet_torch import CPnet

from breastseg.data import load_bbbc039
from breastseg.instance_metrics import published_metrics, remap_labels
from train_bbbc_contact import instance_annotations, touching_contact_target

ROOT = Path(__file__).resolve().parents[1]
FLOW = (0.4, 0.6, 0.8)
CELLPROB = (-1.0, 0.0, 1.0)
TAUS = (None, 0.3, 0.5, 0.7)
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")
CONTACT_WEIGHT = 0.5

_original_labels_to_flows = cp_dynamics.labels_to_flows


def labels_to_flows_with_contact(labels, *args, **kwargs):
    flows = _original_labels_to_flows(labels, *args, **kwargs)
    out = []
    for flow in flows:  # [labels, prob, flowY, flowX]
        contact = touching_contact_target(flow[0].astype(np.int64), radius=1).astype(np.float32)
        out.append(np.concatenate([flow[:2], contact[None], flow[2:]], 0))
    return out


def soft_dice(p, t):
    p, t = p.flatten(1), t.flatten(1)
    return (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()


def loss_with_contact(lbl, y, device):
    # lbl: [prob, contact, flowY, flowX]; y: [flowY, flowX, cellprob, contact]
    veci = 5. * torch.from_numpy(lbl[:, -2:]).to(device)
    loss = F.mse_loss(y[:, :2], veci) / 2.
    loss = loss + F.binary_cross_entropy_with_logits(y[:, 2], torch.from_numpy(lbl[:, 0] > .5).to(device).float())
    contact = torch.from_numpy(lbl[:, 1] > .5).to(device).float()
    ratio = contact.mean()
    pos = ((1 - ratio) / ratio.clamp_min(1e-7)).clamp(1, 50) if ratio > 0 else torch.ones((), device=device)
    contact_loss = .5 * soft_dice(y[:, 3].sigmoid()[:, None], contact[:, None]) + \
        .5 * F.binary_cross_entropy_with_logits(y[:, 3], contact, pos_weight=pos)
    return loss + CONTACT_WEIGHT * contact_loss


_stash = {}
_original_run_net = cp_models.run_net


def run_net_stash(*args, **kwargs):
    yf, styles = _original_run_net(*args, **kwargs)
    _stash["yf"] = yf
    return yf, styles


def contact_split(masks, contact, tau, min_size=16):
    if tau is None:
        return masks
    masks = np.asarray(masks, np.int64)
    out = masks.copy()
    next_label = int(masks.max()) + 1
    keep = (contact <= tau) & (masks > 0)
    for region in measure.regionprops(masks):
        sl = region.slice
        inside = masks[sl] == region.label
        markers = measure.label(keep[sl] & inside, connectivity=1)
        markers = morphology.remove_small_objects(markers, max_size=min_size - 1)
        ids = np.unique(markers)[1:]
        if len(ids) < 2:
            continue
        _, idx = ndi.distance_transform_edt(markers == 0, return_indices=True)
        assigned = markers[tuple(idx)]
        block = out[sl]
        for k, marker_id in enumerate(ids):
            sel = inside & (assigned == marker_id)
            block[sel] = region.label if k == 0 else next_label
            if k:
                next_label += 1
        out[sl] = block
    return remap_labels(out)


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
    study = ROOT / "results/contact_cellpose_bbbc_scratch"
    study.mkdir(parents=True, exist_ok=True)
    cp_dynamics.labels_to_flows = labels_to_flows_with_contact
    cp_train.dynamics.labels_to_flows = labels_to_flows_with_contact
    cp_train._loss_fn_seg = loss_with_contact
    cp_models.run_net = run_net_stash
    for seed in args.seeds:
        out = study / f"contact_cellpose3_scratch_s{seed}"
        if (out / "summary.json").exists():
            print("SKIP", out, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = models.CellposeModel(gpu=True, pretrained_model=False, model_type=None, diam_mean=17.0, nchan=2)
        torch.manual_seed(seed); np.random.seed(seed)  # re-seed so the 4-output net is initialised like vanilla
        model.net = CPnet(model.nbase, 4, sz=3, mkldnn=False, max_pool=True, diam_mean=17.0).to(model.device)
        model.nclasses = 4
        start = time.time()
        path = train.train_seg(model.net, train_data=train_x, train_labels=train_y, channels=[0, 0],
                               normalize=True, n_epochs=args.epochs, save_path=str(out),
                               model_name=f"contact_cellpose3_scratch_s{seed}", min_train_masks=1)
        path = path[0] if isinstance(path, (tuple, list)) else path
        train_seconds = time.time() - start
        model.net.load_model(str(path), device=model.device)
        model.net.eval()

        def predict(cases_, flow, cellprob):
            outs = []
            for c in cases_:
                m, _, _ = model.eval(images[c["id"]], channels=[0, 0], diameter=diameter,
                                     flow_threshold=flow, cellprob_threshold=cellprob)
                yf = _stash["yf"]
                yf = yf[0] if yf.ndim == 4 else yf
                contact = 1 / (1 + np.exp(-cv2.resize(yf[..., 3], (m.shape[1], m.shape[0]),
                                                      interpolation=cv2.INTER_LINEAR)))
                outs.append((m, contact))
            return outs

        grid = []
        cache = {}
        for flow in FLOW:
            for cellprob in CELLPROB:
                preds = predict(split["val"], flow, cellprob)
                cache[(flow, cellprob)] = preds
                for tau in TAUS:
                    rows = [published_metrics(labels[c["id"]], contact_split(m, ct, tau))
                            for c, (m, ct) in zip(split["val"], preds)]
                    grid.append({"flow_threshold": flow, "cellprob_threshold": cellprob, "tau": tau,
                                 "val_caicedo_af1": float(np.mean([r["caicedo_af1"] for r in rows]))})
        best = max(grid, key=lambda g: g["val_caicedo_af1"])
        best_nosplit = max([g for g in grid if g["tau"] is None], key=lambda g: g["val_caicedo_af1"])
        t0 = time.time()
        test_preds = predict(split["test"], best["flow_threshold"], best["cellprob_threshold"])
        results = {}
        for label, cfg, preds in (("selected", best, test_preds),
                                  ("no_split", best_nosplit,
                                   predict(split["test"], best_nosplit["flow_threshold"],
                                           best_nosplit["cellprob_threshold"]))):
            rows = [published_metrics(labels[c["id"]], contact_split(m, ct, cfg["tau"]))
                    for c, (m, ct) in zip(split["test"], preds)]
            with open(out / f"test_{label}_per_image.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id"] + list(rows[0]), lineterminator="\n")
                writer.writeheader()
                for c, r in zip(split["test"], rows):
                    writer.writerow({"id": c["id"], **r})
            results[label] = {"config": cfg, "mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS}}
        (out / "predictions").mkdir(exist_ok=True)
        for c, (m, ct) in zip(split["test"], test_preds):
            cv2.imwrite(str(out / "predictions" / f"{c['id']}_instances.tif"),
                        contact_split(m, ct, best["tau"]).astype(np.uint16))
        summary = {"model": "Cellpose 3 + contact head, random init", "seed": seed, "epochs": args.epochs,
                   "train_diameter_px": diameter, "validation_grid": grid, "train_seconds": train_seconds,
                   "test_seconds": time.time() - t0, **results}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "validation_grid"}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
