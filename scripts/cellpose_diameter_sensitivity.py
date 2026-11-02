#!/usr/bin/env python3
"""Sensitivity of the scratch Cellpose rows to the inference diameter.

The training scripts passed diameter = median over training nuclei of sqrt(area)
(the second output of cellpose.utils.diameters, which is not a diameter), whereas
Cellpose's training-time rescaling uses its own diameter convention
(median sqrt(area) / (sqrt(pi)/2), the first output).  Here each saved Cellpose
network is re-evaluated with the convention-consistent diameter (pooled over the
same training nuclei); flow/cell-probability thresholds are re-selected with the
original grid and criterion (BBBC039: validation mean object F1; NucleusSegData:
training pooled Bilkent nucleus F1) and test images are decoded once.
Writes results/cellpose_diameter_sensitivity/summary.json.
Run with PYTHONPATH=.tools/cellpose_pkgs:src:scripts.
"""
import json
import math
from pathlib import Path

import numpy as np
from cellpose import models, utils

from breastseg.data import load_bbbc039, load_nucleusseg_extended
from breastseg.instance_metrics import bilkent_metrics, published_metrics
from train_bbbc_contact import instance_annotations
from train_cellpose_nucleusseg import ADDITIVE, gray
from train_nucleusseg_contact import aggregate_bilkent

ROOT = Path(__file__).resolve().parents[1]
FLOW = (0.4, 0.6, 0.8)
CELLPROB = (-1.0, 0.0, 1.0)
OUT = ROOT / "results/cellpose_diameter_sensitivity"


def pooled_diameters(ys):
    root_area = np.concatenate([utils.diameters(y)[1] for y in ys if y.max() > 0])
    used = float(np.median(root_area))
    return used, used / (math.sqrt(math.pi) / 2)


def bbbc():
    cases = load_bbbc039(ROOT / "data/bbbc039")
    ann = instance_annotations(ROOT / "data/bbbc039")
    split = {s: [c for c in cases if c["split"] == s] for s in ("train", "val", "test")}
    img = {c["id"]: c["image"].astype(np.float32) for c in cases}
    used, fixed = pooled_diameters([ann[c["id"]].astype(np.int32) for c in split["train"]])
    seeds = {}
    for seed in range(10):
        run = ROOT / f"results/cellpose_bbbc_scratch/cellpose3_scratch_s{seed}"
        model = models.CellposeModel(gpu=True, pretrained_model=str(run / f"models/cellpose3_scratch_s{seed}"))

        def ev(cs, f, p):
            masks, _, _ = model.eval([img[c["id"]] for c in cs], channels=[0, 0], diameter=fixed,
                                     flow_threshold=f, cellprob_threshold=p)
            return [published_metrics(ann[c["id"]], m) for c, m in zip(cs, masks)]
        grid = [(float(np.mean([r["caicedo_af1"] for r in ev(split["val"], f, p)])), f, p)
                for f in FLOW for p in CELLPROB]
        _, f, p = max(grid)
        rows = ev(split["test"], f, p)
        old = json.loads((run / "summary.json").read_text())["mean"]
        seeds[seed] = {"flow": f, "cellprob": p,
                       "per_image": {c["id"]: r for c, r in zip(split["test"], rows)},
                       "mean": {k: float(np.mean([r[k] for r in rows])) for k in ("dice1", "aji", "pq", "caicedo_af1")},
                       "original_mean": {k: old[k] for k in ("dice1", "aji", "pq", "caicedo_af1")}}
        print("bbbc", seed, f, p, {k: round(v, 4) for k, v in seeds[seed]["mean"].items()},
              "orig aji", round(old["aji"], 4), flush=True)
    return {"diameter_used": used, "diameter_convention": fixed, "seeds": seeds}


def nucleusseg():
    cases = load_nucleusseg_extended(ROOT / "data/nucleusseg_extended")
    train = [c for c in cases if c["split"] == "train"]
    test = [c for c in cases if c["split"] == "test"]
    used, fixed = pooled_diameters([c["instances"].astype(np.int32) for c in train])
    seeds = {}
    for seed in range(5):
        run = ROOT / f"results/cellpose_nucleusseg_scratch/cellpose3_scratch_s{seed}"
        model = models.CellposeModel(gpu=True, pretrained_model=str(run / f"models/cellpose3_scratch_s{seed}"))

        def ev(cs, f, p):
            masks, _, _ = model.eval([gray(c["image"]) for c in cs], channels=[0, 0], diameter=fixed,
                                     flow_threshold=f, cellprob_threshold=p)
            return masks
        grid = []
        for f in FLOW:
            for p in CELLPROB:
                rows = [bilkent_metrics(c["instances"], m) for c, m in zip(train, ev(train, f, p))]
                grid.append((aggregate_bilkent(rows)["nucleus_f1"], f, p))
        _, f, p = max(grid)
        sub = {}
        masks = ev(test, f, p)
        rows = [{"subset": c["subset"], **published_metrics(c["instances"], m),
                 **{f"bilkent_{k}": v for k, v in bilkent_metrics(c["instances"], m).items()}}
                for c, m in zip(test, masks)]
        for s in ("Huh7", "HepG2"):
            rs = [r for r in rows if r["subset"] == s]
            sub[s] = {"nucleus_f1": aggregate_bilkent([{k: r[f"bilkent_{k}"] for k in ADDITIVE} for r in rs])["nucleus_f1"],
                      "aji": float(np.mean([r["aji"] for r in rs])), "pq": float(np.mean([r["pq"] for r in rs]))}
        old = json.loads((run / "summary.json").read_text())["test"]
        seeds[seed] = {"flow": f, "cellprob": p, "test": sub, "original_test": old}
        print("nsd", seed, f, p, json.dumps(sub), flush=True)
    return {"diameter_used": used, "diameter_convention": fixed, "seeds": seeds}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"bbbc039": bbbc(), "nucleusseg": nucleusseg()}
    b = result["bbbc039"]["seeds"]
    result["bbbc039"]["summary"] = {
        k: {"fixed": [float(np.mean([b[s]["mean"][k] for s in b])), float(np.std([b[s]["mean"][k] for s in b], ddof=1))],
            "original": float(np.mean([b[s]["original_mean"][k] for s in b]))}
        for k in ("dice1", "aji", "pq", "caicedo_af1")}
    n = result["nucleusseg"]["seeds"]
    result["nucleusseg"]["summary"] = {
        sub: {k: {"fixed": float(np.mean([n[s]["test"][sub][k] for s in n])),
                  "original": float(np.mean([n[s]["original_test"][sub][k] for s in n]))}
              for k in ("nucleus_f1", "aji", "pq")} for sub in ("Huh7", "HepG2")}
    (OUT / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"bbbc": result["bbbc039"]["summary"], "nsd": result["nucleusseg"]["summary"]}, indent=1))


if __name__ == "__main__":
    main()
