#!/usr/bin/env python3
"""Stratified per-nucleus comparison on BBBC039 test (protocol frozen in
configs/stratified_comparison_protocol.json before outcomes were viewed).

Methods: semantic + validation-tuned watershed; full-contour and contact two-head
models decoded component-preserving at their validation-selected thresholds;
Cellpose 3 from scratch (validation-selected thresholds).  For every reference
nucleus: merged / split / missed (no IoU > 0.5 match).  Strata: touching vs
isolated, small (< 128 px) vs large, and image-density tertiles.  Rates are
pooled per seed; mean (seed SD) over ten seeds.  Image-paired contact-minus-
Cellpose and contact-minus-contour differences in AJI use 10,000 bootstrap draws.
"""
import csv
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import bbbc039_reference_instances, published_metrics  # noqa: E402
from analyze_bbbc_merge_split import majority_owner, prediction  # noqa: E402
from analyze_component_preserving import component_preserving  # noqa: E402
from train_bbbc_contact import touching_contact_target  # noqa: E402

RES = ROOT / "results"
METHODS = ("semantic_watershed", "full_contour", "contact", "cellpose")


def cellpose_predictions(cases):
    sys.path.insert(0, str(ROOT / ".tools/cellpose_pkgs"))
    from cellpose import models
    for seed in range(10):
        run = RES / f"cellpose_bbbc_scratch/cellpose3_scratch_s{seed}"
        out = run / "predictions"
        if out.exists() and len(list(out.glob("*_instances.tif"))) == len(cases):
            continue
        out.mkdir(exist_ok=True)
        summ = json.loads((run / "summary.json").read_text())
        model = models.CellposeModel(gpu=True, pretrained_model=str(next((run / "models").iterdir())))
        masks, _, _ = model.eval([c["image"].astype(np.float32) for c in cases], channels=[0, 0],
                                 diameter=summ["train_diameter_px"],
                                 flow_threshold=summ["selected"]["flow_threshold"],
                                 cellprob_threshold=summ["selected"]["cellprob_threshold"])
        for c, m in zip(cases, masks):
            cv2.imwrite(str(out / f"{c['id']}_instances.tif"), np.asarray(m).astype(np.uint16))


def load_pred(method, seed, case_id):
    if method == "semantic_watershed":
        return np.asarray(prediction("watershed", seed, case_id), np.int64)
    if method == "cellpose":
        return cv2.imread(str(RES / f"cellpose_bbbc_scratch/cellpose3_scratch_s{seed}/predictions/{case_id}_instances.tif"),
                          cv2.IMREAD_UNCHANGED).astype(np.int64)
    study = {"full_contour": "bbbc_contour_10seed/vanilla_contour_b7", "contact": "bbbc_contact_10seed/vanilla_contact_b7"}[method]
    d = RES / f"{study}_s{seed}" / "predictions"
    fg = cv2.imread(str(d / f"{case_id}_foreground.png"), cv2.IMREAD_GRAYSCALE) > 0
    bd = cv2.imread(str(d / f"{case_id}_contact.png"), cv2.IMREAD_GRAYSCALE) > 0
    return np.asarray(component_preserving(fg, bd), np.int64)


def per_nucleus(job):
    method, seed, case_id, truth, touching, area, density = job
    pred = load_pred(method, seed, case_id)
    ids = np.arange(1, int(truth.max()) + 1)
    owners = majority_owner(truth, pred)[ids]
    shared = np.bincount(owners[owners > 0], minlength=int(pred.max()) + 1)
    merged = (owners > 0) & (shared[owners] >= 2)
    po = majority_owner(pred, truth)
    split = np.bincount(po[po > 0], minlength=int(truth.max()) + 1)[ids] >= 2
    # missed: no predicted object with IoU > 0.5
    inter = np.zeros((len(ids) + 1, int(pred.max()) + 1))
    np.add.at(inter, (truth.ravel(), pred.ravel()), 1)
    inter = inter[1:, 1:]
    pa = np.bincount(pred.ravel(), minlength=int(pred.max()) + 1)[1:]
    iou = inter / (area[ids][:, None] + pa[None, :] - inter + 1e-9) if pa.size else np.zeros((len(ids), 1))
    missed = iou.max(1) <= 0.5 if pa.size else np.ones(len(ids), bool)
    aji = published_metrics(truth, pred)["aji"]
    return dict(method=method, seed=seed, id=case_id, aji=aji, density=density,
                touching=touching[ids].tolist(), small=(area[ids] < 128).tolist(),
                merged=merged.tolist(), split=split.tolist(), missed=missed.tolist())


def main():
    protocol = json.loads((ROOT / "configs/stratified_comparison_protocol.json").read_text())
    cases = load_bbbc039(ROOT / "data/bbbc039", ("test",))
    cellpose_predictions(cases)
    truths = {}
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        for c in cases:
            raw = cv2.imdecode(np.frombuffer(archive.read(f"masks/{c['id']}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
            t = np.asarray(bbbc039_reference_instances(raw), np.int64)
            touching = np.zeros(int(t.max()) + 1, bool)
            touching[np.unique(t[touching_contact_target(t, radius=0)])] = True
            touching[0] = False
            truths[c["id"]] = (t, touching, np.bincount(t.ravel(), minlength=int(t.max()) + 1))
    counts = {cid: int(v[0].max()) for cid, v in truths.items()}
    cuts = np.percentile(list(counts.values()), [100 / 3, 200 / 3])
    density = {cid: ("low" if n <= cuts[0] else "mid" if n <= cuts[1] else "high") for cid, n in counts.items()}
    jobs = [(m, s, cid, *truths[cid], density[cid]) for m in METHODS for s in range(10) for cid in truths]
    with Pool(16) as pool:
        rows = pool.map(per_nucleus, jobs, chunksize=4)
    strata = {"all": lambda t, sm: np.ones_like(t), "touching": lambda t, sm: t, "isolated": lambda t, sm: ~t,
              "small": lambda t, sm: sm, "large": lambda t, sm: ~sm}
    out = {"protocol": protocol["frozen"], "density_tertile_cuts": cuts.tolist(), "methods": {}}
    for method in METHODS:
        res = {}
        for stratum, fn in strata.items():
            for outcome in ("merged", "split", "missed"):
                per_seed = []
                for seed in range(10):
                    num = den = 0
                    for r in rows:
                        if r["method"] == method and r["seed"] == seed:
                            mask = fn(np.array(r["touching"]), np.array(r["small"]))
                            num += int(np.array(r[outcome])[mask].sum()); den += int(mask.sum())
                    per_seed.append(100 * num / max(den, 1))
                res[f"{stratum}_{outcome}_pct"] = {"mean": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed, ddof=1))}
        for tert in ("low", "mid", "high"):
            vals = [np.mean([r["aji"] for r in rows if r["method"] == method and r["seed"] == s and r["density"] == tert])
                    for s in range(10)]
            res[f"aji_density_{tert}"] = {"mean": float(np.mean(vals)), "seed_sd": float(np.std(vals, ddof=1))}
        out["methods"][method] = res
    rng = np.random.default_rng(0)
    ids = sorted(truths)
    aji = {m: np.array([[next(r["aji"] for r in rows if r["method"] == m and r["seed"] == s and r["id"] == i)
                         for i in ids] for s in range(10)]).mean(0) for m in METHODS}
    for other in ("cellpose", "full_contour", "semantic_watershed"):
        d = aji["contact"] - aji[other]
        draws = [d[rng.integers(0, 50, 50)].mean() for _ in range(10000)]
        out[f"aji_contact_minus_{other}"] = {"difference": float(d.mean()), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
                                             "images_contact_better": int((d > 0).sum())}
        for tert in ("low", "mid", "high"):
            sel = np.array([density[i] == tert for i in ids])
            out[f"aji_contact_minus_{other}"][f"density_{tert}"] = float(d[sel].mean())
    (RES / "bbbc_stratified_comparison.json").write_text(json.dumps(out, indent=2))
    for m, res in out["methods"].items():
        print(m, {k: round(v["mean"], 2) for k, v in res.items() if k.startswith(("touching_merged", "isolated_merged", "small_missed", "all_split", "all_missed", "aji_density"))})
    for k, v in out.items():
        if k.startswith("aji_contact_minus"):
            print(k, v)


if __name__ == "__main__":
    main()
