#!/usr/bin/env python3
"""Merge/split and counting analysis on the BBBC039 test set (ten seeds).

For each test image and method:
  merged nuclei  - reference nuclei whose majority (>50% of area) lies in a
                   predicted object that also holds the majority of another
                   reference nucleus;
  split nuclei   - reference nuclei that contain the majority of >= 2
                   predicted objects;
  count error    - |#predicted - #reference|.
Reference nuclei are also stratified into touching (share an 8-neighbour
boundary with another nucleus) and isolated.  Rates are pooled over the 50
images within each seed, then summarised as mean (seed SD) over ten seeds.
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
from breastseg.instance_metrics import bbbc039_reference_instances  # noqa: E402
from posthoc_bbbc_decoding import semantic_instances  # noqa: E402
from train_bbbc_contact import touching_contact_target  # noqa: E402

RES = ROOT / "results"
POST = RES / "posthoc_bbbc_decoding"


def semantic_stem(seed):
    return (f"public_study/vanilla_b7_s{seed}" if seed < 3
            else f"bbbc_primary_10seed_additional/vanilla_b7_s{seed}")


def prediction(method, seed, case_id):
    if method == "watershed":
        # Table 1 family: bbbc_semantic_ws_10seed, validation-selected watershed (scripts/save_semantic_ws_predictions.py)
        return cv2.imread(str(RES / f"bbbc_semantic_ws_10seed_predictions/s{seed}/{case_id}_instances.tif"),
                          cv2.IMREAD_UNCHANGED)
    from analyze_component_preserving import component_preserving
    study = {"contour": "bbbc_contour_10seed/vanilla_contour_b7", "contact": "bbbc_contact_10seed/vanilla_contact_b7"}[method]
    d = RES / f"{study}_s{seed}" / "predictions"
    fg = cv2.imread(str(d / f"{case_id}_foreground.png"), cv2.IMREAD_GRAYSCALE) > 0
    bd = cv2.imread(str(d / f"{case_id}_contact.png"), cv2.IMREAD_GRAYSCALE) > 0
    return component_preserving(fg, bd)


def majority_owner(labels_a, labels_b):
    """For each object in a, the b-label holding >50% of its area (0 if none)."""
    a, b = labels_a.ravel(), labels_b.ravel()
    n_a = int(labels_a.max())
    owners = np.zeros(n_a + 1, np.int64)
    if n_a == 0:
        return owners
    pairs = np.stack([a, b], 1)[a > 0]
    keys, counts = np.unique(pairs, axis=0, return_counts=True)
    areas = np.bincount(a, minlength=n_a + 1)
    for (ia, ib), c in zip(keys, counts):
        if ib > 0 and c > areas[ia] / 2:
            owners[ia] = ib
    return owners


def score(job):
    method, seed, case_id, truth, touching = job
    pred = np.asarray(prediction(method, seed, case_id), np.int64)
    ref_owner = majority_owner(truth, pred)          # reference -> predicted object
    pred_owner = majority_owner(pred, truth)         # predicted -> reference nucleus
    ids = np.arange(1, int(truth.max()) + 1)
    owners = ref_owner[ids]
    shared = np.bincount(owners[owners > 0], minlength=int(pred.max()) + 1)
    merged = (owners > 0) & (shared[owners] >= 2)
    split_counts = np.bincount(pred_owner[pred_owner > 0], minlength=int(truth.max()) + 1)
    split = split_counts[ids] >= 2
    t = touching[ids]
    return dict(method=method, seed=seed, id=case_id, n_ref=len(ids), n_pred=int(len(np.unique(pred)) - (0 in pred)),
                merged=int(merged.sum()), split=int(split.sum()), touching=int(t.sum()),
                merged_touching=int((merged & t).sum()), split_touching=int((split & t).sum()),
                merged_isolated=int((merged & ~t).sum()), split_isolated=int((split & ~t).sum()))


def main():
    cases = load_bbbc039(ROOT / "data/bbbc039", ("test",))
    truths = {}
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        for case in cases:
            raw = cv2.imdecode(np.frombuffer(archive.read(f"masks/{case['id']}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
            truth = np.asarray(bbbc039_reference_instances(raw), np.int64)
            contact = touching_contact_target(truth, radius=0)
            touching = np.zeros(int(truth.max()) + 1, bool)
            touching[np.unique(truth[contact])] = True
            touching[0] = False
            truths[case["id"]] = (truth, touching)
    jobs = [(m, s, cid, *truths[cid]) for m in ("watershed", "contour", "contact") for s in range(10) for cid in truths]
    with Pool(16) as pool:
        rows = pool.map(score, jobs, chunksize=4)
    out = RES / "bbbc_merge_split"
    out.mkdir(exist_ok=True)
    with open(out / "per_image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in ("watershed", "contour", "contact"):
        per_seed = []
        for seed in range(10):
            r = [x for x in rows if x["method"] == method and x["seed"] == seed]
            n_ref = sum(x["n_ref"] for x in r); n_t = sum(x["touching"] for x in r)
            per_seed.append({
                "merged_pct": 100 * sum(x["merged"] for x in r) / n_ref,
                "split_pct": 100 * sum(x["split"] for x in r) / n_ref,
                "merged_touching_pct": 100 * sum(x["merged_touching"] for x in r) / n_t,
                "merged_isolated_pct": 100 * sum(x["merged_isolated"] for x in r) / (n_ref - n_t),
                "split_touching_pct": 100 * sum(x["split_touching"] for x in r) / n_t,
                "count_mae": float(np.mean([abs(x["n_pred"] - x["n_ref"]) for x in r])),
                "count_bias": float(np.mean([x["n_pred"] - x["n_ref"] for x in r])),
                "touching_pct": 100 * n_t / n_ref, "n_ref": n_ref})
        summary[method] = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
                               "seed_sd": float(np.std([p[k] for p in per_seed], ddof=1))} for k in per_seed[0]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for method, v in summary.items():
        print(method, {k: f"{x['mean']:.2f}({x['seed_sd']:.2f})" for k, x in v.items()})


if __name__ == "__main__":
    main()
