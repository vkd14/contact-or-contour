#!/usr/bin/env python3
"""Component-preserving re-decoding of the contact and full-contour models (BBBC039 test).

The base nearest-marker decoder assigns every foreground pixel to the nearest
marker in the whole image, so a foreground component whose marker vanished can
inherit a label from another component.  Here each foreground connected
component is decoded separately: pixels go to the nearest marker inside the
same component, and a component without markers is kept as its own object.
The saved per-image foreground and thresholded boundary maps (validation-
selected threshold of the original run) are reused unchanged, so only the
decoder differs.  Reports instance metrics and merge/split/count statistics for
both targets, plus image-paired contact-minus-contour differences.
"""
import csv
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import bbbc039_reference_instances, published_metrics, remap_labels  # noqa: E402
from analyze_bbbc_merge_split import majority_owner  # noqa: E402
from train_bbbc_contact import touching_contact_target  # noqa: E402

RES = ROOT / "results"
STUDIES = {"contact": "bbbc_contact_10seed/vanilla_contact_b7", "contour": "bbbc_contour_10seed/vanilla_contour_b7"}
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")


def component_preserving(foreground, boundary, min_size=16):
    foreground = morphology.remove_small_holes(np.asarray(foreground, bool), max_size=min_size - 1)
    foreground = morphology.remove_small_objects(foreground, max_size=min_size - 1)
    markers_all = morphology.remove_small_objects(foreground & ~np.asarray(boundary, bool), max_size=min_size - 1)
    components = measure.label(foreground, connectivity=1)
    out = np.zeros(foreground.shape, np.int64)
    next_label = 1
    for region in measure.regionprops(components):
        sl = region.slice
        inside = components[sl] == region.label
        markers = measure.label(markers_all[sl] & inside, connectivity=1)
        if markers.max() == 0:
            out[sl][inside] = next_label
            next_label += 1
            continue
        _, idx = ndi.distance_transform_edt(markers == 0, return_indices=True)
        assigned = markers[tuple(idx)]
        block = out[sl]
        block[inside] = assigned[inside] + next_label - 1
        out[sl] = block
        next_label += int(markers.max())
    return remap_labels(out)


def merge_stats(truth, pred, touching):
    ids = np.arange(1, int(truth.max()) + 1)
    owners = majority_owner(truth, pred)[ids]
    shared = np.bincount(owners[owners > 0], minlength=int(pred.max()) + 1)
    merged = (owners > 0) & (shared[owners] >= 2)
    po = majority_owner(pred, truth)
    split = np.bincount(po[po > 0], minlength=int(truth.max()) + 1)[ids] >= 2
    t = touching[ids]
    return dict(n_ref=len(ids), n_pred=int(pred.max()), merged=int(merged.sum()), split=int(split.sum()),
                touching=int(t.sum()), merged_isolated=int((merged & ~t).sum()))


def job(args):
    method, seed, case_id, truth, touching = args
    pred_dir = RES / f"{STUDIES[method]}_s{seed}" / "predictions"
    fg = cv2.imread(str(pred_dir / f"{case_id}_foreground.png"), cv2.IMREAD_GRAYSCALE) > 0
    bd = cv2.imread(str(pred_dir / f"{case_id}_contact.png"), cv2.IMREAD_GRAYSCALE) > 0
    pred = np.asarray(component_preserving(fg, bd), np.int64)
    return dict(method=method, seed=seed, id=case_id, **published_metrics(truth, pred), **merge_stats(truth, pred, touching))


def main():
    cases = load_bbbc039(ROOT / "data/bbbc039", ("test",))
    truths = {}
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        for case in cases:
            raw = cv2.imdecode(np.frombuffer(archive.read(f"masks/{case['id']}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
            truth = np.asarray(bbbc039_reference_instances(raw), np.int64)
            touching = np.zeros(int(truth.max()) + 1, bool)
            touching[np.unique(truth[touching_contact_target(truth, radius=0)])] = True
            touching[0] = False
            truths[case["id"]] = (truth, touching)
    jobs = [(m, s, cid, *truths[cid]) for m in STUDIES for s in range(10) for cid in truths]
    with Pool(16) as pool:
        rows = pool.map(job, jobs, chunksize=4)
    out = RES / "bbbc_component_preserving"
    out.mkdir(exist_ok=True)
    with open(out / "per_image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"decoder": "component-preserving nearest marker; thresholds from original validation selection"}
    cube = {}
    for method in STUDIES:
        per_seed = []
        mat = []
        for seed in range(10):
            r = sorted([x for x in rows if x["method"] == method and x["seed"] == seed], key=lambda x: x["id"])
            mat.append([[x[m] for m in METRICS] for x in r])
            n = sum(x["n_ref"] for x in r); nt = sum(x["touching"] for x in r)
            per_seed.append({**{m: float(np.mean([x[m] for x in r])) for m in METRICS},
                             "merged_pct": 100 * sum(x["merged"] for x in r) / n,
                             "split_pct": 100 * sum(x["split"] for x in r) / n,
                             "merged_isolated_pct": 100 * sum(x["merged_isolated"] for x in r) / (n - nt),
                             "count_mae": float(np.mean([abs(x["n_pred"] - x["n_ref"]) for x in r]))})
        cube[method] = np.array(mat)
        summary[method] = {k: {"mean": float(np.mean([p[k] for p in per_seed])),
                               "seed_sd": float(np.std([p[k] for p in per_seed], ddof=1))} for k in per_seed[0]}
    rng = np.random.default_rng(0)
    diff = cube["contact"].mean(0) - cube["contour"].mean(0)  # image x metric
    draws = np.array([diff[rng.integers(0, 50, 50)].mean(0) for _ in range(10000)])
    summary["contact_minus_contour"] = {m: {"difference": float(diff[:, k].mean()),
                                            "ci95": [float(v) for v in np.percentile(draws[:, k], [2.5, 97.5])],
                                            "images_contact_better": int((diff[:, k] > 0).sum())}
                                        for k, m in enumerate(METRICS)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for method in STUDIES:
        print(method, {k: f"{v['mean']:.4f}({v['seed_sd']:.4f})" for k, v in summary[method].items()})
    print({m: (round(v["difference"], 4), [round(c, 4) for c in v["ci95"]], v["images_contact_better"])
           for m, v in summary["contact_minus_contour"].items()})


if __name__ == "__main__":
    main()
