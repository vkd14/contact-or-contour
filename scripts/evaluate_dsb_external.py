#!/usr/bin/env python3
"""Zero-shot external evaluation on the DSB2018 stage-1 test fluorescence subset.

Cohort: the 53 stage-1 test images that are grey-scale with dark background
(fluorescence), audited against all 200 BBBC039 images by normalized
cross-correlation template matching at scales 1 and 2 (max NCC 0.58; no
duplicate) - results/dsb2018_fluorescence_overlap_audit.json.

Every model is trained on BBBC039 only and applied unchanged.  Preprocessing and
each seed's decoder are frozen from BBBC039 validation selection; nothing is
tuned on external labels.  Reference instances are decoded from the official
stage-1 solution RLE (column-major, 1-indexed).
Methods (ten seeds each where available):
  semantic_ws   semantic B7 (bbbc_semantic_ws_10seed, Table 1 family), validation-selected distance watershed
  contour_cp    full-contour two-head B7, validation-selected threshold, component-preserving decoder
  contact_cp    contact two-head B7, validation-selected threshold, component-preserving decoder
  cellpose      Cellpose 3 trained from scratch on BBBC039, validation-selected thresholds and training diameter
"""
import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from breastseg.core import Net, features  # noqa: E402
from breastseg.instance_metrics import published_metrics  # noqa: E402
from analyze_bbbc_merge_split import majority_owner  # noqa: E402
from analyze_component_preserving import component_preserving  # noqa: E402
from posthoc_bbbc_decoding import semantic_instances  # noqa: E402
from train_bbbc_contact import touching_contact_target  # noqa: E402

RES = ROOT / "results"
OUT = RES / "dsb2018_external"
METRICS = ("dice1", "aji", "dq", "sq", "pq", "caicedo_af1")


def rle_decode(rle, height, width):
    flat = np.zeros(height * width, bool)
    values = list(map(int, rle.split()))
    for start, length in zip(values[0::2], values[1::2]):
        flat[start - 1:start - 1 + length] = True
    return flat.reshape(width, height).T


def load_cohort():
    audit = json.loads((RES / "dsb2018_fluorescence_overlap_audit.json").read_text())
    ids = {a["id"] for a in audit if a["max_ncc"] < 0.9}
    refs = defaultdict(list)
    with open(ROOT / "data/bbbc038/stage1_solution.csv") as handle:
        for row in csv.DictReader(handle):
            if row["ImageId"] in ids:
                refs[row["ImageId"]].append((row["EncodedPixels"], int(row["Height"]), int(row["Width"])))
    cases = []
    with zipfile.ZipFile(ROOT / "data/bbbc038/stage1_test.zip") as archive:
        for image_id in sorted(ids):
            im = cv2.imdecode(np.frombuffer(archive.read(f"{image_id}/images/{image_id}.png"), np.uint8),
                              cv2.IMREAD_UNCHANGED)
            gray = im[..., 0] if im.ndim == 3 else im
            h, w = gray.shape
            truth = np.zeros((h, w), np.int64)
            for k, (rle, hh, ww) in enumerate(refs[image_id], 1):
                truth[rle_decode(rle, hh, ww) & (truth == 0)] = k
            cases.append({"id": image_id, "image": gray, "truth": truth})
    return cases


def load_unet(checkpoint, classes):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Net(state.get("variant", "vanilla_b7"), pretrained=False, classes=classes).cuda().eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model


@torch.inference_mode()
def unet_probabilities(model, image):
    x = features(image, edge=False, size=512)[None].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model(x).float().sigmoid()[0].cpu().numpy()
    return np.stack([cv2.resize(c, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR) for c in p])


def semantic_run(seed):
    # Same checkpoint family and validation-selected watershed as Table 1 (selected by object F1 after watershed).
    return f"bbbc_semantic_ws_10seed/vanilla_semantic_ws_b7_s{seed}"


def predict_all(cases):
    preds = defaultdict(dict)  # method -> seed -> {id: labels}
    for seed in range(10):
        run = semantic_run(seed)
        cfg = json.loads((RES / "posthoc_bbbc_decoding" / f"{run.replace('/', '__')}.json").read_text())[
            "settings"]["plain"]["selected"]["config"]
        model = load_unet(ROOT / "checkpoints" / run / "best.pt", 1)
        for case in cases:
            p = unet_probabilities(model, case["image"])
            preds["semantic_ws"].setdefault(seed, {})[case["id"]] = semantic_instances(p[0] > .5, "watershed",
                                                                                      cfg["h"], cfg["sigma"])
        del model
        for method, study in (("contour_cp", "bbbc_contour_10seed/vanilla_contour_b7"),
                              ("contact_cp", "bbbc_contact_10seed/vanilla_contact_b7")):
            tau = json.loads((RES / f"{study}_s{seed}" / "summary.json").read_text())["selected_contact_threshold"]
            model = load_unet(ROOT / "checkpoints" / f"{study}_s{seed}" / "best.pt", 2)
            for case in cases:
                p = unet_probabilities(model, case["image"])
                preds[method].setdefault(seed, {})[case["id"]] = component_preserving(p[0] > .5, p[1] > tau)
            del model
        torch.cuda.empty_cache()
        print("predicted seed", seed, flush=True)
    try:
        sys.path.insert(0, str(ROOT / ".tools/cellpose_pkgs"))
        from cellpose import models
        for seed in range(10):
            run = RES / f"cellpose_bbbc_scratch/cellpose3_scratch_s{seed}"
            if not (run / "summary.json").exists():
                continue
            summ = json.loads((run / "summary.json").read_text())
            model = models.CellposeModel(gpu=True, pretrained_model=str(next((run / "models").iterdir())))
            masks, _, _ = model.eval([c["image"].astype(np.float32) for c in cases], channels=[0, 0],
                                     diameter=summ["train_diameter_px"],
                                     flow_threshold=summ["selected"]["flow_threshold"],
                                     cellprob_threshold=summ["selected"]["cellprob_threshold"])
            for case, m in zip(cases, masks):
                preds["cellpose"].setdefault(seed, {})[case["id"]] = np.asarray(m, np.int64)
    except ImportError as error:
        print("Cellpose unavailable:", error)
    return preds


def error_counts(truth, pred):
    touching = np.zeros(int(truth.max()) + 1, bool)
    touching[np.unique(truth[touching_contact_target(truth, radius=0)])] = True
    touching[0] = False
    ids = np.arange(1, int(truth.max()) + 1)
    owners = majority_owner(truth, pred)[ids]
    shared = np.bincount(owners[owners > 0], minlength=int(pred.max()) + 1)
    merged = (owners > 0) & (shared[owners] >= 2)
    po = majority_owner(pred, truth)
    split = np.bincount(po[po > 0], minlength=int(truth.max()) + 1)[ids] >= 2
    return dict(n_ref=len(ids), n_pred=int(len(np.unique(pred)) - (0 in pred)), merged=int(merged.sum()),
                split=int(split.sum()), touching=int(touching[ids].sum()),
                merged_touching=int((merged & touching[ids]).sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cohort()
    print("cohort", len(cases), "images,", sum(int(c["truth"].max()) for c in cases), "nuclei", flush=True)
    preds = predict_all(cases)
    rows, summary = [], {"cohort_images": len(cases), "methods": {}}
    for method, by_seed in preds.items():
        per_seed = []
        for seed, by_id in sorted(by_seed.items()):
            seed_rows = []
            for case in cases:
                pred = np.asarray(by_id[case["id"]], np.int64)
                row = {"method": method, "seed": seed, "id": case["id"],
                       **published_metrics(case["truth"], pred), **error_counts(case["truth"], pred)}
                seed_rows.append(row); rows.append(row)
            n = sum(r["n_ref"] for r in seed_rows); nt = sum(r["touching"] for r in seed_rows)
            per_seed.append({**{m: float(np.mean([r[m] for r in seed_rows])) for m in METRICS},
                             "merged_pct": 100 * sum(r["merged"] for r in seed_rows) / n,
                             "merged_touching_pct": 100 * sum(r["merged_touching"] for r in seed_rows) / max(nt, 1),
                             "split_pct": 100 * sum(r["split"] for r in seed_rows) / n,
                             "count_mae": float(np.mean([abs(r["n_pred"] - r["n_ref"]) for r in seed_rows])),
                             "count_bias": float(np.mean([r["n_pred"] - r["n_ref"] for r in seed_rows])),
                             "undercount_images_pct": 100 * float(np.mean([r["n_pred"] < r["n_ref"] for r in seed_rows]))})
        summary["methods"][method] = {"n_seeds": len(per_seed), **{
            k: {"mean": float(np.mean([p[k] for p in per_seed])),
                "seed_sd": float(np.std([p[k] for p in per_seed], ddof=1)) if len(per_seed) > 1 else None}
            for k in per_seed[0]}}
    with open(OUT / "per_image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for method, v in summary["methods"].items():
        print(method, v["n_seeds"], {k: round(x["mean"], 4) for k, x in v.items() if isinstance(x, dict)})


if __name__ == "__main__":
    main()
