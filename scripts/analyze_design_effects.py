#!/usr/bin/env python3
"""Effect sizes of design choices on AJI (BBBC039 test, plus NucleusSegData target choice).

Each effect is a paired comparison of seed-averaged per-image AJI between two
otherwise matched configurations (10,000 image bootstrap draws, RNG seed 0).
Two-head models are decoded with the component-preserving decoder at their
validation-selected thresholds.  Writes results/design_effects.json.
"""
import csv
import json
import sys
import zipfile
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from breastseg.data import load_bbbc039  # noqa: E402
from breastseg.instance_metrics import bbbc039_reference_instances, published_metrics  # noqa: E402
from analyze_component_preserving import component_preserving  # noqa: E402

RES = ROOT / "results"
POST = RES / "posthoc_bbbc_decoding"


def csv_aji(paths, key="aji"):
    runs = []
    for p in paths:
        with open(p) as handle:
            runs.append({r["id"]: float(r[key]) for r in csv.DictReader(handle)})
    ids = sorted(runs[0])
    return ids, np.array([[r[i] for i in ids] for r in runs]).mean(0)


def cp_job(args):
    study, seed, case_id, truth = args
    d = RES / f"{study}_s{seed}" / "predictions"
    fg = cv2.imread(str(d / f"{case_id}_foreground.png"), cv2.IMREAD_GRAYSCALE) > 0
    bd = cv2.imread(str(d / f"{case_id}_contact.png"), cv2.IMREAD_GRAYSCALE) > 0
    return study, seed, case_id, published_metrics(truth, component_preserving(fg, bd))["aji"]


def effect(a, b, rng):
    d = a - b
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return {"difference": float(d.mean()), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            "images_better": int((d > 0).sum()), "n_images": int(len(d))}


def main():
    cases = load_bbbc039(ROOT / "data/bbbc039", ("test",))
    truths = {}
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        for c in cases:
            raw = cv2.imdecode(np.frombuffer(archive.read(f"masks/{c['id']}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
            truths[c["id"]] = np.asarray(bbbc039_reference_instances(raw), np.int64)
    ids = sorted(truths)
    studies = {"contact": "bbbc_contact_10seed/vanilla_contact_b7", "contour": "bbbc_contour_10seed/vanilla_contour_b7",
               "carved": "bbbc_carved_10seed/vanilla_carved_b7"}
    jobs = [(st, s, i, truths[i]) for st in studies.values() for s in range(10) for i in ids]
    with Pool(16) as pool:
        out = pool.map(cp_job, jobs, chunksize=4)
    cp = {}
    for name, st in studies.items():
        cp[name] = np.array([[next(a for (x, s2, i2, a) in out if x == st and s2 == s and i2 == i) for i in ids]
                             for s in range(10)]).mean(0)
    sem = sorted(glob(str(POST / "bbbc_semantic_ws_10seed__*__plain__locked_test_per_image.csv")))
    sem_ws = sorted(glob(str(POST / "bbbc_semantic_ws_10seed__*__plain__selected_test_per_image.csv")))
    # TTA rows: component-preserving / contact-watershed grid (scripts/posthoc_tta_component_preserving.py)
    con_tta = [RES / f"posthoc_tta_cp/bbbc_contact_10seed__s{s}.csv" for s in range(10)]
    nat_tta = [RES / f"posthoc_tta_cp/bbbc_contact_native_10seed__s{s}.csv" for s in range(10)]
    cont_tta = [RES / f"posthoc_tta_cp/bbbc_contour_10seed__s{s}.csv" for s in range(10)]
    cellpose = [RES / f"cellpose_bbbc_scratch/cellpose3_scratch_s{s}/test_per_image.csv" for s in range(10)]
    ccell = sorted(glob(str(RES / "contact_cellpose_bbbc_scratch/*/test_selected_per_image.csv")))
    arrays = {}
    for name, paths in (("semantic_cc", sem), ("semantic_ws", sem_ws), ("contact_tta", con_tta),
                        ("native_tta", nat_tta), ("cellpose", cellpose), ("contact_cellpose", ccell),
                        ("contour_tta", cont_tta)):
        got, arr = csv_aji(paths)
        assert got == ids, name
        arrays[name] = arr
    arrays.update(cp)
    searched = []
    for s in range(10):
        d = json.loads((RES / f"bbbc_astra_final_locked_10seed/vanilla_contact_b7_s{s}/astra_test_tta8.json").read_text())
        searched.append({r["id"]: r["aji"] for r in d["settings"]["tta8"]["diagnostic_per_image"]})
    arrays["searched"] = np.array([[r[i] for i in ids] for r in searched]).mean(0)
    rng = np.random.default_rng(0)
    effects = [
        ("Decoder: tuned watershed vs. connected components", "semantic_ws", "semantic_cc"),
        ("Boundary supervision: full contour vs. semantic + watershed", "contour", "semantic_ws"),
        ("Target: contact vs. full contour", "contact", "contour"),
        ("Foreground: intact vs. contacts carved out", "contact", "carved"),
        ("Inference: TTA + selected decoder", "contact_tta", "contact"),
        ("Resolution: native vs. 512^2 (both with TTA)", "native_tta", "contact_tta"),
        ("Representation: Cellpose vs. contact model", "cellpose", "contact"),
        ("Contact head added to Cellpose", "contact_cellpose", "cellpose"),
        ("Target: contact vs. full contour (TTA, same decoder grid)", "contact_tta", "contour_tta"),
        ("Representation: Cellpose vs. best contact model (searched recipe, native + TTA)", "cellpose", "searched"),
    ]
    result = {"dataset": "BBBC039 test (50 images); per-image AJI averaged over seeds", "effects": []}
    for label, a, b in effects:
        e = effect(arrays[a], arrays[b], rng)
        result["effects"].append({"label": label, "a": a, "b": b, **e})
        print(f"{label:62s} {e['difference']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] {e['images_better']}/50")
    nsd = json.loads((RES / "nucleusseg_cp/summary.json").read_text())["subsets"]  # component-preserving re-decode
    for sub in ("Huh7", "HepG2"):
        d, ci, better, n = nsd[sub]["aji_diff"]
        result["effects"].append({"label": f"Target: contact vs. full contour (NucleusSegData {sub})",
                                  "difference": d, "ci95": ci, "images_better": better, "n_images": n})
        print(f"NucleusSegData {sub} contact vs contour {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] {better}/{n}")
    star = json.loads((RES / "nucleusseg_star_pair.json").read_text())  # both targets, widened grid + TTA
    for sub in ("Huh7", "HepG2"):
        d, ci, better = star[sub]["aji_diff"]
        result["effects"].append({"label": f"Target: contact vs. full contour (NucleusSegData {sub}, widened grid + TTA)",
                                  "difference": d, "ci95": ci, "images_better": better, "n_images": star[sub]["n_images"]})
        print(f"NucleusSegData {sub} widened contact vs contour {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] {better}")
    dsb = json.loads((RES / "dsb2018_external/paired.json").read_text())
    for label, key in (("Boundary supervision: full contour vs. semantic + watershed (DSB external)", "contour_cp_minus_semantic_ws"),
                       ("Target: contact vs. full contour (DSB external)", "contact_cp_minus_contour_cp")):
        d, ci, better = dsb[key]["aji"]
        result["effects"].append({"label": label, "difference": d, "ci95": ci, "images_better": better, "n_images": 53})
        print(label, d, ci, better)
    dec = json.loads((RES / "dsb2018_external/decoder.json").read_text())["aji"]["ws_minus_cc"]
    result["effects"].append({"label": "Decoder: tuned watershed vs. connected components (DSB external)",
                              "difference": dec[0], "ci95": dec[1], "images_better": dec[2], "n_images": dec[3]})
    (RES / "design_effects.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
