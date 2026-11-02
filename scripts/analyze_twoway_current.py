#!/usr/bin/env python3
"""Two-way (training seed x test image) bootstrap for the paper's main AJI contrasts.

Each draw resamples seeds of each arm independently and test images jointly (paired),
then averages per-image AJI over the drawn seeds; 10,000 draws, RNG seed 0.
Also the equal-seed (seeds 0-4) Cellpose contact-head comparison.  Writes
results/twoway_bootstrap_current.json.
"""
import csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
def from_long(method):
    rows = [r for r in csv.DictReader(open(R / "bbbc_component_preserving/per_image.csv")) if r["method"] == method]
    seeds = sorted({int(r["seed"]) for r in rows}); ids = sorted({r["id"] for r in rows})
    d = {(int(r["seed"]), r["id"]): float(r["aji"]) for r in rows}
    return ids, np.array([[d[(s, i)] for i in ids] for s in seeds])
def from_csvs(paths, key="aji"):
    runs = [{r["id"]: float(r[key]) for r in csv.DictReader(open(p))} for p in paths]
    ids = sorted(runs[0]); return ids, np.array([[r[i] for i in ids] for r in runs])
def searched():
    runs = []
    for s in range(10):
        d = json.load(open(R / f"bbbc_astra_final_locked_10seed/vanilla_contact_b7_s{s}/astra_test_tta8.json"))
        runs.append({r["id"]: r["aji"] for r in d["settings"]["tta8"]["diagnostic_per_image"]})
    ids = sorted(runs[0]); return ids, np.array([[r[i] for i in ids] for r in runs])
def twoway(a, b, rng, n=10000, paired_seeds=False):
    """paired_seeds: both arms come from the same trained models (e.g. two decoders), so seeds are drawn jointly."""
    ns_a, ns_b, ni = a.shape[0], b.shape[0], a.shape[1]
    point = float((a.mean(0) - b.mean(0)).mean()); draws = np.empty(n)
    for k in range(n):
        im = rng.integers(0, ni, ni)
        sa = rng.integers(0, ns_a, ns_a)
        sb = sa if paired_seeds else rng.integers(0, ns_b, ns_b)
        draws[k] = a[sa][:, im].mean() - b[sb][:, im].mean()
    return {"difference": point, "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            "seeds": [int(ns_a), int(ns_b)]}
def main():
    arms = {
        "contact_cp": from_long("contact"), "contour_cp": from_long("contour"),
        "contact_tta": from_csvs([R / f"posthoc_tta_cp/bbbc_contact_10seed__s{s}.csv" for s in range(10)]),
        "contour_tta": from_csvs([R / f"posthoc_tta_cp/bbbc_contour_10seed__s{s}.csv" for s in range(10)]),
        "semantic_ws": from_csvs(sorted(R.glob("posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*__plain__selected_test_per_image.csv"))),
        "semantic_cc": from_csvs(sorted(R.glob("posthoc_bbbc_decoding/bbbc_semantic_ws_10seed__*__plain__locked_test_per_image.csv"))),
        "cellpose": from_csvs([R / f"cellpose_bbbc_scratch/cellpose3_scratch_s{s}/test_per_image.csv" for s in range(10)]),
        "contact_cellpose": from_csvs(sorted(R.glob("contact_cellpose_bbbc_scratch/*/test_selected_per_image.csv"))),
        "searched": searched(),
    }
    ids = arms["cellpose"][0]; assert all(v[0] == ids for v in arms.values()), {k: len(v[0]) for k, v in arms.items()}
    A = {k: v[1] for k, v in arms.items()}
    rng = np.random.default_rng(0); out = {}
    for name, a, b in (("decoder_ws_minus_cc", "semantic_ws", "semantic_cc"),
                       ("contact_minus_contour_base", "contact_cp", "contour_cp"),
                       ("contact_minus_contour_tta", "contact_tta", "contour_tta"),
                       ("cellpose_minus_searched", "cellpose", "searched"),
                       ("contact_head_cellpose_minus_cellpose", "contact_cellpose", "cellpose")):
        out[name] = twoway(A[a], A[b], rng, paired_seeds=(name == "decoder_ws_minus_cc")); print(name, out[name])
    out["contact_head_equal_seeds_0_4"] = twoway(A["contact_cellpose"], A["cellpose"][:5], rng)
    print("equal seeds", out["contact_head_equal_seeds_0_4"])
    (R / "twoway_bootstrap_current.json").write_text(json.dumps(out, indent=2))
if __name__ == "__main__":
    main()
