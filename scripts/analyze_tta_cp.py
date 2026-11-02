#!/usr/bin/env python3
"""Aggregate the component-preserving TTA re-decoding (results/posthoc_tta_cp) and paired effects."""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; T = R / "posthoc_tta_cp"
M = ("dice1", "aji", "pq", "caicedo_af1")
def load(study):
    runs = [{r["id"]: r for r in csv.DictReader(open(T / f"{study}__s{s}.csv"))} for s in range(10)]
    ids = sorted(runs[0])
    return ids, {m: np.array([[float(r[i][m]) for i in ids] for r in runs]) for m in M}
def astra():
    runs = []
    for s in range(10):
        d = json.load(open(R / f"bbbc_astra_final_locked_10seed/vanilla_contact_b7_s{s}/astra_test_tta8.json"))
        runs.append({r["id"]: r for r in d["settings"]["tta8"]["diagnostic_per_image"]})
    ids = sorted(runs[0]); return ids, {m: np.array([[r[i][m] for i in ids] for r in runs]) for m in M}
def cellpose():
    runs = [{r["id"]: r for r in csv.DictReader(open(R / f"cellpose_bbbc_scratch/cellpose3_scratch_s{s}/test_per_image.csv"))} for s in range(10)]
    ids = sorted(runs[0]); return ids, {m: np.array([[float(r[i][m]) for i in ids] for r in runs]) for m in M}
def eff(a, b, rng):
    d = a - b; dr = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return {"difference": float(d.mean()), "ci95": [float(x) for x in np.percentile(dr, [2.5, 97.5])], "images_better": int((d > 0).sum())}
def main():
    data = {k: load(k) for k in ("bbbc_contour_10seed", "bbbc_contact_10seed", "bbbc_contact_native_10seed")}
    data["searched"] = astra(); data["cellpose"] = cellpose()
    ids = data["cellpose"][0]; assert all(v[0] == ids for v in data.values())
    out = {"rows": {}, "effects": {}, "selected_decoders": {}}
    for k, (_, a) in data.items():
        out["rows"][k] = {m: {"mean": float(a[m].mean()), "seed_sd": float(a[m].mean(1).std(ddof=1))} for m in M}
        print(k, {m: f"{a[m].mean():.4f}({a[m].mean(1).std(ddof=1):.4f})" for m in M})
    for k in ("bbbc_contour_10seed", "bbbc_contact_10seed", "bbbc_contact_native_10seed"):
        out["selected_decoders"][k] = Counter(json.load(open(T / f"{k}__s{s}.json"))["selected"]["decoder"] for s in range(10))
    rng = np.random.default_rng(0)
    for name, a, b in (("contact_tta_minus_contour_tta", "bbbc_contact_10seed", "bbbc_contour_10seed"),
                       ("native_minus_contact512_tta", "bbbc_contact_native_10seed", "bbbc_contact_10seed"),
                       ("searched_minus_native", "searched", "bbbc_contact_native_10seed"),
                       ("cellpose_minus_searched", "cellpose", "searched")):
        out["effects"][name] = {m: eff(data[a][1][m].mean(0), data[b][1][m].mean(0), rng) for m in M}
        print(name, {m: (round(v["difference"], 4), [round(x, 4) for x in v["ci95"]], v["images_better"]) for m, v in out["effects"][name].items()})
    print(out["selected_decoders"])
    (R / "tta_component_preserving_summary.json").write_text(json.dumps(out, indent=2))
if __name__ == "__main__":
    main()
