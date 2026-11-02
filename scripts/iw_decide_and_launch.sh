#!/usr/bin/env bash
# Applies configs/iw_cellpose_decision_rule.json (frozen before outcomes) and launches the final + controls.
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/.tools/cellpose_pkgs:$REPO_DIR/src:$REPO_DIR/scripts"
cd "$REPO_DIR"
until [ $(ls results/iw_cellpose_search/iw_cellpose3_W3_s*/summary.json results/iw_cellpose_search/iw_cellpose3_W10_s*/summary.json 2>/dev/null | wc -l) -ge 4 ]; do sleep 30; done
W=$($PY - <<'PY'
import json
base = {s: json.load(open(f"results/cellpose_bbbc_scratch/cellpose3_scratch_s{s}/summary.json"))["selected"]["val_caicedo_af1"] for s in (0, 1)}
S0 = sum(base.values()) / 2
scores = {}
for W in (3, 10):
    v = {s: json.load(open(f"results/iw_cellpose_search/iw_cellpose3_W{W}_s{s}/summary.json"))["selected"]["val_caicedo_af1"] for s in (0, 1)}
    scores[W] = (sum(v.values()) / 2, all(v[s] >= base[s] for s in (0, 1)), v)
best = max(scores, key=lambda w: (scores[w][0], -w))  # ties -> W=3
S, no_decline, v = scores[best]
decision = {"baseline_val_objF1": base, "S0": S0, "scores": {str(k): {"S": x[0], "no_seed_decline": x[1], "per_seed": x[2]} for k, x in scores.items()},
            "selected_W": best, "gain": S - S0, "promoted": (S - S0 >= 0.002) and no_decline}
json.dump(decision, open("results/iw_cellpose_search/decision.json", "w"), indent=2)
import sys; print(json.dumps(decision), file=sys.stderr)
print(best if decision["promoted"] else "NONE")
PY
)
echo "DECISION W=$W $(date)"
[ "$W" = "NONE" ] && { echo "IW NOT PROMOTED"; exit 0; }
$PY scripts/train_iw_cellpose_bbbc.py --mode final --weight $W --target contact --seeds 0 1 2 3 4 > logs/iw_final_a.log 2>&1 &
$PY scripts/train_iw_cellpose_bbbc.py --mode final --weight $W --target contact --seeds 5 6 7 8 9 > logs/iw_final_b.log 2>&1 &
wait
echo "IW FINAL DONE $(date)"
$PY scripts/train_iw_cellpose_bbbc.py --mode final --weight $W --target random --seeds 0 1 2 3 4 > logs/iw_ctrl_random.log 2>&1 &
$PY scripts/train_iw_cellpose_bbbc.py --mode final --weight $W --target contour --seeds 0 1 2 > logs/iw_ctrl_contour.log 2>&1 &
wait
echo "IW CONTROLS DONE $(date)"
