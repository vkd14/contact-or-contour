#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
PY=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Run only when the GPU is free; all training/inference jobs here are sequential.
# No test image, annotation, prediction, or metric is loaded by this search.
for arm in baseline40 long200 lowlr200 focal200 radius0_200 intensity200; do
    BREASTSEG_BBBC_CONTACT_STUDY="bbbc_astra_search_$arm" "$PY" scripts/train_bbbc_contact.py \
        --data data/bbbc039 --config "configs/bbbc_astra_$arm.json"
    "$PY" scripts/posthoc_astra.py --arm "$arm" --workers "${ASTRA_WORKERS:-8}"
done
"$PY" scripts/summarize_astra_search.py --require-complete
