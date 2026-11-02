#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
PY=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ $# -ne 1 ]]; then
    echo 'Usage: scripts/run_astra_final.sh WINNING_ARM_OR_CONFIG_NAME' >&2
    exit 2
fi
arm=${1##*/}
arm=${arm%.json}
arm=${arm#bbbc_astra_}
"$PY" scripts/prepare_astra_final.py "$arm"
BREASTSEG_BBBC_CONTACT_STUDY=bbbc_astra_final_locked_10seed "$PY" scripts/train_bbbc_contact.py \
    --data data/bbbc039 --config configs/bbbc_astra_final_locked.json
"$PY" scripts/posthoc_astra.py --arm "$arm" --final-lock configs/astra_final_lock.json \
    --workers "${ASTRA_WORKERS:-8}"
