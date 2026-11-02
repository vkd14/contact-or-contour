#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

"$PYTHON_BIN" scripts/download_nucleusseg.py --release extended
"$PYTHON_BIN" scripts/train_nucleusseg_contact.py \
  --data data/nucleusseg_extended \
  --dataset-version extended \
  --study nucleusseg_extended_contact_10seed \
  --config configs/nucleusseg_extended_contact_10seed.json
"$PYTHON_BIN" scripts/analyze_nucleusseg_extended_10seed.py
