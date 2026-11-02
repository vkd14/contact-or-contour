#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

BREASTSEG_BBBC_CONTACT_STUDY=bbbc_contact_10seed \
"$PYTHON_BIN" scripts/train_bbbc_contact.py \
  --data data/bbbc039 \
  --config configs/bbbc_contact_10seed.json
