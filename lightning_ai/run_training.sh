#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

pip install -r requirements.txt
python scripts/train_model.py --model random_forest --output-dir artifacts "$@"
