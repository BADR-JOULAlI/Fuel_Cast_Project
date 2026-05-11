#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
python scripts/train_model.py --model random_forest --output-dir artifacts "$@"
