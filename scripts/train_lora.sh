#!/usr/bin/env bash
# Usage: scripts/train_lora.sh configs/lora_r8.yaml
set -euo pipefail
CFG=${1:-configs/lora_r8.yaml}
python -m src.training.train "$CFG"
