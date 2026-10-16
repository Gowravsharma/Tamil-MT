#!/usr/bin/env bash
# Usage: scripts/evaluate.sh <run_name> <adapter_or_model_dir> [adapter|full]
# Evaluates a trained model on domain test (+ general test if present).
set -euo pipefail
RUN=$1; DIR=$2; MODE=${3:-adapter}
FLAG="--adapter"; [ "$MODE" = "full" ] && FLAG="--model-path"
mkdir -p outputs/preds
for SET in test:domain general_test:general; do
  FILE=data/processed/${SET%%:*}.jsonl; NAME=${SET##*:}
  [ -f "$FILE" ] || continue
  python -m src.inference.translate --config configs/baseline.yaml --input "$FILE" --out outputs/preds/${RUN}_${NAME}.jsonl $FLAG "$DIR"
  python -m src.evaluation.metrics outputs/preds/${RUN}_${NAME}.jsonl
  python -m src.evaluation.error_analysis outputs/preds/${RUN}_${NAME}.jsonl
done
python -m src.evaluation.compare_releases \
  --prev domain=outputs/preds/baseline_domain.jsonl general=outputs/preds/baseline_general.jsonl \
  --cand domain=outputs/preds/${RUN}_domain.jsonl general=outputs/preds/${RUN}_general.jsonl \
  --thresholds configs/regression_thresholds.yaml --report reports/release_${RUN}.md || echo "GATE FAILED - see reports/release_${RUN}.md"
