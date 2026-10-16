#!/usr/bin/env bash
# E0: pretrained model, no adaptation.  Run from repo root.
set -euo pipefail
mkdir -p outputs/preds
python -m src.inference.translate --config configs/baseline.yaml --input data/processed/test.jsonl --out outputs/preds/baseline_domain.jsonl
[ -f data/processed/general_test.jsonl ] && python -m src.inference.translate --config configs/baseline.yaml --input data/processed/general_test.jsonl --out outputs/preds/baseline_general.jsonl
python -m src.evaluation.metrics outputs/preds/baseline_domain.jsonl
python -m src.evaluation.error_analysis outputs/preds/baseline_domain.jsonl
