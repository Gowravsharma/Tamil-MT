#!/usr/bin/env bash
# Full experiment queue (E0-E5). Idempotent: finished steps are skipped, so it is safe to re-run after an interruption.
# Usage: GPU=1 nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1} HF_HOME="$PWD/.hf_cache" TOKENIZERS_PARALLELISM=false
P=outputs/preds; mkdir -p $P logs reports
say() { echo "[$(date '+%F %T')] $*"; }

# translate + score + error analysis for one (run, test set); skipped if metrics already exist
ev_set() {  # run  cfg  file  name  [--adapter DIR | --model-path DIR]
  local run=$1 cfg=$2 file=$3 name=$4; shift 4
  local out=$P/${run}_${name}.jsonl
  [ -f "$P/${run}_${name}.metrics.json" ] && { say "skip eval $run/$name"; return; }
  say "eval $run on $name"
  python -m src.inference.translate --config "$cfg" --input "$file" --out "$out" "$@" \
    && python -m src.evaluation.metrics "$out" && python -m src.evaluation.error_analysis "$out"
}

ev_run() {  # run  cfg  baseline_run  [--adapter DIR | --model-path DIR]
  local run=$1 cfg=$2 base=$3; shift 3
  ev_set "$run" "$cfg" data/processed/test.jsonl domain "$@"
  ev_set "$run" "$cfg" data/processed/general_test.jsonl general "$@"
  [ -f reports/release_${run}.md ] || python -m src.evaluation.compare_releases \
    --prev domain=$P/${base}_domain.jsonl general=$P/${base}_general.jsonl \
    --cand domain=$P/${run}_domain.jsonl general=$P/${run}_general.jsonl \
    --thresholds configs/regression_thresholds.yaml --report reports/release_${run}.md \
    || say "GATE FLAGGED $run - see reports/release_${run}.md"
}

train() {  # run  config
  [ -f outputs/$1/train_stats.json ] && { say "skip train $1"; return; }
  say "train $1"; python -m src.training.train "$2"
}

# ---- baselines -----------------------------------------------------------------------------------------------
ev_set baseline configs/baseline.yaml data/processed/general_test.jsonl general
ev_set baseline_600m configs/baseline_600m.yaml data/processed/test.jsonl domain
ev_set baseline_600m configs/baseline_600m.yaml data/processed/general_test.jsonl general

# ---- LoRA runs (E1, E2, E3) ------------------------------------------------------------------------------------
for R in lora_r8 lora_r16 lora_r8_f10 lora_r8_f25 lora_r8_f50; do
  train $R configs/$R.yaml
  ev_run $R configs/baseline.yaml baseline --adapter outputs/$R/final
done

# ---- full fine-tuning reference (E4), compared against the 600M baseline ---------------------------------------
train full_ft configs/full_finetune.yaml
ev_run full_ft configs/baseline_600m.yaml baseline_600m --model-path outputs/full_ft/final

say "ALL DONE"
