#!/usr/bin/env bash
# Extra general-domain evaluation on IN22-Gen (en->ta) for every finished model. Report-only: not part of the release gate.
# Waits for scripts/run_all.sh to exit, then runs (idempotent). Usage: GPU=1 nohup bash scripts/run_in22.sh > logs/run_in22.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1} HF_HOME="$PWD/.hf_cache" TOKENIZERS_PARALLELISM=false
P=outputs/preds; F=data/processed/in22_gen_test.jsonl
say() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "scripts/run_all.sh" > /dev/null; do sleep 120; done
say "run_all finished; starting IN22-Gen evals"
ev() {  # run cfg [--adapter DIR | --model-path DIR]
  local run=$1 cfg=$2; shift 2
  [ -f $P/${run}_in22.metrics.json ] && { say "skip $run"; return; }
  say "eval $run on in22"
  python -m src.inference.translate --config "$cfg" --input $F --out $P/${run}_in22.jsonl "$@" \
    && python -m src.evaluation.metrics $P/${run}_in22.jsonl && python -m src.evaluation.error_analysis $P/${run}_in22.jsonl
}
ev baseline configs/baseline.yaml
ev baseline_600m configs/baseline_600m.yaml
for R in lora_r8 lora_r16 lora_r8_f10 lora_r8_f25 lora_r8_f50; do [ -d outputs/$R/final ] && ev $R configs/baseline.yaml --adapter outputs/$R/final; done
[ -d outputs/full_ft/final ] && ev full_ft configs/baseline_600m.yaml --model-path outputs/full_ft/final
say "IN22 DONE"
