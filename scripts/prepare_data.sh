#!/usr/bin/env bash
# Rebuild all datasets from the three Anuvaad Legal en-ta zips (download them from the Anuvaad parallel-corpus README into data/raw/ first).
# Usage: bash scripts/prepare_data.sh [OUT_ROOT]   (OUT_ROOT defaults to ".", producing data/ and reports/ under it)
# IN22-Gen is gated: run `huggingface-cli login` (token with accepted terms) before the last step, or skip it.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:-.}; D=$ROOT/data; mkdir -p "$D/interim" "$D/processed" "$ROOT/reports"

# 1. extract (zips in data/raw/ are never modified)
unzip -oq data/raw/ik-2021-v1-en-ta.zip -x '__MACOSX/*' -d data/raw/ik
unzip -oq data/raw/internal-judicial-2021-v1-en-ta.zip -d data/raw/hcsuvas
unzip -oq data/raw/legal-terms-2021-v1-en-ta.zip -x '__MACOSX/*' -d data/raw/legalterms

# 2. combine in a fixed order (ik, hcsuvas, legal-terms); the HC-SUVAS Tamil file has CRLF line endings, so strip them
cat data/raw/ik/ij_2021_v1.en data/raw/hcsuvas/en-ta/ij-train.en data/raw/legalterms/legal-terms.en > "$D/interim/legal_all.en"
for f in data/raw/ik/ij_2021_v1.ta data/raw/hcsuvas/en-ta/ij-train.ta data/raw/legalterms/legal-terms.ta; do sed -e 's/\r$//' "$f"; done > "$D/interim/legal_all.ta"
[ "$(wc -l < "$D/interim/legal_all.en")" = "$(wc -l < "$D/interim/legal_all.ta")" ] || { echo "line counts differ" >&2; exit 1; }

# 3. clean + leakage-aware split
python -m src.data.clean --src-file "$D/interim/legal_all.en" --tgt-file "$D/interim/legal_all.ta" --domain legal \
  --corpus-name anuvaad-legal --out "$D/processed/clean.jsonl" --report "$ROOT/reports/data_quality.json"
python -m src.data.split --input "$D/processed/clean.jsonl" --outdir "$D/processed"

# 4. general-domain test sets
python -m src.data.prepare_wmt24pp --out "$D/processed/general_test.jsonl"
python -m src.data.prepare_in22 --out "$D/processed/in22_gen_test.jsonl" --raw-out "$D/interim/in22_gen_en-ta.jsonl" || echo "IN22-Gen skipped (gated; needs HF token)"
