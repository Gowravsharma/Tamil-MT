# Model card: indic-mt-peft (English → Tamil, legal-domain adaptation of NLLB-200)

All numbers below are copied from `reports/experiment_results.md` (generated from `outputs/`). Single seed per setting.

## What this is
- Base models: `facebook/nllb-200-distilled-1.3B` (LoRA runs) and `facebook/nllb-200-distilled-600M` (full fine-tuning reference). Check the NLLB licence (CC-BY-NC 4.0) before any redistribution or commercial use.
- Adaptation: LoRA on `q_proj,k_proj,v_proj,out_proj` (incl. decoder cross-attention), r=8 (4,718,592 trainable params, 0.343%) and r=16 (9,437,184, 0.684%); full fine-tuning of the 600M model (615,073,792 params).
- Training data: Anuvaad **Legal** en–ta (634,920 train pairs after cleaning; see `reports/data_quality.md`). 1 epoch, lr 1e-4 (LoRA) / 2e-5 (full), effective batch 32, bf16, best checkpoint by validation loss.
- Adapters are in `outputs/<run>/final` (not committed). Anuvaad data licence and NLLB licence must be checked before redistributing weights.

## Results (chrF++; Δ = paired-bootstrap 95% interval vs the matching pretrained baseline)

| Model | In-domain (Anuvaad legal test) | WMT24++ general | IN22-Gen general | IN22-Gen legal subset (80 seg.) |
|---|---|---|---|---|
| NLLB-1.3B pretrained | 61.35 | 44.59 | 46.71 | 48.82 |
| + LoRA r=8 | 75.63 (+14.28 [+14.17, +14.38]) | 44.94 (+0.35 [−0.06, +0.76]) | 46.64 (−0.08 [−0.46, +0.32]) | −0.79 [−2.01, +0.54] |
| + LoRA r=16 | 76.20 (+14.85 [+14.75, +14.95]) | 45.00 (+0.40 [0.00, +0.79]) | 46.65 (−0.07 [−0.46, +0.34]) | −0.99 [−2.27, +0.35] |
| NLLB-600M pretrained | 58.35 | 43.36 | 44.94 | 45.93 |
| + full fine-tune | 77.44 (+19.08 [+18.98, +19.19]) | 44.24 (+0.88 [+0.44, +1.32]) | 45.84 (+0.91 [+0.50, +1.33]) | +1.26 [−0.22, +2.80] |

Data-fraction curve (LoRA r=8, in-domain chrF++): 10% 72.43, 25% 73.66, 50% 74.64, 100% 75.63. Release gate: PASS for all 6 adapted models.

## How to read this
- **In-domain gains are large but probably inflated.** The test set is a random sentence-level split of one corpus with no document IDs, legal text is repetitive, and the references look machine-translated. Part of the gain is learning that reference system's style and boilerplate.
- **The independent legal check does not show the same gain.** On IN22-Gen's 80 legal sentences (a different source) LoRA r=8/r=16 are slightly below the baseline (intervals include 0) and full fine-tuning is slightly above (interval includes 0). With n=80 this is inconclusive, but it means the +14 on the Anuvaad test should not be assumed to carry over to other legal text.
- **No forgetting detected.** General-domain chrF++ is flat to slightly up on WMT24++ and IN22-Gen. Intervals are wide (960 and 1,024 segments); the supported claim is "no measurable degradation".
- **Heuristic error rates fall sharply in-domain** (number mismatch 3.87% → ~0.5%, untranslated/Latin 1.24% → ~0.05% for 1.3B). These are heuristics, not correctness checks.
- LoRA r=16 vs r=8: +0.57 chrF++ in-domain (outside the CI), single seed.
- Full fine-tuning (600M) beats LoRA (1.3B) in-domain (77.44 vs 75.63) but it is a different base model; this is not a like-for-like LoRA vs full comparison.
- Inference timings come from a shared server and are not controlled benchmarks (e.g. the fine-tuned 600M model measured 0.026 s/sentence vs 0.048 for its baseline; the cause was not investigated).

## Not done / limitations
- No human evaluation. A Tamil speaker should review a stratified sample of outputs before any use (the evaluation scripts write `*.review.jsonl`; these contain dataset text, so they are not included in this repository). **Do not use for legal, medical or government translation on the strength of these scores.**
- No COMET (dependency conflict), no multiple seeds, 1 epoch only (validation loss was still decreasing slowly), no document-level split, IN22-Conv and FLORES not used (gated, access not granted).
- Known behaviours seen in outputs: NLLB emits `ஃ` where references use `:`; list markers like "(b)" stay Latin; acronym expansion errors occur (one wrong expansion of "NHM" seen in the baseline).
