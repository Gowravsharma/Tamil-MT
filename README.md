# Tamil-MT

**English → Tamil legal-domain machine translation: LoRA adaptation of NLLB-200, with a leakage-checked, regression-gated evaluation.**

I adapted a pretrained multilingual translation model (NLLB-200) to English→Tamil legal text by training small LoRA adapters
(0.34% of the parameters) instead of the whole network, then measured three things carefully: how much it helps in-domain, whether it
hurts general translation, and whether the gain holds up on an independent test. The last question is where the interesting result is.

## Results at a glance

chrF++ (higher is better). Δ is the paired-bootstrap 95% interval of the difference to the matching pretrained baseline.
Full tables, BLEU, error rates and per-domain breakdowns: [`reports/experiment_results.md`](reports/experiment_results.md).

| Model | In-domain test (Anuvaad legal, 78,751 sent.) | General: WMT24++ (960 seg.) | General: IN22-Gen (1,024 seg.) | IN22-Gen *legal* subset (80 seg., independent source) |
|---|---|---|---|---|
| NLLB-1.3B pretrained | 61.35 | 44.59 | 46.71 | 48.82 |
| + LoRA r=8 (4.7M trainable params) | **75.63** (+14.28 [+14.17, +14.38]) | 44.94 (+0.35 [−0.06, +0.76]) | 46.64 (−0.08 [−0.46, +0.32]) | −0.79 [−2.01, +0.54] |
| + LoRA r=16 (9.4M trainable params) | **76.20** (+14.85 [+14.75, +14.95]) | 45.00 (+0.40 [0.00, +0.79]) | 46.65 (−0.07 [−0.46, +0.34]) | −0.99 [−2.27, +0.35] |
| NLLB-600M pretrained | 58.35 | 43.36 | 44.94 | 45.93 |
| + full fine-tune (615M params) | **77.44** (+19.08 [+18.98, +19.19]) | 44.24 (+0.88 [+0.44, +1.32]) | 45.84 (+0.91 [+0.50, +1.33]) | +1.26 [−0.22, +2.80] |

Data efficiency (LoRA r=8, in-domain chrF++): 10% of the data → 72.43, 25% → 73.66, 50% → 74.64, 100% → 75.63.
10% of the data (63k pairs, 14 minutes of training) already gives about three quarters of the gain.

### The honest reading
- **The large in-domain gain is probably inflated.** The in-domain test set is a random *sentence-level* split of one corpus with no
  document IDs, legal text is highly repetitive, and the reference translations appear to be machine-generated. Part of the +14 is the model
  learning that corpus's style and boilerplate.
- **It did not carry over to an independent legal test.** On the 80 legal sentences of IN22-Gen (a different source), LoRA shows no gain
  (−0.8 and −1.0 chrF++, intervals include zero). Eighty sentences is too few to be conclusive, but it means the in-domain number should
  be treated as an upper bound.
- **No forgetting detected.** General-domain scores are flat to slightly up. With about 1,000 segments per set the intervals are wide, so
  the supported claim is "no measurable degradation", not "improved".
- **Not state of the art.** My baseline reproduces the published figure for its model family (46.71 vs 47.0 on IN22-Gen), and the adapted
  models stay at that level on general text, below IndicTrans2 (49.5, as published). See [`docs/evaluation-context.md`](docs/evaluation-context.md).
- **No human evaluation.** Reference-based scores do not prove the translations are correct. Do not use these models for legal, medical or
  government translation.

## What is in this repository

| Part | What it does |
|---|---|
| **Data pipeline** (`src/data/`, `scripts/prepare_data.sh`) | Cleans 831,177 Anuvaad legal sentence pairs → 792,401 (removes empty, duplicate, badly-sized and non-Tamil-target pairs; flags conflicting sources), makes a leakage-aware 80/10/10 split with zero exact-source overlap, builds the WMT24++ and IN22-Gen test files. Regenerates every file used here byte-for-byte. Report: [`reports/data_quality.md`](reports/data_quality.md). |
| **Training** (`src/training/train.py`, `configs/`) | LoRA (rank 8 / 16) and full fine-tuning with Hugging Face `Trainer`; best checkpoint chosen by validation loss only. |
| **Inference** (`src/inference/translate.py`) | Beam-search translation for the pretrained model, an adapter, or a fully fine-tuned model. |
| **Evaluation** (`src/evaluation/`) | chrF++ and BLEU (sacreBLEU), paired-bootstrap confidence intervals, heuristic error checks (number mismatch, untranslated/Latin output, copied source, omissions, repetition), report generator. |
| **Release gate** (`src/evaluation/compare_releases.py`, `configs/regression_thresholds.yaml`) | Compares a candidate model with the previous release on fixed test sets and fails on an in-domain gain below the minimum, a general-domain drop beyond the allowed amount, or excess untranslated/number-mismatch output. Thresholds were fixed before any adapted-model result (hash in [`results/thresholds.sha256`](results/thresholds.sha256)). All six adapted models pass. |
| **Tests** (`tests/`) | 8 tests: data cleaning/splitting, the gate, and a tiny-model LoRA training-and-translation smoke test. |

## Method

- **Base models:** `facebook/nllb-200-distilled-1.3B` (LoRA) and `facebook/nllb-200-distilled-600M` (full fine-tuning reference, a smaller model chosen for memory reasons).
- **Data:** Anuvaad Legal en–ta (Indian Kanoon judgments, HC-SUVAS judicial, legal terms): 634,920 train / 78,730 validation / 78,751 test pairs.
- **LoRA:** rank 8 (alpha 16) or 16 (alpha 32), dropout 0.05, on the q/k/v/output projections of every attention block, including decoder cross-attention (288 modules; 4,718,592 trainable parameters = 0.34% at rank 8).
- **Training:** 1 epoch, lr 1e-4 (LoRA) / 2e-5 (full), effective batch 32, bf16, length-grouped batches. LoRA r=8 trains in about 2 h on one L40S (peak 32.8 GB).
- **Selection:** checkpoints and settings were chosen on a fixed 4,000-pair validation subset. The test sets were never used to choose anything.
- **Decoding:** beam 5. Inference cost is unchanged by LoRA (about 0.07 s/sentence on an L40S; adapters are evaluated unmerged).
- **Why one epoch:** chosen from measured throughput on a shared GPU; validation loss was still decreasing slowly at the end, so a second epoch might help (not tested).

## Engineering notes (problems found and fixed)

1. **Loss and gradients inflated by the gradient-accumulation factor.** With Transformers 5 the M2M100-style model ignores `num_items_in_batch`, so the
   loss was summed instead of averaged across accumulation steps. Setting `trainer.model_accepts_loss_kwargs = False` brought the logged loss from about 2.2 to about 1.0 and halved the gradient norm.
2. **Padding wasted 61% of computed tokens.** Grouping batches by length cut it to about 15% and gave about 1.9x higher training throughput.
3. **Paired bootstrap was impractically slow** (estimated about 23 h for 78k sentences). Rewriting it on per-sentence sufficient statistics gives identical scores in about 83 s.
4. **A flaky test**: LayerDrop in the tiny test model made the translation test fail about 1 run in 20; fixed by disabling it in the test (0 failures in 40 runs afterwards).
5. **Data details:** the HC-SUVAS Tamil file uses Windows line endings; the data script normalizes them.

## Reproduce

Data and model weights are **not** in this repository (licences and size). See [`docs/running-on-a-gpu-server.md`](docs/running-on-a-gpu-server.md) for details.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126 && pip install -r requirements.txt
python -m pytest -q

# put the three Anuvaad Legal en-ta zips in data/raw/, then:
bash scripts/prepare_data.sh
GPU=0 nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &     # baselines, all trainings, evals, gates (about 17 hours in total on one L40S)
python -m src.evaluation.make_report                               # -> reports/experiment_results.md
```
The per-run summary numbers behind the tables (metrics, error rates, timing, parameter counts, training statistics) are in [`results/`](results/).
Per-sentence outputs are not published because they contain dataset text.

## Limitations

- In-domain test is a sentence-level split of one corpus (see "The honest reading"). The next experiment should hold out a different *source* (for example train on Indian Kanoon, test on HC-SUVAS).
- One seed per setting: differences such as rank 16 vs rank 8 (+0.57 chrF++) have not been checked for stability across seeds.
- General-domain test sets are small (960 and 1,024 segments); IN22-Conv and FLORES-200 were not used (gated on Hugging Face).
- No COMET/xCOMET and no human evaluation.
- The full fine-tune uses a different (smaller) base model than the LoRA runs, so the two are not a like-for-like comparison.
- The adapters were trained and tested only for English→Tamil.

## Future work

The aim is to move from one language pair to **improving the full multilingual model**:

1. **Other Indian languages first.** Repeat the adaptation for the other languages NLLB supports in the IN22 benchmark (Hindi, Bengali, Telugu, Kannada, Malayalam, Marathi, Gujarati and others).
   IN22-Gen is parallel across 22 Indian languages, so every language can be scored on the same sentences.
2. **Then the wider model.** Extend towards all languages the base model covers (NLLB-200 has about 200), comparing one adapter per language with a single shared multilingual adapter, and measuring cross-language interference and forgetting.
3. **Stronger evaluation first.** A source-level held-out test, several seeds, COMET/xCOMET, and review of sampled outputs by native speakers.
4. **Stronger baselines.** Compare against IndicTrans2 and larger multilingual models, with trainable and total parameters reported next to quality and speed.
5. **Adapter design.** Compare LoRA variants (DoRA, rsLoRA, adapters on feed-forward layers) at matched trainable-parameter budgets.

None of this is implemented here yet.

## Data and model notices

- **Code:** MIT (see `LICENSE`).
- **Models:** NLLB-200 weights are CC-BY-NC-4.0 (non-commercial); trained adapters inherit that restriction and are not distributed here.
- **Training data:** [Anuvaad parallel corpus](https://github.com/project-anuvaad/anuvaad-parallel-corpus). Check its licence before redistributing anything derived from it.
- **Test data:** [WMT24++](https://huggingface.co/datasets/google/wmt24pp) (open) and [IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen) (gated; accept the terms on Hugging Face and do not redistribute).

## Model card
[`MODEL_CARD.md`](MODEL_CARD.md) has the same results with the intended use, limitations and cautions.
