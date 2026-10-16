# Running the experiments on a GPU server

Everything reported here was run on one NVIDIA L40S (46 GB) of a shared 2-GPU server.

## Environment
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126      # match your CUDA
pip install -r requirements.txt                                            # exact versions used: requirements-tested.txt
python -m pytest -q                                                        # 8 tests, CPU only
```
Optional COMET scoring (`requirements-optional.txt`) was not used; it breaks pip dependency resolution together with the other pins.

## Data
Download the three Anuvaad **Legal** en–ta zips (ik-2021-v1, internal-judicial-2021-v1, legal-terms-2021-v1) from the Anuvaad
parallel-corpus README into `data/raw/` (never modify them), then:
```bash
bash scripts/prepare_data.sh      # extract, combine, clean, split, build the WMT24++ and (if logged in) IN22-Gen test files
```
IN22-Gen is gated on Hugging Face: accept its terms, then `huggingface-cli login` (keep the token in `~/.cache/huggingface/token`,
mode 600, never inside the project folder). Without it, that last step is skipped.
The script regenerates the exact files used for the reported results (verified byte-for-byte).

## Experiments
```bash
GPU=1 nohup bash scripts/run_all.sh  > logs/run_all.log  2>&1 &   # baselines, 5 LoRA runs, full fine-tune, evals, release gates; idempotent
GPU=1 nohup bash scripts/run_in22.sh > logs/run_in22.log 2>&1 &   # IN22-Gen evals after run_all finishes
python -m src.evaluation.make_report                               # -> reports/experiment_results.md (about 9 min)
```
On a shared server check `nvidia-smi` first and pick a free GPU with `GPU=<id>`. Use `nohup` or `tmux` so jobs survive disconnects.
Timings on the L40S: LoRA r=8 trains in about 2 h (peak 32.8 GB); evaluating on the 78,751-sentence test set takes about 85 min per 1.3B model.
