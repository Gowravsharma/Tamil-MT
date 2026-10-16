"""Translate a JSONL test set. Usage:
python -m src.inference.translate --config configs/baseline.yaml --input data/processed/test.jsonl --out outputs/preds/baseline_domain.jsonl [--adapter outputs/lora_r8/final]
For a full-FT model, pass --model-path instead of --adapter.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.models.load_model import load_model, load_tokenizer


def translate(model, tok, texts, tgt_lang, batch_size=32, num_beams=5, max_new_tokens=256):
    model.eval()
    dev = next(model.parameters()).device
    forced = tok.convert_tokens_to_ids(tgt_lang)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucket for speed
    outs = [None] * len(texts)
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=num_beams, max_new_tokens=max_new_tokens)
        for i, o in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            outs[i] = o
    return outs, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--input", required=True); p.add_argument("--out", required=True)
    p.add_argument("--adapter"); p.add_argument("--model-path")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    name = a.model_path or cfg["model_name"]
    tok = load_tokenizer(name, cfg["src_lang"], cfg["tgt_lang"])
    model = load_model(name, bf16=cfg["bf16"], adapter_path=a.adapter or cfg.get("adapter_path"))
    if torch.cuda.is_available():
        model.cuda(); torch.cuda.reset_peak_memory_stats()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    hyps, secs = translate(model, tok, [r["source_text"] for r in rows], cfg["tgt_lang"],
                           cfg["batch_size"], cfg["num_beams"], cfg["max_new_tokens"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"source": r["source_text"], "reference": r["target_text"], "hypothesis": h,
                                "domain": r.get("domain")}, ensure_ascii=False) + "\n")
    meta = {"sentences": len(rows), "seconds": secs, "sec_per_sentence": secs / max(1, len(rows)),
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
