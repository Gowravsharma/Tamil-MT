"""LoRA / full fine-tuning of an NLLB-style seq2seq model.  Usage: python -m src.training.train configs/lora_r8.yaml"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.models.configure_lora import apply_lora, module_report
from src.models.load_model import count_params, load_model, load_tokenizer


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tok = load_tokenizer(cfg["model_name"], cfg["src_lang"], cfg["tgt_lang"])
    model = load_model(cfg["model_name"], bf16=False)  # keep fp32 master weights; Trainer handles bf16 autocast
    if cfg["adaptation"] == "lora":
        model = apply_lora(model, cfg["lora"])
        print("LoRA modules:", len(module_report(model)))
    params = count_params(model)
    print("params:", params)

    ds = load_dataset("json", data_files={"train": cfg["train_file"], "valid": cfg["valid_file"]})
    if cfg["train_fraction"] < 1.0:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(int(len(ds["train"]) * cfg["train_fraction"])))

    def encode(batch):
        # text_target tokenizes labels with tgt_lang tag; model shifts labels itself -> don't shift manually
        return tok(batch["source_text"], text_target=batch["target_text"], max_length=cfg["max_len"], truncation=True)

    if cfg.get("valid_samples"):  # fixed random validation subset (validation only, never test) to keep periodic eval cheap
        ds["valid"] = ds["valid"].shuffle(seed=0).select(range(min(cfg["valid_samples"], len(ds["valid"]))))

    ds = ds.map(encode, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)  # pads masked from loss

    out = Path(cfg["output_dir"])
    args = Seq2SeqTrainingArguments(
        output_dir=str(out), seed=seed, learning_rate=float(cfg["lr"]), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch"], per_device_eval_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"], bf16=cfg["bf16"],
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_steps=cfg["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=cfg.get("logging_steps", 50), report_to=["tensorboard"], warmup_steps=0.03, lr_scheduler_type="linear",
        remove_unused_columns=False,
        train_sampling_strategy="group_by_length" if cfg.get("group_by_length") else "random",  # ~2x fewer padded tokens
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["valid"],
                             data_collator=collator, processing_class=tok)
    # M2M100/NLLB forward accepts **kwargs but ignores num_items_in_batch, so with grad_accum>1 the Trainer would
    # skip dividing the loss by the accumulation steps (loss and gradients come out grad_accum x too large).
    trainer.model_accepts_loss_kwargs = False
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    trainer.train(resume_from_checkpoint=True if list(out.glob('checkpoint-*')) else None)  # resume if a previous run was interrupted
    stats = {
        "run_name": cfg["run_name"], "config": cfg, "params": params,
        "train_seconds": time.time() - t0,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "train_examples": len(ds["train"]), "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    trainer.save_model(str(out / "final"))  # LoRA: adapter only; full FT: whole model
    tok.save_pretrained(str(out / "final"))
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
