from peft import LoraConfig, TaskType, get_peft_model


def apply_lora(model, cfg):
    lc = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"], bias="none",
    )
    model = get_peft_model(model, lc)
    # fail loudly if injection missed: only LoRA params should be trainable
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    assert not bad, f"non-LoRA trainable params: {bad[:5]}"
    return model


def module_report(model):
    """Which module names got adapters (verify cross-attention is covered)."""
    return sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters() if ".lora_" in n})
