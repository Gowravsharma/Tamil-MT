import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_tokenizer(model_name, src_lang, tgt_lang):
    return AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)


def load_model(model_name, bf16=True, adapter_path=None):
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "pct": 100 * trainable / total}
