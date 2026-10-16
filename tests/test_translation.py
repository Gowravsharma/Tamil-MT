"""Smoke test: LoRA injection on a tiny random M2M100/NLLB-architecture model (CPU, no download)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import M2M100Config, M2M100ForConditionalGeneration

from src.models.configure_lora import apply_lora, module_report
from src.models.load_model import count_params


def tiny():
    cfg = M2M100Config(vocab_size=100, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=64,
                       encoder_layerdrop=0.0, decoder_layerdrop=0.0)  # LayerDrop would randomly skip the only layer -> no grad
    return M2M100ForConditionalGeneration(cfg)


def test_lora_only_trains_adapters_and_covers_cross_attn():
    m = apply_lora(tiny(), {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]})
    p = count_params(m)
    assert 0 < p["trainable"] < 0.5 * p["total"]
    assert any("encoder_attn" in n for n in module_report(m)), "cross-attention not adapted"
    out = m(input_ids=torch.tensor([[5, 6, 7]]), labels=torch.tensor([[8, 9, 2]]))
    out.loss.backward()
