import json

from src.evaluation.compare_releases import compare
from src.evaluation.error_analysis import flags
from src.evaluation.metrics import score

TH = {"test_sets": {"domain": {"max_drop": 0.0, "min_gain": 0.0}},
      "checks": {"max_untranslated_rate": 0.5, "max_number_mismatch_rate": 0.5, "max_latin_ratio": 0.3}}


def write(path, hyps):
    refs = ["நெல் சாகுபடி இங்கே", "பயிருக்கு தண்ணீர் ஊற்றவும்"]
    with open(path, "w", encoding="utf-8") as f:
        for h, r in zip(hyps, refs):
            f.write(json.dumps({"source": "rice crop","hypothesis": h, "reference": r}, ensure_ascii=False) + "\n")


def test_identical_models_do_not_regress(tmp_path):
    write(tmp_path / "a.jsonl", ["நெல் சாகுபடி இங்கே", "பயிருக்கு தண்ணீர் ஊற்றவும்"])
    findings, _ = compare({"domain": str(tmp_path / "a.jsonl")}, {"domain": str(tmp_path / "a.jsonl")}, TH)
    assert findings == []


def test_degraded_model_is_flagged(tmp_path):
    write(tmp_path / "a.jsonl", ["நெல் சாகுபடி இங்கே", "பயிருக்கு தண்ணீர் ஊற்றவும்"])
    write(tmp_path / "b.jsonl", ["completely wrong", "also wrong"])
    findings, _ = compare({"domain": str(tmp_path / "a.jsonl")}, {"domain": str(tmp_path / "b.jsonl")}, TH)
    assert any("REGRESSION" in f for f in findings)


def test_flags_detect_untranslated_and_number():
    f = flags({"source": "Apply 5 kg", "hypothesis": "Apply kg", "reference": "5 கிலோ இடவும்"})
    assert "untranslated_or_latin" in f and "number_mismatch" in f


def test_score_perfect():
    assert score(["அ ஆ"], ["அ ஆ"])["chrf++"] > 99
