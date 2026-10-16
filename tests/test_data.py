import pandas as pd

from src.data.clean import clean, normalize
from src.data.split import split


def make_df():
    return pd.DataFrame({
        "source_text": ["Apply 5 kg urea per acre.", "Apply 5 kg urea per acre.", "Water the crop daily.", "", "Hello world", "Irrigate the field twice a week."] + [f"Sentence number {i} about rice farming" for i in range(60)],
        "target_text": ["ஏக்கருக்கு 5 கிலோ யூரியா இடவும்.", "ஏக்கருக்கு 5 கிலோ யூரியா இடவும்.", "பயிருக்கு தினமும் தண்ணீர் ஊற்றவும்.", "x", "Hello world", "வயலுக்கு வாரம் இருமுறை நீர் பாய்ச்சவும்."] + [f"நெல் சாகுபடி பற்றிய வாக்கியம் எண் {i}" for i in range(60)],
    })


def test_normalize_keeps_tamil():
    assert normalize("  ஏக்கருக்கு   5 ") == "ஏக்கருக்கு 5"


def test_clean_removes_dup_empty_untranslated():
    out, rep = clean(make_df())
    assert rep["removed_exact_duplicates"] == 1
    assert rep["removed_empty"] == 1
    assert rep["removed_target_not_tamil"] + rep["removed_source_equals_target"] >= 1
    assert not (out.source_text == "Hello world").any()


def test_split_no_overlap():
    out, rep = clean(make_df())
    parts, _ = split(out)
    s = {k: set(v.source_text) for k, v in parts.items()}
    assert not (s["train"] & s["valid"]) and not (s["train"] & s["test"]) and not (s["valid"] & s["test"])
