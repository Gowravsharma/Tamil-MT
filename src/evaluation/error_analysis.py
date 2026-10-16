"""Automatic failure heuristics + a stratified sample for manual error-taxonomy review by a Tamil speaker."""
import argparse
import json
import random
import re
from pathlib import Path

NUM = re.compile(r"\d+(?:[.,]\d+)?")
TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")
LATIN = re.compile(r"[A-Za-z]")
TAMIL = re.compile(r"[஀-௿]")


def latin_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(bool(LATIN.match(c)) for c in letters) / len(letters) if letters else 0.0


def numbers(t):
    return set(NUM.findall(t.translate(TAMIL_DIGITS)))


def flags(row):
    src, hyp, ref = row["source"], row["hypothesis"], row["reference"]
    f = []
    if not hyp.strip():
        f.append("empty")
    if latin_ratio(hyp) > 0.5:
        f.append("untranslated_or_latin")
    if hyp.strip().lower() == src.strip().lower():
        f.append("copied_source")
    if numbers(src) - numbers(hyp):
        f.append("number_mismatch")
    if len(hyp) < 0.4 * max(1, len(ref)):
        f.append("possible_omission")
    if len(hyp) > 2.0 * max(1, len(ref)):
        f.append("possible_addition_or_repetition")
    if re.search(r"(\b\S+\b)(?:\s+\1){3,}", hyp):
        f.append("repetition_loop")
    return f


def analyse(rows):
    counts, flagged = {}, []
    for r in rows:
        fl = flags(r)
        for x in fl:
            counts[x] = counts.get(x, 0) + 1
        if fl:
            flagged.append({**r, "flags": fl})
    n = max(1, len(rows))
    return {"n": len(rows), "rates": {k: v / n for k, v in counts.items()}, "counts": counts}, flagged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preds"); p.add_argument("--sample", type=int, default=100)
    a = p.parse_args()
    rows = [json.loads(l) for l in open(a.preds, encoding="utf-8")]
    summary, flagged = analyse(rows)
    random.Random(0).shuffle(flagged)
    base = Path(a.preds).with_suffix("")
    Path(f"{base}.errors.json").write_text(json.dumps(summary, indent=2))
    # manual-review sheet: flagged first, then random unflagged, with empty taxonomy column to fill in
    review = flagged[: a.sample // 2] + random.Random(1).sample(rows, min(a.sample // 2, len(rows)))
    with open(f"{base}.review.jsonl", "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps({**r, "error_category": None}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
