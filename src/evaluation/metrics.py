"""BLEU / chrF++ / optional COMET + paired bootstrap. Usage:
python -m src.evaluation.metrics outputs/preds/x.jsonl [--comet]
"""
import argparse
import json
import random
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

_bleu = BLEU(tokenize="none")  # Tamil: sacrebleu's default 13a is fine for spaces; 'none' avoids English-specific rules
_chrf = CHRF(word_order=2)     # chrF++


def load_preds(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r["source"] for r in rows], [r["hypothesis"] for r in rows], [r["reference"] for r in rows]


def score(hyps, refs):
    return {"bleu": _bleu.corpus_score(hyps, [refs]).score, "chrf++": _chrf.corpus_score(hyps, [refs]).score,
            "sacrebleu_signature": {"bleu": str(_bleu.get_signature()), "chrf": str(_chrf.get_signature())}}


def comet_score(srcs, hyps, refs, model_name="Unbabel/wmt22-comet-da", gpus=1):
    from comet import download_model, load_from_checkpoint
    m = load_from_checkpoint(download_model(model_name))
    out = m.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)], batch_size=32, gpus=gpus)
    return out.system_score


def paired_bootstrap(hyps_a, hyps_b, refs, n=1000, seed=0, metric="chrf++"):
    """P(B better than A) and 95% CI of (B - A) via paired resampling of test sentences.

    Per-sentence sufficient statistics are extracted once; each resample's corpus score is then computed from the
    summed statistics (identical to re-scoring the resampled corpus, but ~10^4x faster on large test sets).
    """
    import numpy as np
    m = _chrf if metric == "chrf++" else _bleu
    A = np.array(m._extract_corpus_statistics(hyps_a, [refs]), dtype=np.float64)
    B = np.array(m._extract_corpus_statistics(hyps_b, [refs]), dtype=np.float64)
    N = len(refs)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)  # resample multiplicities
        diffs[k] = m._compute_score_from_stats((w @ B).tolist()).score - m._compute_score_from_stats((w @ A).tolist()).score
    diffs.sort()
    return {"mean_diff": float(diffs.mean()), "ci95": [float(diffs[int(0.025 * n)]), float(diffs[int(0.975 * n)])],
            "p_b_better": float((diffs > 0).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preds"); p.add_argument("--comet", action="store_true")
    a = p.parse_args()
    srcs, hyps, refs = load_preds(a.preds)
    res = score(hyps, refs)
    if a.comet:
        res["comet"] = comet_score(srcs, hyps, refs)
    Path(a.preds).with_suffix(".metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
