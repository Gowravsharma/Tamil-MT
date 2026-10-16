"""Release gate: compare a candidate against the previous release on a fixed suite; exit 1 on regression.

python -m src.evaluation.compare_releases \
   --prev domain=outputs/preds/baseline_domain.jsonl general=outputs/preds/baseline_general.jsonl \
   --cand domain=outputs/preds/lora_r8_domain.jsonl  general=outputs/preds/lora_r8_general.jsonl \
   --thresholds configs/regression_thresholds.yaml --report reports/release_comparison.md
"""
import argparse
import sys

import yaml

from src.evaluation.error_analysis import analyse
from src.evaluation.metrics import load_preds, paired_bootstrap, score


def parse_pairs(items):
    return dict(i.split("=", 1) for i in items)


def compare(prev, cand, th):
    findings, lines = [], ["| set | metric | prev | cand | delta | 95% CI (chrF++) |", "|---|---|---|---|---|---|"]
    for name, cfg in th["test_sets"].items():
        _, hp, refs = load_preds(prev[name]); _, hc, refs2 = load_preds(cand[name])
        assert refs == refs2, f"{name}: prev/cand evaluated on different references"
        sp, sc = score(hp, refs), score(hc, refs)
        bs = paired_bootstrap(hp, hc, refs)
        for m in ("chrf++", "bleu"):
            d = sc[m] - sp[m]
            ci = f"[{bs['ci95'][0]:.2f}, {bs['ci95'][1]:.2f}]" if m == "chrf++" else ""
            lines.append(f"| {name} | {m} | {sp[m]:.2f} | {sc[m]:.2f} | {d:+.2f} | {ci} |")
        d = sc["chrf++"] - sp["chrf++"]
        if d < -cfg["max_drop"]:
            findings.append(f"REGRESSION: {name} chrF++ dropped {d:.2f} (allowed {cfg['max_drop']})")
        if d < cfg["min_gain"]:
            findings.append(f"INSUFFICIENT GAIN: {name} chrF++ {d:+.2f} < required {cfg['min_gain']}")
        rows = [dict(source=s, hypothesis=h, reference=r) for s, h, r in zip(*load_preds(cand[name]))]
        summ, _ = analyse(rows)
        chk = th["checks"]
        if summ["rates"].get("untranslated_or_latin", 0) > chk["max_untranslated_rate"]:
            findings.append(f"{name}: untranslated rate {summ['rates']['untranslated_or_latin']:.3f} > {chk['max_untranslated_rate']}")
        if summ["rates"].get("number_mismatch", 0) > chk["max_number_mismatch_rate"]:
            findings.append(f"{name}: number mismatch rate {summ['rates']['number_mismatch']:.3f} > {chk['max_number_mismatch_rate']}")
    return findings, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prev", nargs="+", required=True); p.add_argument("--cand", nargs="+", required=True)
    p.add_argument("--thresholds", required=True); p.add_argument("--report", required=True)
    a = p.parse_args()
    th = yaml.safe_load(open(a.thresholds))
    findings, table = compare(parse_pairs(a.prev), parse_pairs(a.cand), th)
    status = "FAIL" if findings else "PASS"
    md = ["# Release comparison", f"**Gate: {status}**", ""] + table + [""] + [f"- {f}" for f in findings]
    open(a.report, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
