# Evaluation methods and published reference points (English → Tamil)

Prepared from papers found by web search on 2026-10-15. Numbers marked **[paper]** were read from the cited paper's PDF; numbers marked **[ours]** are from `reports/experiment_results.md`. Search-result summaries that were not checked against the full paper are labelled as such.

## 1. Sanity check of our pipeline against a published number

IndicTrans2 (Gala et al., TMLR 2023, [arXiv 2305.16307](https://arxiv.org/abs/2305.16307)), Table 12, chrF++ on **IN22-Gen**, English → Tamil (`tam_Taml`):

| System | chrF++ |
|---|---|
| IndicTrans1 | 45.5 **[paper]** |
| NLLB 1.2B distilled | 47.0 **[paper]** |
| NLLB 54.5B MoE | 47.5 **[paper]** |
| Google Translate | 48.5 **[paper]** |
| Azure Translate | 49.4 **[paper]** |
| IndicTrans2 | 49.5 **[paper]** |
| GPT-3.5 (Table 26) | 34.0 **[paper]** |
| **Our pretrained NLLB-200-distilled-1.3B** | **46.71 [ours]** |
| Our LoRA r=8 / r=16 / 600M full fine-tune (different base) | 46.64 / 46.65 / 45.84 (600M base 44.94) **[ours]** |

Our baseline is within 0.3 chrF++ of the published NLLB-1.2B-distilled figure, so the pipeline (data, decoding with beam 5, scoring) is reproducing known behaviour. Differences that remain: the paper tokenizes Indic text with IndicNLP before scoring and its model is the "1.2B distilled" (the checkpoint we use is named 1.3B distilled; I did not verify whether they are the same weights).

The paper's chrF++ signature is `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no`; our `sacrebleu` `CHRF(word_order=2)` uses the same settings (chrF++ = word n-grams up to 2 and character n-grams up to 6). Its BLEU for En→Indic uses `tok:none`, as ours does.

## 2. What the field uses to evaluate MT

- **chrF++ and BLEU (what we report).** The standard for Indic MT papers (IndicTrans2, NLLB). chrF++ is preferred for morphologically rich languages like Tamil. In the WMT23 metrics task, chrF and BLEU correlated with human judgments less well than neural metrics (average correlation about 0.69 for both, against 0.78–0.83 for COMET-family and MetricX-23 metrics; figures from a search-result summary of [WMT23 Metrics Shared Task](https://aclweb.org/anthology/2023.wmt-1.51.pdf), not re-checked against the full table).
- **Neural metrics (COMET, xCOMET, MetricX).** The usual complement to chrF++. We did not run them (the `unbabel-comet` package breaks pip dependency resolution on this server). Not having them is a gap.
- **Indian-language meta-evaluation.** [IndicMT Eval](https://arxiv.org/abs/2212.10180) (ACL 2023): 7,000 expert MQM annotations for English → 5 Indian languages including Tamil. It found that pretrained metrics such as COMET correlate better with human scores than n-gram metrics, but that existing metrics under-capture fluency errors in Indian languages. It also trained "Indic-COMET". (From a search-result summary; the dataset download status on the project page should be checked.)
- **Human evaluation.** MQM-style error annotation by Tamil speakers is the reference standard. We have not done this.
- **Significance testing.** Paired bootstrap resampling (used here, 1000 resamples) is the usual way to put confidence intervals on metric differences.

## 3. Benchmarks used in the literature for this pair
IN22-Gen, IN22-Conv, FLORES-200 (all in the IndicTrans2 paper), plus WMT24++ (used here). We used IN22-Gen and WMT24++; IN22-Conv and FLORES are gated and were not accessed.

## 4. Related work on LoRA for domain adaptation (search results, not verified in full text)
- LoRA on distilled NLLB for an oil-and-gas English–Norwegian domain: reported +24.62 BLEU over the base model with under 0.4% trainable parameters ([EAMT 2026](https://aclanthology.org/2026.eamt-1.25/)). In-domain gains only; forgetting is not measured.
- [SemiAdapt/SemiLoRA](https://arxiv.org/pdf/2510.18725) uses NLLB-600M for Irish domain adaptation.
- Searches did not turn up a paper that directly measures forgetting for LoRA-adapted NLLB. Our general-set before/after comparison is a reasonable design for that question, but it should not be presented as an established standard protocol.

## 5. How our results fit
- In-domain gains of +14 to +19 chrF++ are far larger than what papers report between strong systems on general benchmarks (IndicTrans2 vs NLLB: about 2 points). That is expected for same-corpus fine-tuning and is not comparable to those numbers; see the caveats in `MODEL_CARD.md` (sentence-level split, machine-looking references, no gain on the independent IN22-Gen legal subset).
- The general-domain figures (IN22-Gen 46.6–46.7 for LoRA vs 46.71 baseline) are the ones directly comparable with the published table, and show the adapted models staying at NLLB level rather than reaching IndicTrans2 level (49.5).

## 6. What is state of the art for English → Tamil? (checked 2026-10-15; incomplete)

There is no single authoritative leaderboard for this pair that I could find. What the primary sources support:

| Source | What it shows | Checked how |
|---|---|---|
| [IndicTrans2](https://arxiv.org/abs/2305.16307) (2023) | IN22-Gen en→ta chrF++ 49.5, above Google (48.5), Azure (49.4), NLLB-54.5B (47.5), GPT-3.5 (34.0) | Read from the paper's Table 12 / Table 26 |
| [Krutrim-Translate](https://huggingface.co/krutrim-ai-labs/Krutrim-Translate) (2025) | self-reported IN22-Gen en→ta chrF++ 48.3, IN22-Conv 38.7; small distilled model (6 enc / 3 dec layers) | Model card; no comparison table on the page; gated, custom licence |
| [Sarvam-Translate](https://huggingface.co/sarvamai/sarvam-translate) | open 4B model based on Gemma-3-4B-IT, 22 Indian languages, GPL-3.0 | Model card lists no scores or benchmark |
| [WMT24++](https://arxiv.org/abs/2502.12404) (the general set we used) | By MetricX-24, the frontier LLMs tested (OpenAI o1, Claude, Gemini; Sept 2024 snapshots) ranked highest in every one of the 55 languages, ahead of traditional MT providers; no human evaluation yet | Read the paper text; Tamil-specific scores are in figures I could not read |
| [WMT25 findings](https://aclanthology.org/2025.wmt-1.22/) | English→Indic pairs in the task were Bengali, Hindi, Kannada and Marathi (automatic-evaluation track only); **Tamil is not among them** | Read the pair list in the paper text |

Not verified / not used: an unofficial Hugging Face model card (`bodhan-ai/indic-translate`) that claims a ranking, a Substack comparison on FLORES, and any IndicTrans3 or TranslateGemma figures (no primary source found).

What this means for our models:
- Our general-domain results (IN22-Gen 46.6–46.7 chrF++) are at NLLB-1.2B level (47.0) and below IndicTrans2 (49.5) and the commercial systems. We make no claim to be state of the art on general en→ta translation.
- The published evidence suggests current frontier LLMs are the strongest general en→ta systems (WMT24++, MetricX-24, 2024 models). Newer models were not checked, and the LLM numbers are not chrF++ on IN22-Gen, so they cannot be placed in the table above.
- Our contribution is the adaptation study (parameter-efficient vs full fine-tuning, data efficiency, forgetting, release gate), not a new best system.
