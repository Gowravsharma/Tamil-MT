# Data quality report

Numbers come from `reports/data_quality.json` and `data/processed/split_report.json` (written by `src.data.clean` / `src.data.split`).

## Training / in-domain data: Anuvaad legal en→ta

Source: Anuvaad parallel corpus (project-anuvaad), the three **Legal** en–ta sets, downloaded as zips from the corpus S3 bucket into `data/raw/` (zips and extracted files are untouched). Files are line-aligned plain text (`.en` / `.ta`); alignment (equal line counts) was checked.

| Set | Raw pairs |
|---|---|
| ik-2021-v1 (Indian Kanoon judgments) | 816,731 |
| internal-judicial-2021-v1 (HC-SUVAS) | 10,851 |
| legal-terms-2021-v1 | 3,595 |
| **Total** | **831,177** |

`legal-terms` is not a clean glossary: it contains case headers and counsel names (0.4% of the data).

### Cleaning (`clean.py --domain legal --corpus-name anuvaad-legal`)

| Step | Pairs affected |
|---|---|
| Raw | 831,177 |
| Removed: empty | 5 |
| Removed: exact duplicate pairs | 27,161 |
| Removed: length (min 3, max 1200 chars) or length-ratio outside [1/3, 3] | 2,164 |
| Removed: target < 50% Tamil script | 9,446 |
| Removed: source == target | 0 |
| **Clean pairs** | **792,401** |
| Flagged, **not removed**: same source with different targets | 4,035 |

### Split (`split.py`)

| Split | Pairs |
|---|---|
| train | 634,920 |
| valid | 78,730 |
| test | 78,751 |

Train rows dropped for leakage: 0. Exact-source overlap train/valid, train/test, valid/test: 0 / 0 / 0.

## General-domain test sets (used only for evaluation, never for training or selection)

| Set | Segments | Notes |
|---|---|---|
| WMT24++ en→ta_IN (`google/wmt24pp`) | 960 | 998 rows minus 38 flagged `is_bad_source`; sub-domains: social 494, literary 206, news 149, speech 111 |
| IN22-Gen en→ta (`ai4bharat/IN22-Gen`, gated) | 1,024 | 13 domains (80 each, news 64), includes an 80-segment legal subset; exact-source overlap with train: 0 |

## Caveats

- **The split is by sentence, not by document** (the corpus has no document IDs). Sentences from the same judgment can be in train and test, and legal text is highly repetitive, so in-domain test scores are probably optimistic. Exact duplicates are removed across splits but near-duplicates and templated sentences are not.
- The references look machine-translated (odd phrasing in several inspected examples). A model fine-tuned on them learns that system's style, which a reference-based metric rewards. This is not the same as better translation.
- The Tamil-script and length filters are default heuristics and were not tuned.
- The 4,035 conflicting-source rows (same English, different Tamil) remain in the data.
- IN22-Gen is gated; keep it under `data/` (out of Git) and do not redistribute it.
