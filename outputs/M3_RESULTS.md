# Milestone 3 — Baseline & Model Development: first results

> **Update (M4):** M4 found that picking a model on this small validation set is unreliable. It
> switched to cross-validation and added an ensemble, which is now the model in use. See
> [`M4_EVALUATION.md`](M4_EVALUATION.md). This file records the M3 state as it was.

*Run on 25 Nov 2026 on the M2 **pilot** data (1,000 issues, 10 repositories).*
*Every run is logged in `experiments/experiment_log.jsonl`.*

> **Read this first:** these numbers prove the pipeline works end to end. They are **not** final
> results. The validation set has only **6 Feature and 2 Documentation** issues, so one
> prediction more or less moves Macro-F1 by about 0.1. We re-run everything after the bigger,
> authenticated data collection. The test split has **not** been touched; it is saved for M4.

## How to reproduce

```bash
.venv/bin/python scripts/train_models.py        # classifiers + search index -> models/
.venv/bin/python scripts/evaluate_retrieval.py  # compares 4 search methods
.venv/bin/python -m unittest discover -s tests  # 18 tests
```

## Task 1 — Classification (Bug / Feature / Documentation)

Training: 366 issues. Validation: 77 issues (69 Bug, 6 Feature, 2 Documentation).

| Model | Accuracy | **Macro-F1** | Weighted F1 | F1 Bug | F1 Feature | F1 Docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class (always "Bug") | 0.896 | 0.315 | 0.847 | 0.945 | 0.000 | 0.000 |
| **TF-IDF word+char n-grams + LogReg** (C=4) | **0.948** | **0.768** | **0.940** | 0.972 | 0.667 | 0.667 |
| MiniLM sentence embeddings + LogReg (C=4) | 0.935 | 0.751 | 0.935 | 0.971 | 0.615 | 0.667 |

C was tried at 1, 4 and 16 for both models. All 7 runs are in the log.

**What this tells us:**

- **Why we use Macro-F1:** "always Bug" scores 89.6% accuracy but 0.315 Macro-F1. It never
  gets a single Feature or Docs issue right.
- **TF-IDF beats the embedding model for now.** With only 45 Feature and 28 Docs training
  examples, a simple model that matches exact words ("docs", "README", "support for") works best.
  The embedding model may win once we have more data, and M4 checks that.
- **Selected for the app:** TF-IDF + LogReg. It scores higher, trains in 2 seconds, and needs no GPU.

## Task 2 — Similar-issue retrieval

Real duplicate pairs need GitHub's GraphQL API, which only works with a token. Until we have
one, we run a **proxy check**: search with an issue's *title* and see whether we find that
issue's own *body* among 981 bodies. This tests that the search engine works. It is **not** a
duplicate-detection score.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | ms / query |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF cosine | 0.863 | **0.943** | 0.956 | **0.900** | 1.7 |
| BM25 | 0.863 | 0.939 | 0.952 | 0.897 | 0.7 |
| MiniLM embeddings | 0.781 | 0.904 | 0.932 | 0.835 | 9.8 |
| Hybrid (0.6 embedding + 0.4 TF-IDF) | 0.853 | 0.940 | **0.960** | 0.893 | 9.7 |

**What this tells us:** a title shares many exact words with its own body, so this proxy favours
word-matching methods. Real duplicates are often written by different people in different
words. A hand-made example shows this:

| | Text |
| --- | --- |
| Issue A | "App crashes while uploading large images … 20MB PNG … OutOfMemoryError" |
| Issue B | "Crash when I upload a big photo … large picture … out of memory" |

The hybrid search ranked A **first** for B, out of 1,001 issues. That is the point of the
semantic part. But the score was 0.60, below our provisional 0.70 "manual review" threshold,
so the decision was "no duplicate found". **The thresholds in `config/model_config.json` are
guesses.** In M4 we set them using real duplicate pairs from the validation split.

## Engineering delivered in M3

| Deliverable | Where |
| --- | --- |
| Majority, TF-IDF and embedding classifiers | `src/issue_triage/classification.py` |
| TF-IDF, BM25, embedding and hybrid retrieval; past-only search; self-exclusion | `src/issue_triage/retrieval.py` |
| Metrics (Macro-F1, per-class, confusion matrix, Recall@K, MRR) | `src/issue_triage/metrics.py` |
| One `predict()` core for the CLI, API and webhook | `src/issue_triage/predictor.py` |
| Training + model selection on validation only | `scripts/train_models.py` |
| Retrieval evaluation (real pairs or proxy) | `scripts/evaluate_retrieval.py` |
| Experiment log (model, settings, data hash, git commit, metrics) | `experiments/experiment_log.jsonl` |
| Settings and thresholds, versioned | `config/model_config.json` |
| 11 new tests (18 total) | `tests/test_models.py` |

**Safety behaviour already built into `predict()`:**

- Empty issue → clear error.
- Very short issue ("help pls") or low confidence → `needs_manual_review`, no label.
- Weak similarity → `no_duplicate_found`; the system never makes up a duplicate.
- If the embedding model can't load, search falls back to TF-IDF and the service keeps running.
- New issues can be added live (`remember()`), so a later near-copy can be matched.

**Measured speed** (MacBook, CPU only, hybrid search over 1,000 issues): **p50 11 ms, p95 14 ms**
per prediction after a one-time 8-second model load at startup. The target was ≤ 2 s.

## Still to do (M3 on full data → M4)

1. Run the authenticated collection, then re-run both scripts on the bigger dataset.
2. Collect `duplicate_pairs.csv`, then run `evaluate_retrieval.py --pairs …` for real duplicate scores.
3. M4: evaluate on the test split once, tune thresholds on validation, error analysis,
   robustness tests.
