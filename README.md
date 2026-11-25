# GitHub Issue Auto-Labeler & Duplicate Finder

An NLP system (Industry Track, milestones M1–M6) that:

1. predicts a standardized issue category from the title and body; and
2. retrieves earlier, semantically similar issues and recommends duplicate/manual review.

The project deliberately separates **classification labels** from **verified duplicate
relationships**. A label such as `duplicate` is useful evidence, but it does not by itself
identify the canonical issue.

**Start here:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) explains the whole project, the demo
design, GitHub rate limits and the M3–M6 plan in plain language.

| Milestone | Status | Evidence |
| --- | --- | --- |
| M1 Problem definition | ✅ Done | `docs/m1_m2/M1_M2_PROJECT_HANDBOOK.md` |
| M2 Data preparation | ✅ Pilot done; full collection needs a token | `docs/m1_m2/M2_EDA_REPORT.md`, `data/raw/pilot/` |
| M3 Models | ✅ Built and run on the pilot data | [`outputs/M3_RESULTS.md`](outputs/M3_RESULTS.md) |
| M4 Evaluation | ✅ Done on the pilot (duplicate Recall@5 waits for the token) | [`outputs/M4_EVALUATION.md`](outputs/M4_EVALUATION.md) |
| M5 Deployment (FastAPI + GitHub webhook) | ✅ Built and tested; the live bot needs your demo repo | [`outputs/M5_PROTOTYPE.md`](outputs/M5_PROTOTYPE.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| M6 Final delivery | ⏳ After the full data collection | |

## Run the app

```bash
.venv/bin/uvicorn issue_triage.service.app:app --app-dir src --port 8000
```

Open http://localhost:8000. Setting up the live GitHub bot: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Set up and train (M3/M4)

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_dataset.py --input data/raw/pilot \
  --taxonomy config/label_taxonomy.json --output data/processed
.venv/bin/python scripts/train_models.py        # 8 candidates, 5-fold CV selection -> models/
.venv/bin/python scripts/evaluate_retrieval.py  # TF-IDF vs BM25 vs embeddings vs hybrid
.venv/bin/python scripts/evaluate_models.py     # M4: test set, errors, robustness, thresholds
.venv/bin/python -m unittest discover -s tests  # 33 tests
```

To create a GitHub token, see [`docs/GITHUB_TOKEN_SETUP.md`](docs/GITHUB_TOKEN_SETUP.md).

## What M1–M2 delivered

- Auditable repository and label-taxonomy configuration.
- GitHub REST collection with pagination, caching, rate-limit logging, checksums, and
  collection manifests.
- A lightweight search-API pilot mode for collecting 100 recent issues per repository.
- Deterministic preparation, cleaning, taxonomy mapping, quality checks, and a chronological
  70/15/15 classification split.
- A presentation handbook and an automatically generated M2 EDA summary.

## Quick start

The M1–M2 pipeline uses only the Python standard library.

```bash
# Small pilot: one Search API request per repository, at most 100 issues each
python3 scripts/collect_github.py \
  --mode search \
  --repositories config/repositories.json \
  --output data/raw/pilot

# Full history: paginated repository Issues API
export GITHUB_TOKEN="your-read-only-token"
python3 scripts/collect_github.py \
  --mode issues \
  --repositories config/repositories.json \
  --output data/raw/full \
  --max-pages 20

# Better class balance: collect configured source labels separately, then deduplicate
python3 scripts/collect_github.py \
  --mode labeled \
  --repositories config/repositories.json \
  --taxonomy config/label_taxonomy.json \
  --output data/raw/labeled \
  --max-pages 5

# Clean, map labels, assign chronological splits, and create EDA
python3 scripts/verify_manifest.py data/raw/pilot/collection_manifest.json
python3 scripts/prepare_dataset.py \
  --input data/raw/pilot \
  --taxonomy config/label_taxonomy.json \
  --output data/processed
python3 scripts/generate_eda.py \
  --data data/processed/issues_clean.csv \
  --quality data/processed/data_quality.json \
  --output outputs/M2_EDA_REPORT.md

# After authenticated collection, resolve structured duplicate pairs via GraphQL
python3 scripts/collect_duplicate_pairs.py \
  --issues data/processed/issues_clean.csv \
  --output data/processed/duplicate_pairs.csv

# Run the test suite
python3 -m unittest discover -s tests -v
```

`GITHUB_TOKEN` is optional for public data, but GitHub currently allows only 60 core REST
requests per hour without authentication versus 5,000 per hour for a typical authenticated
user. Never commit a token.

## Important folders

- `config/`: selected repositories and the auditable label/Issue-Type taxonomy.
- `scripts/`: collection, preparation, and reporting entry points.
- `src/issue_triage/`: reusable collection and preparation logic.
- `data/raw/`: immutable snapshots returned by GitHub.
- `data/processed/`: cleaned CSVs and quality metadata.
- `docs/`: project plan, token guide, and the M1/M2 handbook and reports.
- `outputs/`: milestone results (M3 onwards) and the NLBSE baseline reproduction.
- `experiments/experiment_log.jsonl`: one line per training/evaluation run.
- `models/`: trained model bundle (not committed; rebuild with `scripts/train_models.py`).
- `tests/`: cross-checks for cleaning, mapping, and leakage-sensitive fields.

## Presentation route

Open `docs/m1_m2/M1_M2_PROJECT_HANDBOOK.md` first. It contains the problem framing,
research decisions, data provenance, architecture, demo sequence, likely viva questions,
and the exact boundary between completed work and later milestones.

