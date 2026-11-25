# GitHub Issue Auto-Labeler & Duplicate Finder

Milestones 1–2 foundation for an NLP system that:

1. predicts a standardized issue category from the title and body; and
2. retrieves earlier, semantically similar issues and recommends duplicate/manual review.

The project deliberately separates **classification labels** from **verified duplicate
relationships**. A label such as `duplicate` is useful evidence, but it does not by itself
identify the canonical issue.

## What is ready

- Auditable repository and label-taxonomy configuration.
- GitHub REST collection with pagination, caching, rate-limit logging, checksums, and
  collection manifests.
- A lightweight search-API pilot mode for collecting 100 recent issues per repository.
- Deterministic preparation, cleaning, taxonomy mapping, quality checks, and a chronological
  70/15/15 classification split.
- A presentation handbook and an automatically generated M2 EDA summary.

## Quick start

The pipeline uses only the Python standard library for Milestones 1–2.

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
- `outputs/`: presentation-ready handbook and reports.
- `tests/`: cross-checks for cleaning, mapping, and leakage-sensitive fields.

## Presentation route

Open `outputs/M1_M2_PROJECT_HANDBOOK.md` first. It contains the problem framing,
research decisions, data provenance, architecture, demo sequence, likely viva questions,
and the exact boundary between completed work and later milestones.

