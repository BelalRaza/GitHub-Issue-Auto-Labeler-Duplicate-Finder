#!/usr/bin/env python3
"""Generate a dependency-free Markdown EDA report from the prepared pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    with args.data.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    repo_targets: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        target = row["classification_target"]
        if target:
            repo_targets[row["repository"]][target] += 1

    classes = ["Bug", "Feature", "Documentation", "Question"]
    repository_rows = []
    for repository, total in quality["repositories"].items():
        counts = repo_targets[repository]
        repository_rows.append([repository, total, *[counts.get(name, 0) for name in classes]])

    target_total = sum(quality["target_distribution"].values()) or 1
    target_rows = [
        [name, count, f"{count / target_total:.1%}"]
        for name, count in sorted(quality["target_distribution"].items())
    ]
    split_rows = [[name, count] for name, count in sorted(quality["split_distribution"].items())]
    top_label_rows = [[name, count] for name, count in quality["top_original_labels"][:20]]

    report = f"""# Milestone 2 EDA — Pilot Snapshot

Generated from the saved GitHub snapshot and its checksum manifest. This is a **pipeline-validation
sample**, not the final modeling corpus: Search API collection is capped at 100 recent issues per
repository and therefore over-represents current triage practice.

## Dataset health

- Rows: **{quality['total_rows']}**
- Unique `(repository, issue_number)` keys: **{quality['unique_repository_issue_keys']}**
- Exact duplicate keys: **{quality['duplicate_rows_by_key']}**
- Classification-eligible rows: **{quality['classification_eligible_rows']}**
- Missing bodies: **{quality['missing_bodies']}**
- Cross-class taxonomy conflicts excluded: **{quality['taxonomy_conflicts']}**
- Target sources (labels / Issue Types / both): **{quality['target_source_distribution']}**
- Issues closed with GitHub's structured `state_reason=duplicate`: **{quality['verified_duplicate_status_rows']}**
- Issues carrying a duplicate-like label: **{quality['duplicate_label_rows']}**
- Eligible text length: median **{quality['text_length_words']['median']}** words; p90
  **{quality['text_length_words']['p90']}**; maximum **{quality['text_length_words']['max']}**.

## Coverage by repository and standardized target

{md_table(['Repository', 'All rows', *classes], repository_rows)}

## Target balance

{md_table(['Target', 'Rows', 'Share of eligible'], target_rows)}

## Provisional chronological split

{md_table(['Split', 'Rows'], split_rows)}

## Most common original labels

{md_table(['Original label', 'Rows'], top_label_rows)}

## Interpretation and decisions

1. The repository list is a candidate panel, not a promise that all ten repositories belong in
   the final classifier. Repositories without enough examples in the four-class taxonomy should
   be removed or used only for held-out evaluation.
2. Only issues mapping to exactly one standardized class are classification-eligible. Original
   labels remain intact so every mapping can be audited.
3. `state_reason=duplicate` is reliable positive-status evidence. A canonical pair requires the
   GraphQL `duplicateOf` field or a separately audited maintainer reference; similarity alone is
   never ground truth.
4. The final collection must use paginated full history (or a documented time window), then rerun
   this report before any target or split is frozen.

## Known limitations

""" + "\n".join(f"- {warning}" for warning in quality["warnings"]) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

