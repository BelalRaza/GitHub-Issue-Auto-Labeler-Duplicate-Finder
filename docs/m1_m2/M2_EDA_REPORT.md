# Milestone 2 EDA — Pilot Snapshot

Generated from the saved GitHub snapshot and its checksum manifest. This is a **pipeline-validation
sample**, not the final modeling corpus: Search API collection is capped at 100 recent issues per
repository and therefore over-represents current triage practice.

## Dataset health

- Rows: **1000**
- Unique `(repository, issue_number)` keys: **1000**
- Exact duplicate keys: **0**
- Classification-eligible rows: **524**
- Missing bodies: **3**
- Cross-class taxonomy conflicts excluded: **3**
- Target sources (labels / Issue Types / both): **{'issue_type': 7, 'labels': 516, 'labels+issue_type': 1}**
- Issues closed with GitHub's structured `state_reason=duplicate`: **18**
- Issues carrying a duplicate-like label: **5**
- Eligible text length: median **350** words; p90
  **733**; maximum **2998**.

## Coverage by repository and standardized target

| Repository | All rows | Bug | Feature | Documentation | Question |
| --- | --- | --- | --- | --- | --- |
| ansible/ansible | 100 | 66 | 10 | 5 | 0 |
| flutter/flutter | 100 | 6 | 0 | 0 | 0 |
| godotengine/godot | 100 | 69 | 4 | 2 | 0 |
| huggingface/transformers | 100 | 41 | 9 | 0 | 0 |
| kubernetes/kubernetes | 100 | 45 | 8 | 1 | 0 |
| microsoft/TypeScript | 100 | 39 | 4 | 0 | 1 |
| microsoft/vscode | 100 | 8 | 11 | 0 | 0 |
| pandas-dev/pandas | 100 | 56 | 6 | 11 | 0 |
| rust-lang/rust | 100 | 57 | 6 | 4 | 0 |
| scikit-learn/scikit-learn | 100 | 34 | 9 | 12 | 0 |

## Target balance

| Target | Rows | Share of eligible |
| --- | --- | --- |
| Bug | 421 | 80.3% |
| Documentation | 35 | 6.7% |
| Feature | 67 | 12.8% |
| Question | 1 | 0.2% |

## Provisional chronological split

| Split | Rows |
| --- | --- |
| test | 80 |
| train | 366 |
| validation | 78 |

## Most common original labels

| Original label | Rows |
| --- | --- |
| bug | 183 |
| Bug | 129 |
| needs-triage | 74 |
| C-bug | 58 |
| T-compiler | 56 |
| Needs Triage | 49 |
| kind/bug | 46 |
| topic:editor | 38 |
| has_pr | 36 |
| Needs Investigation | 31 |
| triage/accepted | 29 |
| sig/node | 27 |
| has reproducible steps | 25 |
| needs testing | 25 |
| triage-needed | 25 |
| T-libs | 25 |
| archived | 22 |
| bot_closed | 20 |
| regression | 20 |
| needs_template | 19 |

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

- Pilot search snapshots contain only the newest issues and are not a representative training corpus.
- Current labels are retrospective targets; label assignment timestamps are not available in the issue list response.
- Verified duplicate status does not always expose the canonical issue in REST list results; use GraphQL duplicateOf or audited maintainer comments for pair extraction.
- The chronological split is provisional until the full-history collection and duplicate-pair audit are complete.
