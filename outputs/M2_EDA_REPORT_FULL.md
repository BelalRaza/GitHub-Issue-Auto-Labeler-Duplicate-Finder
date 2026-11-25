# Milestone 2 EDA — Full Corpus

Generated from four checksum-verified snapshots (pilot, label-targeted, recent history, and the
canonical originals of verified duplicates), merged and de-duplicated by (repository, issue).
Label-targeted collection deliberately over-samples rare classes, so class shares below describe
the **training corpus**, not how often each type occurs on GitHub.

## Dataset health

- Rows: **16302**
- Unique `(repository, issue_number)` keys: **16302**
- Exact duplicate keys: **0**
- Classification-eligible rows: **12569**
- Missing bodies: **77**
- Cross-class taxonomy conflicts excluded: **548**
- Target sources (labels / Issue Types / both): **{'issue_type': 54, 'labels': 12423, 'labels+issue_type': 92}**
- Issues closed with GitHub's structured `state_reason=duplicate`: **216**
- Issues carrying a duplicate-like label: **89**
- Eligible text length: median **185** words; p90
  **527**; maximum **5099**.

## Coverage by repository and standardized target

| Repository | All rows | Bug | Feature | Documentation | Question |
| --- | --- | --- | --- | --- | --- |
| RocketChat/Rocket.Chat | 110 | 40 | 65 | 0 | 0 |
| ansible-collections/ansible.posix | 1 | 0 | 0 | 0 | 0 |
| ansible-collections/ansible.utils | 1 | 0 | 0 | 0 | 0 |
| ansible/ansible | 1046 | 645 | 169 | 158 | 0 |
| directus/directus | 471 | 290 | 116 | 0 | 0 |
| excalidraw/excalidraw | 696 | 296 | 297 | 0 | 0 |
| flutter/flutter | 738 | 47 | 1 | 0 | 0 |
| godotengine/godot | 878 | 561 | 19 | 94 | 0 |
| godotengine/godot-docs | 2 | 1 | 1 | 0 | 0 |
| godotengine/godot-proposals | 2 | 0 | 0 | 0 | 0 |
| huggingface/transformers | 1479 | 499 | 667 | 44 | 0 |
| kubernetes/kubernetes | 923 | 164 | 66 | 49 | 302 |
| microsoft/TypeScript | 1985 | 502 | 517 | 7 | 499 |
| microsoft/vscode | 2426 | 139 | 498 | 0 | 979 |
| nextcloud/server | 205 | 71 | 110 | 0 | 0 |
| outline/outline | 293 | 96 | 192 | 0 | 0 |
| pandas-dev/pandas | 591 | 106 | 336 | 77 | 0 |
| rust-lang/goals | 1 | 0 | 0 | 0 | 0 |
| rust-lang/libs-team | 1 | 0 | 0 | 0 | 0 |
| rust-lang/rust | 2033 | 514 | 934 | 294 | 0 |
| scikit-learn/scikit-learn | 1785 | 479 | 895 | 68 | 115 |
| supabase/supabase | 635 | 293 | 243 | 14 | 0 |

## Target balance

| Target | Rows | Share of eligible |
| --- | --- | --- |
| Bug | 4743 | 37.7% |
| Documentation | 805 | 6.4% |
| Feature | 5126 | 40.8% |
| Question | 1895 | 15.1% |

## Chronological split

| Split | Rows |
| --- | --- |
| test | 1901 |
| train | 8791 |
| validation | 1877 |

## Most common original labels

| Original label | Rows |
| --- | --- |
| bug | 2692 |
| Bug | 1400 |
| enhancement | 946 |
| needs-triage | 857 |
| T-compiler | 797 |
| Enhancement | 770 |
| Question | 621 |
| C-bug | 586 |
| Feature request | 506 |
| *question | 504 |
| Suggestion | 502 |
| feature-request | 501 |
| kind/support | 498 |
| C-feature-request | 498 |
| New Feature | 498 |
| C-enhancement | 492 |
| *dev-question | 485 |
| T-libs | 444 |
| Needs Triage | 420 |
| has_pr | 415 |

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
