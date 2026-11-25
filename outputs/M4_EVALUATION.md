# Milestone 4 — Evaluation & Optimization

> **Final model at a glance:** repo-balanced TF-IDF, 16 repositories, test Macro-F1 **0.831**
> (accuracy 90.0%), unseen-repository Macro-F1 **0.704**; duplicate search within the same
> repository, Recall@5 **0.737** on 156 verified pairs. See A7–A8 for how we got there.

This report has two parts:

- **Part A, the final results** on the full corpus of 13,892 issues, collected with a token.
- **Part B, the pilot run** on 1,000 issues, which shaped the method. It is kept so the
  reasoning is visible.

---

# Part A — Final results on the full corpus

*Results: `outputs/m4_full/m4_results.json`, `outputs/m4_full/test_errors.csv`,
`outputs/m4_full/retrieval_pairs.json`. Data: `outputs/M2_EDA_REPORT_FULL.md`.*

## A1. What changed from the pilot

| | Pilot | Full corpus |
| --- | ---: | ---: |
| Issues | 1,000 | 13,892 |
| Labelled Bug / Feature / Docs issues | 523 | 8,551 |
| Test set | 80 | 1,570 (newest 15% of each repo) |
| Verified duplicate pairs (GitHub `duplicateOf`) | 0 | 173 (156 usable) |

**Decisions made on the data, using the M2 freeze rules:**

- **Question stays out of the classifier.** It now has 1,895 examples, but 52% come from VS
  Code (the limit is 40%) and only 4 repos use the label.
- **Splits are per-repository by time.** The label-targeted collection reached back to 2015
  for some labels, so one global time cut left only 27 Docs issues in test. Per repository,
  each repo's newest 15% is the test set, which gives 84 Docs test issues.
- **The 116 missing originals of duplicates were fetched.** A duplicate can only be found if
  its original is in the index, as it would be in a real deployment.

## A2. Classification: model choice and test results

Chosen by 5-fold cross-validation on train+validation; the test set played no part:

| Candidate | CV Macro-F1 |
| --- | ---: |
| Majority | 0.235 |
| **TF-IDF + LogReg (C=4)** | **0.897 ± 0.008** |
| Ensemble (TF-IDF + MiniLM) | 0.873 ± 0.004 |
| MiniLM + LogReg | 0.764 ± 0.008 |

**With 15× more data, the simple word-matching model overtook the ensemble.** MiniLM's frozen
general-purpose embeddings stop adding value once TF-IDF has enough examples of each repo's
vocabulary.

Results on the **test** set (1,570 newest issues):

| Model | Accuracy | **Macro-F1** | F1 Bug | F1 Feature | F1 Docs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority ("Feature") | 0.189 | 0.106 | 0.000 | 0.317 | 0.000 |
| **TF-IDF + LogReg (chosen)** | **0.868** | **0.799** | **0.917** | 0.746 | **0.733** |
| MiniLM + LogReg | 0.777 | 0.622 | 0.858 | 0.707 | 0.302 |
| Ensemble | 0.872 | 0.777 | 0.921 | **0.768** | 0.642 |

**M1 target of Macro-F1 ≥ 0.75: met (0.799).** Note that the majority class flipped to
"Feature": the training data is Feature-heavy (label-targeted collection), while the newest
issues are mostly bugs.

Confusion matrix (rows = true label, columns = predicted):

| | → Bug | → Feature | → Docs |
| --- | ---: | ---: | ---: |
| **Bug** (1,190) | 1,029 | **151** | 10 |
| **Feature** (296) | 17 | 278 | 1 |
| **Docs** (84) | 9 | 20 | 55 |

## A3. Error analysis on the full test set

- **151 Bugs predicted as Feature. 97 of them are VS Code.** VS Code's *training* issues are
  mostly feature requests (498 Feature vs 139 Bug, from label-targeted collection), so the model
  partly learned "VS Code writing style → Feature". M1 listed this risk: the repository's style
  standing in for the label. VS Code issues in the validation window are 95% correct, so it is
  also a recent change over time.
  - *Tried:* weighting training so every (repo, class) group counts equally. Validation
    Macro-F1 rose 0.698 → 0.731. This is a candidate for future work, not adopted, because it
    wasn't validated with the full selection protocol.
- **Feature written as a bug report** (e.g. "Go-to-definition does not work on `/// <reference
  lib>`" labelled Bug by TypeScript maintainers): the same grey zone as in the pilot.
- **Docs label on a code-behaviour description** (e.g. "`bool` and `MaybeUninit<bool>` are not
  actually ABI compatible", labelled docs in Rust): the label says where the fix goes, which
  the text can't reveal.

## A4. Confidence threshold (trade-off)

Validation-chosen rule: lowest threshold with ≥ 95% accuracy on validation → **0.90**.

| Threshold | Test: share auto-labelled | Test: accuracy on those |
| ---: | ---: | ---: |
| 0.45 (pilot value) | 99.4% | 87.1% |
| 0.70 | 84.7% | 90.9% |
| 0.85 | 66.7% | 94.8% |
| **0.90 (rule)** | **57.8%** | **96.0%** |

This is a product choice: a higher threshold means fewer, more reliable suggestions, and more
issues go to a human.

## A5. Robustness and unseen repositories (full data)

| Input change | Macro-F1 |
| --- | ---: |
| Original | 0.799 |
| Typos in 15% of words | 0.788 |
| Very long + log noise | 0.823 |
| First 20 words only | 0.549 |
| **Title only** | **0.446** |

**Title-only issues are the weak spot** (confirmed in the live demo: "Incorrect command in the
setup.md file"). That supports sending very short issues to manual review.

Repository held out of training: **mean Macro-F1 0.654**. It ranges from 0.887 (pandas) to
0.369 (Godot) and 0.020 (Flutter, which has almost no Feature or Docs labels to learn from).
Generalising to an unseen repository is the clearest limitation.

## A6. Duplicate detection on 156 real verified pairs

Each duplicate is searched against everything created before it (13,892 issues in total):

| Method | Recall@1 | **Recall@5** | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.385 | 0.603 | 0.699 | 0.482 |
| BM25 | 0.359 | 0.564 | 0.673 | 0.462 |
| MiniLM embeddings | 0.551 | 0.737 | 0.782 | 0.633 |
| **Hybrid (used)** | **0.590** | **0.756** | **0.814** | **0.663** |

- **M1 target of Recall@5 ≥ 0.80: not met (0.756).** Recall@10 is 0.814.
- **Meaning-based search clearly wins on real duplicates.** This is the opposite of the pilot's
  title-to-body proxy, because real duplicates are written by different people in different
  words.

**The yes/no duplicate decision is the weak part.** The top-1 similarity of a real duplicate
(median ≈ 0.46–0.50) is barely higher than that of an ordinary issue against its nearest
neighbour (median ≈ 0.41–0.45). Template-based and bot-filed issues look alike without being
duplicates. So no threshold gives good precision. Thresholds are therefore chosen to cap false
alarms, on the **older half** of the pairs:

| Threshold (chosen on older pairs) | Ordinary issues flagged (older → newer half) |
| --- | ---: |
| "Possible duplicate" ≥ 0.90 | 0.5% → 3.2% |
| "Check by hand" ≥ 0.74 | 1.9% → 7.4% |

**Product conclusion:** the useful output is the **ranked list of similar issues** (the
original is in the top 5 for 76% of duplicates) for a maintainer to check, not an automatic
yes/no. The app and the bot are worded that way.

## A7. Changes after the live demo (owner feedback)

The live demo on `zayn-ish/issue-triage-demo` raised two problems. Both led to measured changes.

**1. "Why compare with other people's repositories?"** In GitHub's own records, **164 of 173
verified duplicates (95%) point to an issue in the same repository.** Duplicate search is now
**same-repository only**: the app fetches the repository's recent issues live (one API call)
and compares against those plus anything collected for it. Re-scored on the 156 real pairs
(`outputs/m4_full/retrieval_pairs_same_repo.json`): hybrid **Recall@5 0.737, Recall@10 0.795,
MRR@10 0.652**. That is slightly below the all-repo search, because the 9 cross-repository
duplicates now count as misses.

**Thresholds set by the project owner to 0.70.** The measured cost:

| "Possible duplicate" at | Real duplicates caught (older / newer pairs) | Ordinary issues flagged |
| ---: | ---: | ---: |
| 0.90 | 3% / 3% | 0.5% / 3.2% |
| **0.70 (now)** | **4% / 3%** | **2.3% / 7.9%** |
| 0.55 ("check by hand" now) | 13% / 18% | 8.9% / 16.9% |

A threshold can't separate real duplicates well, which confirms A6. The ranked list of
similar issues is the main output.

**2. "It mislabels my login bugs."** The demo repo is a web app. All ten training repos are
developer tools, and the model called "login button does nothing" a Feature. Since the bot is
meant for *any* repository, we compared labelling strategies **on repositories the model never
saw** (train on 9 repos, test on the 10th; train+validation data only;
`outputs/m4_full/label_strategies.json`):

| Strategy | Unseen-repo Macro-F1 | … + the repo's own earlier labels |
| --- | ---: | ---: |
| TF-IDF, class-balanced (previous) | 0.653 | 0.668 |
| **TF-IDF, repo-balanced (adopted)** | 0.654 | **0.673** |
| Ensemble, repo-balanced | 0.654 | 0.669 |

- **Repo-balanced training:** each (repository, label) group weighs the same, so VS Code's
  498 feature requests can't teach "VS Code style → Feature".
- **Repo adaptation:** similar earlier issues in the *same* repository that maintainers
  already labelled vote on the label. The vote is similarity-weighted and gets up to 60%
  weight once 3 or more such issues are found. It helped in 8 of 9 held-out repositories
  (up to +0.07 on transformers).
- **Honest limit:** these gains are small. On the three login issues the model still leans
  Feature for 2 of 3. Its confidence stays below 0.70, so they go to manual review rather than
  getting a wrong label. **The real gap is domain coverage: no web-app repositories in
  training.** The fix is to add some (see the project plan's next steps).

## A8. Closing the domain gap: six web-application repositories (final model)

A7 showed the remaining problem was **missing knowledge, not settings**. We screened 12 popular
web-app repositories for clean, frequently used Bug/Feature/Docs labels (GitHub label lists +
Search API counts). Six were added: **supabase, nextcloud, Rocket.Chat, directus, outline,
excalidraw**. That's about 2,450 issues from ~80 API calls, checksum-verified, in
`data/raw/*_webapps/`. Grafana and appsmith were rejected (labels don't map cleanly), as were
mattermost (labels almost unused) and twenty (too few issues). The web apps contribute 1,086 Bug
and 1,023 Feature examples but only 14 Docs, so web-app *documentation* issues remain thin.

Per-repository time splits keep the original 10 repositories' train/test issues **identical**,
so before/after numbers on them are directly comparable.

**Did it help on a web app the model never saw?** (`outputs/m4_full/domain_coverage.json`;
each web app held out; train+validation only)

| Held-out web app | Dev tools only | Dev tools + other web apps |
| --- | ---: | ---: |
| supabase | 0.835 | **0.863** |
| directus | 0.683 | **0.777** |
| outline | 0.742 | **0.801** |
| excalidraw | 0.619 | **0.659** |
| nextcloud | **0.970** | 0.956 |
| Rocket.Chat | **0.841** | 0.697 |
| **Mean** | 0.782 | **0.792** |

The model improved on 4 of 6 web apps. It got worse on Rocket.Chat, whose `type: feature`
issues read differently from the other apps' feature requests. The overall gain is modest but
consistent with the cause identified in A7.

**Final model:** repo-balanced TF-IDF + logistic regression, trained on train+validation of all
16 repositories (`outputs/m4_webapps/m4_results.json`):

| Test set | Before web apps | **Final** |
| --- | ---: | ---: |
| Original 10 repositories (same 1,570 issues) | 0.838 Macro-F1 / 88.6% acc | **0.856 / 90.5%** |
| All 16 repositories (1,894 issues) | — | **0.831 / 90.0%** |
| Unseen-repository mean (leave-one-out) | 0.654 | **0.704** |
| At the 0.70 confidence gate | — | 87% of issues labelled, **94% correct** |

**Live demo issues:** the three login issues that were called "Feature" are now **Bug** (0.60,
0.73, 0.88). New demo issues #12 "Login button gets stuck on the loading screen" → **Bug 0.92**;
#13 "Unable to access dashboard after submitting login form" → **Bug 0.96**, plus **"might be
the same as #12" (0.56)**.

**What we'd do next:** add web apps with documentation labels; collect more repositories per
domain (mobile apps, data tools) so no single domain is missing; and track how often
maintainers keep or change the bot's suggestion. That last measure tells us whether the
suggestions are actually useful.

---

# Part B — Pilot run (1,000 issues): how the method was developed

*Run on 25 Nov 2026 on the M2 pilot data. Results: `outputs/m4/m4_results.json`; test errors:
`outputs/m4/test_errors.csv`.*

## How to reproduce

```bash
.venv/bin/python scripts/train_models.py     # CV model selection, refit on train+validation
.venv/bin/python scripts/evaluate_models.py  # everything in this report
```

## 1. The improvement we made: picking the model fairly

**Problem found:** in M3 we picked the model using the validation set. That set has only **6
Feature and 2 Documentation** issues, so a single lucky or unlucky guess swings Macro-F1 by
about 0.1. TF-IDF won on validation (0.768 vs 0.751). On the test set, the embedding model did
better instead. A choice that flips like that is not reliable.

**Fix:** choose with **5-fold cross-validation** on train+validation (443 issues). That means
training 5 times, each time scoring on a different fifth of the data, and averaging the scores.
We also added a new candidate: an **ensemble** that averages TF-IDF's and MiniLM's probabilities.
The test set played no part in the choice.

| Candidate | Validation Macro-F1 (M3 method) | **5-fold CV Macro-F1** |
| --- | ---: | ---: |
| Majority class | 0.315 | 0.300 ± 0.001 |
| TF-IDF + LogReg, C=1 / 4 / 16 | 0.744 / 0.768 / 0.768 | 0.811 / 0.806 / 0.805 (± ~0.04) |
| MiniLM + LogReg, C=1 / 4 / 16 | 0.734 / 0.751 / 0.751 | 0.685 / 0.719 / 0.700 (± ~0.07) |
| **Ensemble (TF-IDF + MiniLM)** | **0.788** | **0.848 ± 0.021** |

The ensemble wins on both measures and has the **smallest spread** (± 0.021), so it is also the
most stable. It is now the model the app uses, refitted on all 443 non-test issues.

## 2. Classification results on the TEST set (80 issues, never used for any choice)

All models were fitted on the same train+validation data.

| Model | Accuracy | **Macro-F1** | Weighted F1 | F1 Bug | F1 Feature | F1 Docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.738 | 0.283 | 0.626 | 0.849 | 0.000 | 0.000 |
| TF-IDF + LogReg | 0.825 | 0.647 | 0.790 | 0.894 | 0.476 | 0.571 |
| MiniLM + LogReg | 0.813 | 0.648 | 0.823 | 0.887 | 0.750 | 0.308 |
| **Ensemble (chosen)** | **0.900** | **0.764** | **0.893** | **0.936** | **0.857** | **0.500** |

**Target from M1: Macro-F1 ≥ 0.75 → met on the pilot (0.764).**

Confusion matrix of the chosen ensemble (rows = true label, columns = predicted):

| | → Bug | → Feature | → Docs |
| --- | ---: | ---: | ---: |
| **Bug** (59) | 58 | 0 | 1 |
| **Feature** (16) | 4 | 12 | 0 |
| **Docs** (5) | 3 | 0 | 2 |

**Why the ensemble helps:** the two models make *different* mistakes. TF-IDF matches exact words
and almost always says "Bug": it found only 5 of the 16 Features. MiniLM understands meaning and
finds Features (12 of 16), but confuses Bug and Docs. Averaging them keeps the strengths of both.

## 3. Error analysis: the 8 test mistakes, grouped

| Group | Count | Example | What's going on |
| --- | ---: | --- | --- |
| **A. Feature written as a complaint** | 4 | "Dictionary inspector page resets on 'Save As'"; "disk full during compilation very badly handled" | The text describes unwanted behaviour, like a bug report. The maintainers decided it was a missing capability. This is a genuine Bug/Feature grey zone, and the report predicted exactly this confusion. |
| **B. Docs label on bug-like text** | 2 | "Kubelet can report an old container instance … after a clock rollback" (labelled documentation) | The label says *where the fix goes* (the docs), not what the text describes. This is label noise from repository policy. Nothing in the text could reveal it. |
| **C. Docs vocabulary too rare** | 1 | "Bad wording in std::mem::take() docs" | Only 30 Docs examples in training. Confidence was **0.36**, below the 0.45 threshold, so in the app this goes to **manual review** instead of getting a wrong label. |
| **D. Misleading word** | 1 | "**ansible-doc** strips relative-link markup…" → predicted Docs | The tool's name contains "doc". It's a classic word-matching trap. |

**What would fix them:** more Feature and Docs data (from the label-targeted collection) for C
and D. For A, accept that it's ambiguous and rely on manual review. B can't be fixed from text
alone, and we report it as a limitation of maintainer labels as ground truth.

## 4. Confidence threshold: when to say "needs manual review"

Rule: pick the **lowest** confidence threshold whose accuracy on **validation** is ≥ 95%. That
keeps as many issues auto-labelled as possible while staying accurate. To be fair, the
validation confidences come from a model fitted on train only.

| Threshold | Validation: share auto-labelled | Validation: accuracy | **Test: share auto-labelled** | **Test: accuracy** |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 100% | 94.8% | 100% | 90.0% |
| **0.45 (picked)** | **97.4%** | **96.0%** | **97.5%** | **91.0%** |
| 0.60 | 94.8% | 97.3% | 81.3% | 92.3% |
| 0.80 | 67.5% | 98.1% | 41.3% | 93.9% |

On test, the system auto-labels **97.5%** of issues at **91%** accuracy and sends 2.5% to a human.
Test accuracy is lower than validation because test holds the newest issues, and the issue mix
changed over time (see §6). The threshold is in `config/model_config.json` and can be changed
without retraining.

## 5. Robustness tests (test set, chosen model)

| What we changed in the input | Macro-F1 | Same prediction as original | ms / issue |
| --- | ---: | ---: | ---: |
| Nothing (original) | 0.764 | 100% | 7 |
| Typos in 15% of words | 0.730 | 97.5% | 7 |
| Lowercase, no punctuation | 0.836 | 96.3% | 7 |
| Very long (text × 5 + 200 lines of log noise) | 0.764 | 100% | 26 |
| Title only (empty body) | 0.673 | 90.0% | 1 |
| Only the first 20 words | 0.594 | 83.8% | 1 |

| Edge case | What happened |
| --- | --- |
| Empty title and body | Rejected with a clear error message (no crash) |
| "help pls" | **needs_manual_review** (too short) |
| Emoji only "🔥🔥🔥 😭" | **needs_manual_review** |
| 50,000-character body | Handled in 116 ms; low confidence → **needs_manual_review** |
| Code block only (a Python traceback) | Bug, 0.90. Sensible |
| Spanish text | Bug, 0.73. Plausible, but the model was trained on English only (limitation) |

**Takeaways:** typos and long logs barely matter, thanks to the letter-chunk features and the
fixed-length embeddings. **Short issues are the weak spot.** Most of the signal is in the body,
so a title alone loses about 0.09 Macro-F1.

## 6. Repositories the model has never seen

We train on 9 repos and test on the 10th, then repeat for each repo:

| Held-out repo | Macro-F1 | | Held-out repo | Macro-F1 |
| --- | ---: | --- | --- | ---: |
| microsoft/vscode | 0.892 | | rust-lang/rust | 0.690 |
| huggingface/transformers | 0.882 | | kubernetes/kubernetes | 0.618 |
| microsoft/TypeScript | 0.739 | | ansible/ansible | 0.600 |
| pandas-dev/pandas | 0.726 | | scikit-learn/scikit-learn | 0.589 |
| godotengine/godot | 0.412 | | flutter/flutter | skipped (only 6 Bugs) |

**Mean: 0.683**, compared with 0.764 when the repo is in training. The ~8-point drop is within
the M1 rule of "no collapse > 10 points". Godot is hardest: its Feature requests are written like
bug reports (group A above).

**Change over time:** 9 of TF-IDF's 13 Feature misses in the first M4 run were VS Code issues
from the newest weeks. A time-based split exposes this kind of change. A random split would
have hidden it.

## 7. Duplicate detection: what we can measure without a token

Real duplicate pairs (GitHub's `duplicateOf` field) need the GraphQL API, which requires a
token, so the formal Recall@5 is **still pending**. Meanwhile we checked the threshold from the
other side: for the newest 30% of issues, how often does an ordinary issue look like a duplicate?
Each issue was searched only against issues created before it.

| Similarity threshold | Share of non-duplicate issues flagged |
| ---: | ---: |
| 0.70 (manual review) | 8.8% |
| **0.85 (possible duplicate)** | **6.8%** |
| 0.90 | 4.8% |

**Then we read every flagged case (≥ 0.85).** Most of them are **not** false alarms:

- **ansible:** someone filed the *same 13 issues twice* (#87784–#87796, then #87797–#87809).
  The maintainers closed the second batch as "not planned" rather than "duplicate".
- **flutter:** a monitoring bot files the same "linux-8 lost external connection" alert again
  and again.
- **vscode:** the two recent issues that really were closed as duplicates scored **1.0** and **0.915**.

**Lesson for the viva:** GitHub's "closed as duplicate" marker *under-counts* real duplicates. So
our "false alarm" rate is really an upper limit, and the true precision is higher. We keep 0.85 /
0.70 for now. We'll tune them properly on real `duplicateOf` pairs from the validation period
once the token is available.

## 8. Final model choice

| Component | Choice | Why |
| --- | --- | --- |
| Classifier | Ensemble: TF-IDF (word + letter-chunk) + MiniLM, averaged | Best CV Macro-F1 (0.848), most stable, best on test (0.764) |
| Confidence threshold | 0.45, plus at least 6 words | ≥ 95% validation accuracy at 97% coverage |
| Duplicate search | Hybrid: 0.6 × MiniLM + 0.4 × TF-IDF, past issues only | Keeps exact error-message matching *and* catches reworded duplicates |
| Duplicate thresholds | 0.85 possible duplicate / 0.70 manual review | Interim, see §7 |

## 9. M4 checklist

- [x] Evaluation report with all metrics (this file + `m4_results.json`)
- [x] Model comparison tables and confusion matrix
- [x] Threshold analysis: classification confidence (done); duplicate (interim, needs pairs)
- [x] Error analysis grouped into categories
- [x] Improvement iteration documented (CV selection + ensemble)
- [x] Robustness results, unseen-repository test, final model choice
- [ ] Duplicate Recall@5 on real `duplicateOf` pairs, **waiting for the GitHub token**
