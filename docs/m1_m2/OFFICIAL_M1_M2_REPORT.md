# GitHub Issue Auto-Labeler & Duplicate Finder

## Official Milestone 1 and Milestone 2 report

**Prepared:** 29 October 2026  
**Course source:** *NLP Research Project Guideline* supplied by the student  
**Research area:** Natural-language processing for software engineering  
**Official status:** M1 complete; M2 complete for one feasible baseline; M3–M6 not claimed complete

---

## Executive verdict

The project is now aligned to the milestone definitions in the supplied university guideline.
The earlier project plan used different names: it called problem definition “M1” and data
preparation “M2.” Those are valuable engineering foundations, but they are **not** the official M1
and M2 in the course PDF. The corrected structure is:

| Course milestone | Official requirement | Evidence completed here |
| --- | --- | --- |
| M1 | Literature survey, five-paper comparison, one anchor paper, reproducibility audit | Five papers compared; NLBSE’24 selected and justified; all ten checklist questions answered |
| M2 | Reproduce a feasible baseline and report the reproduction gap | NLBSE’24 per-repository fastText baseline rerun on the public 3,000-issue benchmark; full configuration, environment, predictions, metrics and gap saved |
| M3 | State one original hypothesis before experimentation | Not yet claimed; a candidate direction is proposed only |
| M4 | One-variable experiment or ablation | Not started |
| M5 | Error analysis on 5–10 failure cases | Not started |
| M6 | IEEE-style paper and final presentation | Not started |

The official M2 result is credible: the published artifact’s aggregate F1 is **0.71839** and our
independent rerun obtained **0.71595**, an absolute gap of **−0.00244** (about −0.24 percentage
points). The run exactly matched the checked-in result for two of five repositories and differed
slightly for three, which is a successful reproduction with a small, documented gap—not an exact
bit-for-bit replication claim.

---

## 1. The whole project, reiterated

### 1.1 Problem

Open-source maintainers repeatedly have to (a) decide what kind of report a newly opened GitHub
issue is and (b) search older issues to determine whether the same problem already exists. Manual
triage is slow, inconsistent across repositories, and difficult to scale.

### 1.2 Proposed system

The system accepts the repository, issue title and issue body available when the issue is created.
It provides two forms of decision support:

1. **Issue-type classification:** predict one standardized class such as Bug, Feature or Question,
   with confidence and a manual-review path.
2. **Duplicate retrieval:** search only issues that existed before the query, return the top-K
   candidates, and recommend duplicate, not duplicate or manual review.

The classifier and retriever share text preparation but have different targets and metrics. Textual
similarity is never treated as duplicate ground truth. A verified duplicate pair requires structured
GitHub evidence such as `duplicateOf`, or a separately audited maintainer reference.

### 1.3 Intended workflow

```text
New issue (repository + title + body)
                |
        conservative cleaning
         /                   \
issue-type classifier      text embedding / lexical index
         |                   |
label + confidence         earlier issues only
                             |
                         top-K candidates
                             |
                   duplicate / no / review
```

### 1.4 Research question

**Can a reproducible issue-type classification baseline be extended into a cross-repository GitHub
triage system that preserves minority-class performance and retrieves verified duplicates without
using future information?**

### 1.5 What is already engineered

The project repository already contains an auditable GitHub collection and preparation pipeline, a
repository-aware label taxonomy, checksum manifests, leakage-sensitive tests, a 1,000-issue live
pilot, a structured duplicate-pair collector, and the newly completed baseline reproduction.

The live pilot contains 100 recent issues from each of ten public repositories. Of 524
classification-eligible rows, the standardized labels are 421 Bug, 67 Feature, 35 Documentation and
1 Question. This pilot is useful for pipeline and taxonomy validation, but it is not representative
training data because the Search API sample is recent-only. Eighteen rows have GitHub’s structured
duplicate state; a canonical pair still requires `duplicateOf` or manual audit.

### 1.6 Clear project boundary

- **Completed for M1/M2:** literature evidence, anchor selection, reproducibility audit, baseline
  reproduction, authentic-data pipeline and pilot data audit.
- **Not yet claimed:** a trained final product model, a completed duplicate detector, a novel
  hypothesis test, an ablation, final error analysis or deployment.
- **Human role:** the system suggests; maintainers retain the final label and duplicate decision.

---

## 2. Official Milestone 1 — literature survey

### 2.1 Search scope and paper-selection logic

The survey intentionally covers both halves of the project and the bridge between them:

- GitHub issue-type classification at operational, neural and benchmark scales;
- classical duplicate-bug retrieval using both text and metadata; and
- efficient semantic embeddings suitable for top-K retrieval.

Primary paper pages, DOI records and official/public replication repositories were cross-checked.
“Publicly visible” is distinguished from “fully reproducible”: a paper may have a public PDF or
repository and still omit a license, transitive dependency versions, exact hardware or training
time.

### 2.2 Five selected papers — method and evaluation comparison

| ID | Paper and research problem | Method / model | Dataset | Metrics |
| --- | --- | --- | --- | --- |
| P1 | **Kallis et al. (2024), The NLBSE’24 Tool Competition.** Benchmark automated issue-report classification on small repositories. | Per-repository three-class benchmark with SetFit, RoBERTa and fastText baselines; multiple submitted models. | 3,000 balanced GitHub issues from React, TensorFlow, VS Code, Bitcoin and OpenCV; 1,500 train and 1,500 test; Bug, Feature, Question. | Per-class precision/recall/F1; repository cross-class F1; arithmetic mean over five repositories. |
| P2 | **Kallis et al. (2021), Predicting Issue Types on GitHub.** Automatically categorize incoming issues in a GitHub workflow. | Ticket Tagger using title and description with a fastText-based classifier and GitHub App integration. | Approximately 30,000 GitHub issues. | Classification effectiveness including precision, recall and F1 as reported in the paper. |
| P3 | **Siddiq & Santos (2022), BERT-Based GitHub Issue Report Classification.** Improve three-class issue typing over fastText. | Fine-tuned `bert-base-uncased` sequence classifier using concatenated, whitespace-normalized title and body. | 803,417 issues: 614,464 train, 108,435 validation and 80,518 test; Bug, Enhancement, Question. | Per-class P/R/F1 and global score across epochs. |
| P4 | **Sun et al. (2011), Towards More Accurate Retrieval of Duplicate Bug Reports.** Rank the original report for a duplicate query. | REP: BM25F-style textual similarity plus categorical fields, optimized using two-round SGD. | Mozilla, Eclipse and OpenOffice; also a large Eclipse set of 209,058 reports. | Recall@K and mean average precision (MAP). |
| P5 | **Reimers & Gurevych (2019), Sentence-BERT.** Make transformer sentence similarity practical for retrieval. | Siamese/triplet BERT or RoBERTa embeddings compared by cosine similarity. | Standard semantic textual similarity and transfer-learning datasets; not GitHub-specific. | STS correlation and transfer-task metrics; retrieval computation time. |

### 2.3 Five selected papers — findings, artifacts and limitations

| ID | Main finding | Code/data availability | Limitation relevant to our project | Possible extension in this project |
| --- | --- | --- | --- | --- |
| P1 | The reported fastText baseline reaches aggregate F1 **0.718**; submitted systems reach higher scores, and the paper notes that the Question class remains difficult for most approaches. | Public benchmark CSVs, notebooks and result JSONs are in the NLBSE’24 GitHub repository. The checked commit contains an empty `LICENSE` file, so visibility is clear but reuse terms are not. | Only five repositories and three labels; labels reflect repository policy; the artifact does not specify exact original hardware or training time. | Reproduce fastText, then test temporal and repository-held-out generalization on a separately collected current corpus. |
| P2 | A lightweight classifier can be embedded into GitHub triage and was evaluated on about 30,000 issues. | Paper and a public GitHub App exist; dataset material is publicly visible, but the exact end-to-end environment is less self-contained than P1. | Cross-project label semantics and class policy can be confounded; an operational app alone is not a reproducibility package. | Compare a simple fastText/TF-IDF baseline with repository-aware calibration and current labels. |
| P3 | Average F1 **0.8571**, peak global score **0.8586**, versus the reported fastText baseline **0.8162**; Question remains the weakest class. | Public source repository and paper; the paper specifies base model, preprocessing, splits and GPU-driven batch-size constraint. | More than 800k records and end-to-end BERT fine-tuning are heavier than needed for a first reproducible baseline; random splitting may overstate future-use performance. | Use as a stronger later classifier, but evaluate chronologically and per repository against the light baseline. |
| P4 | REP reports 10–27% relative Recall@K and 17–23% relative MAP improvement over the authors’ earlier model; on 209,058 Eclipse reports it reports MAP 47%. | Paper and named datasets are available; modern, exact code/environment reproduction is less direct than P1. | Old tracker domains and metadata schemas; repository-specific categorical fields may not transfer to GitHub. | Build BM25/BM25F-style title+body retrieval, then add semantic embeddings and compare under the same temporal candidates. |
| P5 | SBERT reduces a 10,000-sentence similarity search described as about 65 hours with pairwise BERT to about 5 seconds while retaining strong semantic accuracy. | Official paper, open-source sentence-transformers implementation and public pretrained models. | General-domain semantic similarity is not duplicate-issue equivalence; high cosine similarity does not prove a duplicate. | Use SBERT only as a retriever or reranker, trained/evaluated on verified duplicate pairs with Recall@K and MRR. |

### 2.4 Synthesis and research gap

The literature establishes that issue-type classification is feasible and that transformer models
often outperform fastText, but it also exposes three gaps that matter here:

1. **Repository policy is entangled with language.** “Question,” “enhancement,” and documentation
   practices vary by project; pooled random splits can reward repository recognition.
2. **Reproducibility is more than a public notebook.** Exact transitive dependencies, hardware,
   duration and licensing may still be missing.
3. **Similarity and duplication are different.** SBERT makes semantic search efficient, while REP
   shows the value of fields and supervised ranking, but verified relationships and temporal
   candidate restrictions are necessary for a defensible GitHub evaluation.

Our project therefore separates the anchor reproduction from the later independent contribution.
The NLBSE fastText run answers, “Can we reproduce the benchmark?” The later project will ask a new,
pre-registered question about cross-repository/temporal behavior or hybrid duplicate retrieval.

### 2.5 Anchor paper selected

**Anchor:** P1, *The NLBSE’24 Tool Competition*.

**Why it is the best M2 anchor:**

- It is directly about the classification half of this project.
- The exact 3,000-row dataset, fixed train/test files, notebooks and result JSONs are public.
- The dataset is balanced and the aggregation rule is explicit, so metric reconstruction is
  auditable.
- fastText is small enough to reproduce on CPU, satisfying the guideline’s feasibility criterion.
- The benchmark leaves meaningful extensions: stronger encoders, chronological testing,
  held-out-repository testing, current GitHub data and a connected duplicate-retrieval task.

The **chosen baseline within the anchor** is its per-repository supervised fastText model, not the
best competition submission. Reproduction is supposed to establish a clear reference point; it is
not supposed to maximize the result before the project’s own hypothesis is stated.

### 2.6 Official reproducibility checklist

| Guideline question | Finding | Verdict |
| --- | --- | --- |
| Is the dataset public? | Yes. Train/test CSVs are in the public NLBSE’24 repository. | Yes |
| Is dataset size documented? | Yes: 3,000 total, 1,500 train and 1,500 test; 500 examples of each class in each split; five repositories with 300 rows per split. | Yes |
| Is code public? | Yes: dataset, SetFit, RoBERTa and fastText notebooks plus output JSONs. | Yes |
| Is a public pretrained model available? | Not required for the selected fastText baseline; it trains from the provided text. The stronger SetFit/RoBERTa alternatives use public bases. | N/A for chosen baseline |
| Are hardware requirements known? | Exact original hardware is not stated. The selected baseline is CPU-feasible; our run used an 8-logical-CPU, 8 GiB Apple-arm64 machine. | Partial |
| Is training time known? | The anchor does not report it. Our five models trained in about 1.22 s total; the full script took 2.74 s. | Missing in anchor; measured here |
| Are dependencies listed? | Top-level versions are pinned, but transitive versions are not. Reproduction required compatible `pyarrow`, `scipy` and `huggingface-hub` pins. | Partial |
| Is the evaluation metric defined? | Yes: class P/R/F1, cross-class average, then arithmetic mean over five repository scores. | Yes |
| Is a feasible baseline available for Colab/Kaggle/free GPU? | Yes. The selected baseline runs on CPU; no GPU is required. | Yes |
| Are reported results sufficiently reproducible? | Yes for fastText: our F1 differs by only −0.00244, with exact repository-level matches for Bitcoin and VS Code. | Yes, with documented gap |

### 2.7 M1 completion decision

M1 is **complete against the supplied guideline**: the research area is defined; five relevant
papers are compared on all required fields; one anchor is selected with explicit justification; and
every reproducibility question is answered. The literature survey does not claim that every paper
was rerun—only the anchor baseline belongs to M2.

---

## 3. Official Milestone 2 — baseline reproduction

### 3.1 Reproduction target

**Target:** the per-repository supervised fastText baseline in the NLBSE’24 issue-report
classification artifact, specifically `4-Template-fastText.ipynb` and its checked-in result JSONs.

**Frozen anchor commit:** `2927bc67eb42db8affd16eaf3e5a6d74f3063961`  
**Anchor repository:** <https://github.com/nlbse2024/issue-report-classification>

### 3.2 Dataset and integrity

| Item | Value |
| --- | --- |
| Train rows | 1,500 |
| Test rows | 1,500 |
| Labels | `bug`, `feature`, `question` |
| Train distribution | 500 / 500 / 500 |
| Test distribution | 500 / 500 / 500 |
| Repositories | React, TensorFlow, VS Code, Bitcoin, OpenCV |
| Train SHA-256 | `18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5` |
| Test SHA-256 | `4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85` |

The fixed test set was used as supplied. There was no resampling, rebalancing, tuning on test
scores, or substitution with the separate 1,000-issue project pilot.

### 3.3 Preprocessing

The run follows the anchor notebook:

1. concatenate title and body with one space;
2. remove the reserved fastText `__label__` sequence from text;
3. collapse consecutive whitespace using gensim;
4. shuffle training and test data independently with seed 42; and
5. train one classifier per repository.

No future comments, closure information or new GitHub pilot fields are used.

### 3.4 Model and hyperparameters

| Setting | Value |
| --- | ---: |
| Model | fastText supervised classifier |
| Models trained | 5, one per repository |
| Epochs | 100 |
| Learning rate | 0.1 |
| Embedding dimension | 100 |
| Context window | 5 |
| Word n-grams | 1 |
| Minimum count | 1 |
| Bucket | 0 |
| fastText threads observed | 7 |
| Random seed | 42 |

These are the actual arguments observed from the trained models, not guessed defaults.

### 3.5 Environment and compute

| Item | Reproduction value |
| --- | --- |
| Python | 3.11.13 |
| pandas | 2.1.4 |
| scikit-learn | 1.3.1 |
| fasttext | 0.9.2 |
| gensim | 4.3.2 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| datasets | 2.16.1 |
| PyArrow | 14.0.2 |
| huggingface-hub | 0.20.2 |
| Hardware | Apple arm64, 8 logical CPUs, 8 GiB RAM; no GPU |
| Five training durations | 0.23–0.27 s per repository |
| Total script duration | 2.7372 s |

The anchor pins direct dependencies but not all transitive packages. Current PyArrow and SciPy
releases were incompatible with the 2024 stack, so compatible versions were explicitly pinned. This
is reported as a reproduction finding rather than hidden.

### 3.6 Evaluation protocol

For every repository, precision, recall and F1 are calculated for Bug, Feature and Question. The
repository “average” is scikit-learn’s weighted average; because each repository test set is
balanced, it is equivalent to the macro average. Cross-repository results are the arithmetic mean
of the five repository values.

### 3.7 Original result versus reproduced result

| Scope | Anchor F1 | Reproduced F1 | Gap (reproduced − anchor) |
| --- | ---: | ---: | ---: |
| React | 0.78757 | 0.78416 | −0.00341 |
| TensorFlow | 0.70627 | 0.70035 | −0.00592 |
| VS Code | 0.72752 | 0.72752 | 0.00000 |
| Bitcoin | 0.66184 | 0.66184 | 0.00000 |
| OpenCV | 0.70875 | 0.70588 | −0.00288 |
| **Cross-repository aggregate** | **0.71839** | **0.71595** | **−0.00244** |

Reproduced aggregate metrics by class:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Bug | 0.76112 | 0.70200 | 0.72837 |
| Feature | 0.71407 | 0.76600 | 0.73704 |
| Question | 0.68319 | 0.68200 | 0.68243 |
| **Average** | **0.71946** | **0.71667** | **0.71595** |

The Question class is again the weakest by F1, consistent with the anchor paper’s discussion. The
large live-pilot scarcity of Question issues also warns that a balanced benchmark does not describe
operational prevalence.

### 3.8 Reproduction-gap analysis

The aggregate gap is small and is not treated as a failure. Plausible causes, ordered by evidence,
are:

1. **Native fastText nondeterminism:** a seed is supplied, but multithreaded native training does
   not guarantee bit-identical results across machines. The model used seven threads.
2. **Architecture and compiler differences:** the checked result may have been generated on x86-64
   Linux, whereas this reproduction used Apple arm64 macOS.
3. **Unpinned transitive dependencies/build toolchain:** the anchor omits exact SciPy, PyArrow,
   Hugging Face Hub, compiler and C++ runtime versions.
4. **No dataset or metric mismatch found:** the fixed CSV hashes, preprocessing, seed, labels,
   split and aggregation were checked; two repositories match exactly.

The evidence therefore supports “successfully reproduced with a small environment-sensitive gap.”
It does not support claiming that our score is better, or that the original result is wrong.

### 3.9 M2 completion decision

M2 is **complete for the selected feasible baseline**. The dataset, preprocessing, model,
hyperparameters, training setup, metrics, anchor result, reproduced result and gap are all
documented. Machine-readable predictions, confusion matrices, exact environment freeze, result
JSONs and a run manifest are retained under `outputs/baseline_reproduction/`.

---

## 4. Authentic data for the later independent project

The M2 benchmark is not silently mixed with the project’s future research dataset.

### 4.1 Source-of-truth hierarchy

1. **GitHub REST Issues API:** current title, body, labels, Issue Type, timestamps and structured
   state reason; pull requests returned by the endpoint are filtered out.
2. **GitHub GraphQL API:** canonical duplicate relationship via `Issue.duplicateOf`.
3. **GitHub timeline/comments:** audit fallback only when an explicit maintainer relationship is
   preserved with evidence.
4. **Published datasets:** benchmark and sanity-check sources, not replacements for project
   provenance.

### 4.2 Leakage rules

- Model inputs use only information available at issue creation time.
- Comments, future edits, final closure decisions and canonical duplicate links are excluded from
  input features.
- Classification evaluation will use chronological and repository-held-out checks, not only a
  random split.
- Duplicate retrieval candidates must predate the query issue.
- Linked/near-identical issue groups must not leak across partitions.

### 4.3 Next hypothesis is not yet registered

The guideline requires one hypothesis to be written before M3 experiments, with an independent
variable, direction, approximate effect and mechanism. A promising direction is a hybrid
lexical-plus-semantic duplicate retriever, or repository-held-out classification calibration. The
exact hypothesis and expected effect size should be frozen only after duplicate-pair sufficiency
and final repository coverage are audited. This report deliberately does not backfill a hypothesis
after looking at future results.

---

## 5. How to navigate and demonstrate the project

### 5.1 Folder map

| Location | What to show |
| --- | --- |
| `outputs/OFFICIAL_M1_M2_REPORT.pdf` | Main presentation document; official M1/M2 evidence |
| `outputs/baseline_reproduction/run_manifest.json` | Exact M2 data hashes, environment, model and timing |
| `outputs/baseline_reproduction/fasttext_metric_gaps.json` | Machine-readable original/reproduced differences |
| `outputs/baseline_reproduction/predictions.csv` | Row-level M2 predictions for audit |
| `scripts/reproduce_nlbse_fasttext.py` | Reproduction code |
| `config/repositories.json` | Candidate repositories for the independent dataset |
| `config/label_taxonomy.json` | Auditable mapping of repository labels to standard classes |
| `data/raw/pilot/collection_manifest.json` | Authentic GitHub queries, timestamps, row counts and hashes |
| `data/processed/data_quality.json` | Pilot integrity and class-distribution checks |
| `outputs/M2_EDA_REPORT.md` | Engineering pilot EDA; groundwork, not the official course M2 |

### 5.2 Eight-minute presentation route

1. **Problem (45 s):** maintainers categorize issues and search for duplicates; explain the two
   connected tasks.
2. **Milestone correction (30 s):** state that the university PDF defines M1 as literature survey
   and M2 as baseline reproduction; engineering collection work is supporting groundwork.
3. **M1 comparison (2 min):** show the two five-paper tables; explain classification, duplicate
   retrieval and SBERT; point out the reproducibility gap in the literature.
4. **Anchor decision (45 s):** explain why NLBSE’24 and why the CPU-feasible fastText baseline.
5. **M2 protocol (1 min):** show the frozen commit, data hashes, 3,000 rows, five repositories,
   preprocessing, 100 epochs and seed 42.
6. **M2 result (1 min):** show 0.71839 versus 0.71595 and the −0.00244 gap; do not say “same
   accuracy.” Say “small environment-sensitive reproduction gap.”
7. **Independent project data (1 min):** open the pilot manifest, taxonomy and EDA; explain why
   similarity is not duplicate truth and why the recent-only pilot is not final training data.
8. **Boundary/next step (1 min):** M1/M2 complete, M3 hypothesis not yet claimed; next freeze the
   full corpus and pre-register one testable hypothesis.

### 5.3 Whether live repository browsing is necessary

It is **not necessary** for proving M1 or M2. The literature artifacts, frozen benchmark files,
hashes, saved predictions and report are enough. If internet access is reliable, one GitHub issue
page can be used only as a short motivation example. Do not depend on a live API call during the
presentation; rate limits or repository changes would make the demonstration fragile.

### 5.4 Safe live-demo sequence

```bash
# 1. Re-run the official M2 baseline using the already prepared environment
work/.venv-nlbse/bin/python scripts/reproduce_nlbse_fasttext.py \
  --anchor-dir work/nlbse2024-anchor \
  --output outputs/baseline_reproduction \
  --work-dir work/nlbse_fasttext_models

# 2. Verify project preparation tests
python3 -m unittest discover -s tests -v

# 3. Verify the saved pilot snapshot against its manifest
python3 scripts/verify_manifest.py data/raw/pilot/collection_manifest.json
```

For the classroom, prefer opening saved results rather than rerunning setup or installing packages.

---

## 6. Questions you should be ready to answer

**Why fastText when stronger models exist?**  
M2 asks for a feasible baseline reproduction. fastText is the anchor’s explicit low-cost baseline,
has a checked result, runs on CPU in seconds and creates a clean reference for M3.

**Is the −0.00244 gap a failure?**  
No. The course guideline treats a gap as evidence to analyze. Data hashes, split, preprocessing and
metrics match; two repositories reproduce exactly. Native multithreading, architecture and omitted
transitive/build versions plausibly explain the small differences.

**Why not use the 1,000 live issues for M2?**  
That snapshot is recent-only, severely imbalanced and has a different taxonomy. Substituting it
would no longer reproduce the anchor. It supports the independent contribution after M2.

**Why are labels authentic if maintainers add them later?**  
Labels are retrospective supervision, not prediction-time input. The model receives only title/body
and creation-time fields. The limitation and potential label drift are documented.

**How do you know two issues are duplicates?**  
Not from cosine similarity or a `duplicate` label alone. We require GitHub’s canonical
`duplicateOf` relationship or a separately audited explicit maintainer link.

**Why is Question scarce in the live pilot but balanced in the benchmark?**  
The benchmark was deliberately balanced. Many current repositories redirect questions elsewhere or
use different labels. Operational prevalence and benchmark balance answer different questions.

**What is novel?**  
M1/M2 establish evidence and reproducibility, not novelty. The independent contribution will be one
pre-registered, one-variable experiment—likely cross-repository/temporal classification or hybrid
duplicate retrieval—after data sufficiency is verified.

**Are you claiming the final application is finished?**  
No. Official M1/M2 are complete. The final classifier, duplicate retriever, hypothesis test,
ablation, failure analysis and IEEE paper belong to M3–M6.

---

## 7. Evidence index and reproducibility commands

The reproduction program writes seven auditable outputs:

- `run_manifest.json`: commit, hashes, data counts, preprocessing, packages, hardware, timing and
  headline result;
- `environment_freeze.txt`: exact installed Python packages;
- `fasttext_anchor_results.json`: checked-in artifact results normalized for comparison;
- `fasttext_reproduced_results.json`: our complete metrics;
- `fasttext_metric_gaps.json`: cell-by-cell result differences;
- `predictions.csv`: all 1,500 predictions; and
- `confusion_matrices.csv`: repository/class confusion counts.

The live GitHub preparation pipeline and the benchmark reproduction are intentionally separate.
This prevents a polished engineering pilot from being mistaken for the official baseline and
prevents anchor test data from becoming the project’s later training corpus.

---

## 8. References

1. R. Kallis, G. Colavito, A. Al-Kaswan, L. Pascarella, O. Chaparro, and P. Rani, “The NLBSE’24
   Tool Competition,” ICSE/NLBSE, 2024. Paper: <https://ojcchar.github.io/files/30-nlbse24-tools.pdf>.
   Replication artifact: <https://github.com/nlbse2024/issue-report-classification>.
2. R. Kallis, A. Di Sorbo, G. Canfora, and S. Panichella, “Predicting Issue Types on GitHub,”
   *Science of Computer Programming*, vol. 205, 102598, 2021.
   DOI: <https://doi.org/10.1016/j.scico.2020.102598>.
3. M. L. Siddiq and J. C. S. Santos, “BERT-Based GitHub Issue Report Classification,” NLBSE, 2022.
   DOI: <https://doi.org/10.1145/3528588.3528660>. Code:
   <https://github.com/s2e-lab/BERT-Based-GitHub-Issue-Classification>.
4. C. Sun, D. Lo, S.-C. Khoo, and J. Jiang, “Towards More Accurate Retrieval of Duplicate Bug
   Reports,” ASE, 2011. DOI: <https://doi.org/10.1109/ASE.2011.6100061>.
5. N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,”
   EMNLP-IJCNLP, 2019. DOI: <https://doi.org/10.18653/v1/D19-1410>.
6. *NLP Research Project Guideline*, student-supplied course document, 3 pages, dated 5 October
   2026 in PDF metadata.

---

## Final sign-off

The project is now officially geared to the supplied PDF. M1 is supported by a five-paper,
cross-checked literature survey, explicit anchor choice and full reproducibility checklist. M2 is
supported by an actual CPU baseline rerun with frozen data, code, environment, metrics and gap—not
only a plan. The correct next move is to freeze the independent dataset and write the M3 hypothesis
before running its experiment.
