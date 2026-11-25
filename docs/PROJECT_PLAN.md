# GitHub Issue Auto-Labeler & Duplicate Finder — Project Plan (M1 → M6)

*Written in plain language so anyone (student, reviewer, interviewer) can follow it.*
*Last updated: 25 November 2026*

---

## 1. The project in 30 seconds

Big open-source projects (VS Code, pandas, Kubernetes…) get hundreds of new GitHub issues every
week. For each one, a maintainer has to do two boring jobs:

1. **Decide what kind of issue it is** — Bug? Feature request? Documentation problem?
2. **Check whether someone already reported it** — search old issues for the same problem.

Our system does both automatically the moment an issue is opened:

```
New issue (title + body)
        │
        ▼
  Clean the text  (keep code & error messages, remove links/@mentions noise)
        │
        ├──► Classifier ──────────► "Bug" (87% confident)
        │
        └──► Similarity search ───► Top-5 older issues that look the same
                                    + "Possible duplicate of #1234" / "Not a duplicate" / "Needs a human"
```

It only **suggests**. A human maintainer always makes the final call.

---

## 2. Where we are right now (M1 and M2 check)

I compared the project plan with what is actually in the code and the saved files.

### Milestone 1 — Problem Definition: ✅ Done

| What the plan asks for | Is it there? | Where |
| --- | --- | --- |
| Problem brief, task definitions | ✅ | `M1_M2_PROJECT_HANDBOOK.md` §1–2 |
| Target users and business value | ✅ | Handbook §2.2 |
| How we pick repositories + the list | ✅ | `config/repositories.json` (10 repos), Handbook §2.5–2.6 |
| Success metrics and targets | ✅ | Handbook §2.7 (Macro-F1 ≥ 0.75, Recall@5 ≥ 0.80, p95 ≤ 2 s) |
| Feasibility and risk list | ✅ | Handbook §2.9 (13 risks with fixes) |
| First system architecture | ✅ | Handbook §1 diagram |

### Milestone 2 — Data Preparation: ✅ Done as a pilot (3 small items left)

| What the plan asks for | Is it there? | Evidence |
| --- | --- | --- |
| Data card | ✅ | Handbook §3.3 |
| Repository list + collection pipeline | ✅ | `scripts/collect_github.py` (3 modes, pagination, rate-limit logging) |
| Raw and cleaned datasets | ✅ | 1,000 issues, 10 repos; checksums verified again today (10/10 OK) |
| Label taxonomy + why we mapped labels | ✅ | `config/label_taxonomy.json` + Handbook §3.5 |
| EDA and statistics | ✅ | `M2_EDA_REPORT.md` |
| Train / validation / test split | ✅ | Time-based 70/15/15 (366 / 78 / 80) |
| Leakage and privacy rules | ✅ | Handbook §3.8 |
| Tests | ✅ | 7 out of 7 tests pass |

**What is left from M2. We do these as step 0 of M3:**

1. **The pilot is too small to train on.** It has 524 usable issues, and 80% of them are Bugs. We
   must run the bigger collection with a GitHub token (the script already exists).
2. **Freeze the classes.** The pilot found only **1** "Question" issue, so we go with
   **3 classes: Bug, Feature, Documentation**. That is the handbook's own recommendation.
3. **Collect real duplicate pairs** with the GraphQL `duplicateOf` script (already written).

**A housekeeping problem we must fix:** this Git repo only has the *code*. The data, the EDA
report, the handbook and the NLBSE baseline results are stored in another folder
(`~/Documents/Codex/2026-10-28/cr/`), and copies sit in `~/Desktop/nlp/`. Before M3 we copy
them into this repo so that one folder holds the whole project.

> **Note: there are two milestone schemes.** The plan you pasted follows the **Industry Track**
> (M1 Problem → M2 Data → M3 Models → M4 Evaluation → M5 Deployment → M6 Delivery). Your saved
> `OFFICIAL_M1_M2_REPORT.md` follows a **Research Track** (M1 Literature survey → M2 Reproduce a
> paper's baseline. That reproduction is done: F1 0.716 against the paper's 0.718). **The work is
> complete under both schemes.** The rest of this plan follows the Industry Track, because it ends
> in a working product.

---

## 3. What problem we solve, and how we prove it

| Who | Their pain today | What our tool gives them |
| --- | --- | --- |
| Maintainer | Reads every issue just to label it | A suggested label + how confident the model is |
| Maintainer | Searches old issues by hand for duplicates | The 5 most similar old issues, ranked |
| Person reporting the issue | Doesn't know it was already reported | Sees "this might be the same as #1234" |
| Project lead | Can't measure triage quality | A dashboard: how many issues, how accurate, how fast |

**How we prove it works (with numbers, not just a demo):**

- Classification → **Macro-F1** on a test set of *future* issues the model never saw.
  We use Macro-F1, not plain accuracy. With 80% Bugs, a model that always answers "Bug" already
  gets 80% accuracy and is useless.
- Duplicates → **Recall@5**: "for a real duplicate, is the original issue in our top 5?"
- Speed → time from "issue opened" to "suggestion ready". The target is under 2 seconds.

---

## 4. How we show it to people (the demo design)

Your idea was right: build a FastAPI app where you paste a GitHub link. I've added two more
ways to use it. All three run the **same prediction code**, so this is one app, not three projects.

### Way 1 — "Paste an issue link" (the easiest demo, works on ANY public repo)

```
You paste:  https://github.com/pandas-dev/pandas/issues/60123
App does:   1 GitHub API call to fetch that issue
            → clean text → classify → search similar older issues
Shows:      every step on screen: cleaned text → label probabilities → top-5 similar → decision
```

Interviewers like this one: they can pick *any* issue themselves and watch it work.
We also compare our predicted label with the label the maintainers actually gave it.

### Way 2 — "Paste a repo link" (classify a whole repo's recent issues)

```
You paste:  https://github.com/godotengine/godot
App does:   fetch the latest ~100 open issues (1 API call per 100 issues)
            → build/refresh that repo's search index (saved to disk, so it's reused)
Shows:      a table: issue | predicted label | real label | possible duplicate
```

This shows the tool working at scale, plus a live "how often were we right" number.

### Way 3 — Live mode: a real bot on a real repo (the "wow" moment)

```
1. We create our own demo repo, e.g.  github.com/<you>/issue-triage-demo
2. We add a webhook: "when an issue is opened, send it to our app"
3. In the interview, someone opens an issue there: "App crashes while uploading large images"
4. Within a few seconds a comment appears on that issue:

   🤖 Suggested label: Bug (confidence 0.87)
   Possibly related issues:
     • #45  "Crash when uploading big PNG files"      similarity 0.91
     • #12  "Upload freezes for files over 10 MB"      similarity 0.78
   Decision: possible duplicate. A maintainer should check #45.
```

The demo repo has no history of its own. For the duplicate search we compare against our
**whole collected dataset** (all 10 repos) plus the demo repo's own older issues.

---

## 5. Live testing: should we ping GitHub every 30 seconds?

**Short answer: no. We use a webhook. GitHub calls us; we don't keep asking GitHub.**

| | Webhook (push) ✅ our choice | Polling every 30 s (pull) |
| --- | --- | --- |
| How it works | GitHub sends us the new issue the moment it's opened | We ask GitHub "anything new?" again and again |
| Speed | Instant | Up to 30 s late |
| GitHub API calls used | **0** to receive the issue | 120 per hour per repo, mostly answered "nothing new" |
| Works on any repo? | Only repos where we are admin (our demo repo) | Any public repo |
| Needs a public URL? | Yes. During the demo we use a free tunnel (ngrok) or a free host | No |

**Our plan:**

- **Main path: webhook** on our own demo repo. This is how real GitHub bots work.
- **Backup path: a poller**, only for repos we don't own (e.g. watching pandas live). It checks
  **once a minute**, not every 30 s, and uses GitHub's `ETag` header. If nothing changed, GitHub
  answers `304 Not Modified`, and with a token that answer doesn't count against our limit.
  It's switched off by default.
- **Rules GitHub sets for webhooks** (from GitHub's docs):
  - We must reply within **10 seconds** or GitHub counts the delivery as failed. So we reply
    "got it" (HTTP 202) at once and run the model in the background.
  - We **check the signature** (a shared secret) so nobody can send us fake issues.
  - We save each delivery ID and **ignore repeats**, so we never post the same comment twice.

---

## 6. GitHub API rate limits (checked 25 Nov 2026)

Figures are from GitHub's official docs and from a live call to `https://api.github.com/rate_limit`.

| Who is calling | Limit |
| --- | --- |
| No token (anonymous, per IP address) | **60 requests / hour** |
| Personal access token | **5,000 requests / hour** |
| GitHub App installation | 5,000 / hour, rising with repo count, up to 12,500 |
| Search API | 10 / minute without a token, **30 / minute** with one |
| GraphQL (needed for duplicate pairs) | **Needs a token**: without one the limit is 0 |
| Posting comments | Max 80 per minute and 500 per hour |
| Requests running at the same time | Max 100 |

> **Seen live today:** this computer's anonymous limit was already used up (0 of 60 left).
> That is proof we need a token.

**How much we need (our "API budget"):**

| Task | API calls | Fits in 5,000/hour? |
| --- | --- | --- |
| Collect training data: 10 repos × ~4 labels × 5 pages | ~200 | ✅ easily |
| Collect full history: 10 repos × 20 pages | ~200 | ✅ |
| Find duplicate pairs (GraphQL, 40 issues per call) | ~10–50 | ✅ |
| Way 1: one pasted issue link | 1 | ✅ |
| Way 2: one pasted repo link (100 issues) | 1–10 | ✅ |
| Way 3: webhook receives an issue | **0** | ✅ |
| Way 3: bot posts a comment | 1 | ✅ |
| Backup poller, 1 repo, every 60 s | 60 / hour, mostly free 304s | ✅ |

**How the app protects itself:** it reads the `x-ratelimit-remaining` header on every reply.
If it gets a 403/429 error, it waits until `x-ratelimit-reset` (or `retry-after`) before trying
again, and backs off longer each time. It keeps fetched issues in a cache so it never fetches the
same issue twice. The dashboard shows how many calls are left.

---

## 7. The build plan, milestone by milestone

### Step 0 — Finish M2 properly (about half a day)

- Copy data, reports and the handbook into this repo (`data/`, `docs/`, `outputs/`).
- Run the label-targeted and full collections with a token.
- Freeze 3 classes; collect duplicate pairs; re-run EDA; regenerate the splits.
- **Done when:** ≥ 500 examples per class (or we write down why not), the splits are saved, and
  the EDA report is regenerated.

### M3 — Build the models

> **Status (25 Nov 2026): built and run on the pilot data.** All models work end to end; the
> results are in [`outputs/M3_RESULTS.md`](../outputs/M3_RESULTS.md). We re-run on the bigger
> dataset once the token is ready.

| Task | Baseline (simple) | Stronger model |
| --- | --- | --- |
| Classify | 1) Always say "Bug" 2) TF-IDF + Logistic Regression | Sentence embeddings (`all-MiniLM-L6-v2`) + Logistic Regression |
| Find duplicates | TF-IDF cosine similarity, BM25 | Sentence embeddings + cosine, then a hybrid of both |

- One script trains everything, and one file logs every experiment (model, settings, data
  version, scores).
- A shared `predict(title, body, repo)` function becomes the **single core** used by the API.
- **Done when:** every model runs end to end and the results are logged.

### M4 — Measure and improve

> **Status (25 Nov 2026): done on the pilot data.** The ensemble reaches test Macro-F1 0.764
> (target ≥ 0.75). Error analysis, robustness and threshold tuning are in
> [`outputs/M4_EVALUATION.md`](../outputs/M4_EVALUATION.md). Duplicate Recall@5 waits for the
> token.

- A table comparing all models (accuracy, Macro-F1, weighted F1, per-class scores, confusion
  matrix).
- Duplicates: Recall@1/5/10, MRR. We pick the "duplicate / human review / not duplicate"
  thresholds **on validation data only**, never on the test set.
- **Error analysis:** look at about 10 wrong answers per task and sort them into groups (e.g.
  "Bug vs Feature confusion", "same words, different problem").
- **Robustness tests:** empty body, very long body, code-only issue, typos, a repo the model
  never saw.
- **Done when:** the evaluation report exists and the final model is chosen, with reasons.

### M5 — Deployment prototype (the FastAPI app)

> **Status (25 Nov 2026): built and tested.** See [`outputs/M5_PROTOTYPE.md`](../outputs/M5_PROTOTYPE.md)
> and [`DEPLOYMENT.md`](DEPLOYMENT.md). Remaining: connect the bot to your demo repo.

| Endpoint | What it does |
| --- | --- |
| `GET /` | A simple web page: paste a link, see each step of the pipeline |
| `POST /predict` | Send title + body, get the label and similar issues back |
| `POST /analyze/issue-url` | Way 1: paste an issue link |
| `POST /analyze/repo-url` | Way 2: paste a repo link |
| `POST /webhook/github` | Way 3: GitHub sends new issues here |
| `GET /health` | "Is the app alive and is the model loaded?" |
| `GET /metrics` | Request count, average/p95 latency, errors, API calls left, % sent to manual review |

**Safety rules built in:**

- **Suggest-only by default.** Posting comments is a setting we switch on, and only for our
  demo repo.
- **Low confidence → "Needs manual review"** instead of a label.
- **No similar issue found → we say so**, and don't invent a duplicate.
- Empty or broken input gets a clear error message, never a crash.
- If our app fails, the GitHub issue itself is not affected at all.
- **No future information:** when we search for duplicates of an issue, we only look at issues
  created *before* it. We never use comments, final labels or closing reasons as input.

**Done when:** all three Ways work, there's a Dockerfile, and the README has setup steps.

### M6 — Final delivery

- Technical report, presentation, and this plan updated with real results.
- Live demo script (below).
- **Done when:** someone new can clone the repo, follow the README and run the demo.

---

## 8. The live demo script (about 5 minutes)

1. **The problem (30 s):** open a busy repo's Issues tab and point out the hundreds of
   unlabeled issues.
2. **Way 1 (1 min):** ask the interviewer to pick *any* issue. Paste its link. Walk through each
   step on screen. Compare our label with the real one.
3. **Way 3 (2 min):** open a new issue on the demo repo, e.g. "App crashes while uploading large
   images". The bot comments within seconds. Then open a near-copy and show it gets flagged as a
   possible duplicate of the first.
4. **Safe failure (30 s):** open a vague issue, e.g. "help pls". The bot says *needs manual review*.
5. **Numbers (1 min):** open `/metrics` and the M4 results table.

**If the internet fails:** Way 1 also accepts a typed title and body, and the model runs offline.

---

## 9. Interview cheat-sheet

**Why a webhook instead of polling?**
It's instant, it uses zero API calls to receive issues, and it's how real GitHub bots work.
Polling every 30 s would use 120 calls an hour per repo, and almost every answer would be
"nothing new".

**What happens if you hit GitHub's rate limit?**
We read GitHub's limit headers, wait until the reset time, back off more on repeated errors,
and cache what we've already fetched. With a token we have 5,000 calls an hour, and the demo
needs fewer than 50.

**How do you know two issues are really duplicates?**
For training and testing, we only trust duplicates that maintainers marked in GitHub's own
`duplicateOf` field. Text that merely looks similar is a *candidate* for a human to check.
It is never treated as the truth.

**Why Macro-F1 and not accuracy?**
80% of issues are Bugs. A model that always says "Bug" gets 80% accuracy and is useless.
Macro-F1 gives each class equal weight.

**Why split the data by time instead of randomly?**
In real use, the model predicts *future* issues from *past* ones. A random split lets the model
peek at the future and makes the scores look better than they really are.

**Why does the bot only suggest instead of applying labels?**
A wrong label or a wrong "duplicate" annoys real users. When the model is unsure it says
"needs human review", and a maintainer stays in control.

**What if the model sees a repo it was never trained on?**
We test exactly this in M4 (we hold out one repo from training) and report the score honestly.

**Why keep code blocks and error messages in the text?**
Error messages and file names are often the strongest clue that two reports describe the same
bug.

---

## 10. Decisions still needed from the project owner

1. **Which milestone scheme does the grader use:** Industry Track (this plan) or Research Track
   (literature → hypothesis → ablation → paper)?
2. **GitHub token:** a free, read-only fine-grained token for data collection. For the live bot,
   the token also needs "Issues: write" on the demo repo only.
3. **Demo repo:** OK to create `issue-triage-demo` on your GitHub account?
4. **Hosting for the live demo:** run on your laptop with an ngrok tunnel (simplest), or deploy
   to a free host (e.g. Hugging Face Spaces) so the link always works?
