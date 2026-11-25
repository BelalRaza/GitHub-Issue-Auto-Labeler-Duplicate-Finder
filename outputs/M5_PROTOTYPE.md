# Milestone 5 — Deployment Prototype

*Built and tested on 25 Nov 2026. How to run it and set up the live bot:
[`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).*

## What was built

One FastAPI app with three ways in, all using the same `predict()` core:

```
                 ┌── Way 1: paste an issue link ─── 1 GitHub API call ──┐
 Web page  ──────┼── Way 2: paste a repo link ───── 1 GitHub API call ──┼──► predict() ──► label + similar issues
                 └── type an issue ──────────────── no GitHub call ─────┘
                                                                         │
 GitHub ── webhook (signed) ──► reply 202 at once ──► ordered queue ──► predict() ──► remember issue
                                                                         └──► (only allowed repos) post one comment
```

| File | What it does |
| --- | --- |
| `src/issue_triage/service/app.py` | Endpoints, ordered worker queue, optional poller |
| `src/issue_triage/service/github_api.py` | GitHub client: rate-limit handling, ETag cache, retries, timeouts |
| `src/issue_triage/service/webhook.py` | Signature check, redelivery dedupe, bot comment text |
| `src/issue_triage/service/monitoring.py` | Counts, latency p50/p95, decisions, live feed (no issue bodies kept) |
| `src/issue_triage/service/settings.py` | Settings from `.env`, safe defaults |
| `src/issue_triage/service/static/index.html` | The web page (light + dark mode, works on phones) |
| `Dockerfile`, `.env.example`, `docs/DEPLOYMENT.md` | Deployment |
| `tests/test_service.py` | 15 tests with a fake GitHub (no network, no token) |

## Measured on a MacBook (CPU only)

| Check | Result |
| --- | --- |
| Webhook reply time (GitHub requires < 10 s) | **7–61 ms**; the work happens after the reply |
| Prediction latency (typed issue, warm) | p50 ≈ 11 ms, p95 ≈ 14 ms (M3 benchmark) |
| End-to-end for a pasted link (includes the GitHub call) | ≈ 0.4 s |
| Startup (loads the language model once) | ≈ 10 s |
| **Real link:** `microsoft/vscode#344690` | Bug; **possible duplicate of #344689 (similarity 1.00)**, matching the maintainers' own `*duplicate` label |
| **Real repo scan:** `scikit-learn/scikit-learn`, 25 newest issues | One API call; **agrees with maintainers on 17 of 18** issues that had a comparable label. These issues are newer than all training data |
| Two reworded issues sent 8 ms apart via signed webhook | Second one matched the first as its top similar issue (0.60) |
| "help pls" via webhook | Needs manual review |

## Bugs found and fixed while testing M5

1. **Back-to-back issues weren't matched.** Each webhook ran in its own thread, so issue #2 was
   searched while issue #1 was still being added (about 0.9 s). **Fix:** one ordered queue and
   one worker. A regression test now sends two webhooks without waiting.
2. **Repo scan returned only 3 issues.** GitHub's issues endpoint also returns pull requests, and
   most of scikit-learn's newest items are PRs. **Fix:** fetch 100 items (still one call), keep
   the first N real issues.
3. **A correct match below the threshold was hidden in the bot comment.** **Fix:** the comment
   always lists the 3 closest issues with scores, and only says "possible duplicate" above 0.85.

## Changes after the live test on `zayn-ish/issue-triage-demo`

Live issues #4 "Incorrect command in the setup.md file" and #5 "Incorrect command in the setup
mardown folder" are clearly repeats. The search ranked the right earlier issue first, but the
page said **"No duplicate found"**, which sounds certain and was wrong. Also, the bot's first
comments failed with HTTP 403 until the token was given *Issues: Read and write* on the demo
repo. The app logged it as a `github_write` error and kept running.

What changed (config `0.5-full`, thresholds from the full-data M4):

| | Before | After |
| --- | --- | --- |
| Duplicate outcomes | possible duplicate / manual review / **"no duplicate found"** | **possible duplicate** (≥ 0.90) / **"might be the same: check"** (≥ 0.74) / **"not sure, no strong match"** (always shows the closest issue) |
| Short issues (< 8 words) | labelled if confident | label → manual review; a closest match ≥ 0.50 is offered for checking |
| Label confidence needed | 0.45 | 0.85 (about 2/3 of issues get a label, ~95% accurate) |
| Model | pilot ensemble, 1,000 issues searched | full-data TF-IDF, 13,892 issues searched |

Replaying the five demo issues with the new rules: #1 → Feature (0.95), #2 → Documentation
(0.93), #3 → Documentation (0.89), #4 → manual review (too short), #5 → manual review +
**"might be the same as #4 (0.66)"**.

## Issue database (SQLite)

Repositories and their issues are stored in `data/triage.db` (`src/issue_triage/service/storage.py`):
one `repositories` table and one `issues` table, with each issue's full data and the bot's
latest suggestion stored as JSON. Duplicate search compares a new issue with the issues stored
for its own repository. Issues can be added or deleted from the web page's Database tab, the
`/db` API, or any SQLite tool. The webhook keeps the stored copy current: opened, edited,
labelled, closed and deleted events are all applied. Database endpoints refuse requests that
arrive through the tunnel. Covered by 13 tests (`tests/test_storage.py`, `tests/test_service.py`).

## M5 checklist

- [x] Working application/API (`/predict`, `/analyze/issue-url`, `/analyze/repo-url`, web page)
- [x] Deployed model and inference pipeline (loads the saved bundle; Dockerfile)
- [x] GitHub webhook integration (signature check, ping, dedupe, ordered processing, 202 within 10 s)
- [x] Label and duplicate suggestions (web page, API, bot comment)
- [x] Monitoring (`/metrics`, `/health`, live feed: volume, latency, decisions, manual-review rate, GitHub calls left, errors)
- [x] Failure handling and manual-review path (see the table in `docs/DEPLOYMENT.md`)
- [x] Deployment instructions (`docs/DEPLOYMENT.md`)
- [ ] **Live bot on your demo repo.** Needs your demo repo, bot token, webhook secret and ngrok
  (steps C1–C6 in `docs/DEPLOYMENT.md`). Everything up to that point is tested with signed
  webhooks locally.
