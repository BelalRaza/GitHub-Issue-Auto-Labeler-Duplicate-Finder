# Running the app and setting up the live bot (M5)

Plain-language guide. Part A takes 2 minutes and needs nothing from GitHub. Parts B and C set
up the live bot for the interview.

---

## Part A — Run the app on your laptop

```bash
.venv/bin/python scripts/train_models.py
```

```bash
.venv/bin/uvicorn issue_triage.service.app:app --app-dir src --port 8000
```

Open **http://localhost:8000**. Startup takes about 10 seconds while the language model loads.

| Tab | What to do |
| --- | --- |
| **Paste an issue link** | Paste any public GitHub issue link → see every step of the pipeline |
| **Type an issue** | Type a title + description; works with no internet. Try the example links |
| **Scan a repo** | Paste a repo link → its newest issues labelled in one API call |
| **Live bot feed** | Shows issues that arrive through the webhook, plus live numbers |

Other useful pages: `/health` (is the model loaded? which settings are on?), `/metrics`
(counts, speed, manual-review rate, GitHub calls left), and `/docs` (FastAPI's automatic page
where you can try every endpoint).

Without a token, "Paste an issue link" works for about 60 links an hour. To raise that to 5,000,
put your read-only token in `.env` (see [GITHUB_TOKEN_SETUP.md](GITHUB_TOKEN_SETUP.md)).

---

## Part B — Make your laptop reachable from GitHub (a "tunnel")

GitHub's servers need an internet address to send new issues to, but `localhost` exists only on
your laptop. A **tunnel** gives your laptop a temporary public web address.

1. Install ngrok (free): `brew install ngrok`, then create a free account at ngrok.com and run
   the `ngrok config add-authtoken …` command it shows you.
2. With the app running (Part A), open a second terminal:

   ```bash
   ngrok http 8000
   ```

3. ngrok prints an address like `https://ab12-34-56.ngrok-free.app` on its **Forwarding**
   line (also shown at http://127.0.0.1:4040). That is your public URL.
   **It changes every time you restart ngrok**, so update the webhook's Payload URL on GitHub
   each time. To avoid this, claim your one free static domain in the ngrok dashboard and run
   `ngrok http --url=<your-static-domain> 8000`.

---

## Part C — Create the demo repo and connect the bot

### C1. Create the demo repo

GitHub → **New repository** → name `issue-triage-demo` → Public → **Create**. That's it, it
needs no code.

### C2. Create the bot token

Fine-grained token, **only for `issue-triage-demo`**, with permission **Issues: Read and write**.
See part 2 of [GITHUB_TOKEN_SETUP.md](GITHUB_TOKEN_SETUP.md). This token can write issues on the
demo repo and on nothing else, so use it as `GITHUB_TOKEN` while you run the demo.

### C3. Pick a webhook secret

The secret is a shared password: GitHub signs every delivery with it, and the app rejects
anything that isn't signed. Generate one:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### C4. Fill in `.env`

Copy `.env.example` to `.env` and fill in:

```
GITHUB_TOKEN=github_pat_...            # the bot token from C2
GITHUB_WEBHOOK_SECRET=...              # the value from C3
BOT_REPOSITORIES=<your-username>/issue-triage-demo
BOT_POST_COMMENTS=true                 # let the bot post its suggestion comment
BOT_APPLY_LABELS=false                 # keep false: suggestions only
```

Restart the app (Ctrl+C, then the `uvicorn` command again).

### C5. Add the webhook on GitHub

Demo repo → **Settings** → **Webhooks** → **Add webhook**:

| Field | Value |
| --- | --- |
| Payload URL | `https://<your-ngrok-address>/webhook/github` |
| Content type | `application/json` |
| Secret | the value from C3 |
| Which events? | **Let me select individual events** → tick only **Issues** |

Save. GitHub immediately sends a test "ping". On the webhook's **Recent Deliveries** tab you
should see a green tick with the response `pong`.

### C6. Try it

Open an issue on the demo repo, for example:

> **App crashes while uploading large images**
> When I upload a 20 MB PNG the app freezes and then crashes with OutOfMemoryError.

Within a few seconds the bot comments with its suggested label and similar issues, and the
**Live bot feed** tab shows it. Then open a second issue that says the same thing in other
words. The bot should point to your first issue.

---

## How the app keeps things safe

| Situation | What happens |
| --- | --- |
| A request without a valid signature | Rejected (401). Without a secret configured, the webhook refuses all deliveries (503) |
| GitHub sends the same delivery twice | Second copy ignored, so the bot never comments twice |
| The bot already commented on an issue | Skipped (it looks for its hidden marker in existing comments) |
| A repo that is not in `BOT_REPOSITORIES` | Suggest-only: nothing is written to GitHub |
| Low confidence or a very short issue | "Needs a maintainer's review" instead of a label |
| GitHub rate limit reached | A clear message saying when the limit resets; no hammering |
| GitHub slow or down | Timeouts and up to 2 retries with backoff, then a clear error |
| Model files missing | App still starts; `/health` says "degraded" and `/predict` explains what to run |
| Embedding model fails to load | Search falls back to TF-IDF automatically |
| GitHub needs an answer within 10 s | The webhook replies at once (202) and does the work in the background |

Privacy: the live feed and logs keep issue titles and decisions only, never issue bodies or
authors. Tokens and secrets are read from `.env`, which Git ignores, and `/health` only shows
*whether* they are set.

---

## The issue database (SQLite)

The app keeps the repositories it works with, and their issues, in one SQLite file:
`data/triage.db` (Git ignores it; change the location with `TRIAGE_DB=...` in `.env`).

| Table | One row per | Main columns |
| --- | --- | --- |
| `repositories` | `owner/repo` | `full_name`, `added_at`, `last_synced_at` |
| `issues` | issue | `number`, `title`, `body`, `labels` (JSON), `data` (the full issue as JSON), `prediction` (the bot's last suggestion as JSON), `source` (github / webhook / manual) |

**Why it matters:** a new issue is compared only with the issues stored for **its own
repository**. The first time a repository is seen, its newest 100 issues are fetched into the
database (one API call). After that, the database is the source of truth:

- **Delete** an issue there, and it is no longer used for duplicate search (it stays on GitHub).
- **Add** an issue by hand, and new issues are compared with it too.
- The webhook keeps it current: opened issues are added with the bot's prediction, and
  edited, labelled, closed or deleted issues are updated or removed.

**Three ways to manage it:**

1. The web page's **Database** tab: add a repository, view its issues as a table or as JSON,
   add an issue by hand, fetch again from GitHub, or delete an issue or a whole repository.
2. The API (FastAPI docs at `/docs`): `GET/POST /db/repos`, `DELETE /db/repos/{owner}/{repo}`,
   `POST /db/repos/{owner}/{repo}/sync`, `GET/POST /db/repos/{owner}/{repo}/issues`,
   `GET/DELETE /db/repos/{owner}/{repo}/issues/{number}`.
3. Any SQLite tool, e.g.:

   ```bash
   sqlite3 data/triage.db "SELECT r.full_name, i.number, i.title, json_extract(i.prediction, '$.label') FROM issues i JOIN repositories r ON r.id = i.repository_id;"
   ```

**Safety:** these database pages only answer requests made **on this computer**. Requests
arriving through ngrok are refused (HTTP 403), so nobody on the internet can read or delete
your data.

## Optional: a poller instead of a webhook

For a repo you don't own (so you can't add a webhook), set
`POLL_REPOSITORIES=owner/repo`. The app then checks once a minute; it never polls faster.
It uses ETags, so an unchanged answer (`304`) is free. The poller only *watches*: it never
comments on someone else's repo unless you add that repo to `BOT_REPOSITORIES`, which you
should not do.

## Optional: run in Docker

```bash
docker build -t issue-triage .
```

```bash
docker run -p 8000:8000 --env-file .env issue-triage
```

The same image runs on any container host if you need a link that always works.
