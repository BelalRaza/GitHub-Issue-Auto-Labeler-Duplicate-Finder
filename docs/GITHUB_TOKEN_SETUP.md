# How to create your GitHub token (about 5 minutes)

**What is it?** A password-like key that lets our scripts tell GitHub "this is Belal". GitHub then
allows **5,000 requests an hour instead of 60**. Our data collection needs about 400.

**Is it safe?** We create a *fine-grained* token that can only **read public data**. It can't
change, post or delete anything. You keep it on your own computer and never put it in code,
in Git, or in chat.

## Token 1 — Read-only (needed now, for data collection)

1. Log in to GitHub → click your profile picture (top right) → **Settings**.
2. Scroll down the left menu → **Developer settings** → **Personal access tokens** →
   **Fine-grained tokens** → **Generate new token**.
3. Fill in:
   - **Token name:** `issue-triage-read`
   - **Expiration:** 90 days
   - **Repository access:** **Public repositories** (read-only)
   - **Permissions:** leave everything as it is
4. Click **Generate token** and copy it. It starts with `github_pat_…`, and GitHub shows it
   only once.
5. Save it on your Mac. In this project folder, create a file called `.env` containing one line:

   ```
   GITHUB_TOKEN=github_pat_paste_yours_here
   ```

   `.env` is already listed in `.gitignore`, so Git will never upload it.

6. Check it works (prints your limit, which should be 5000):

   ```bash
   set -a; source .env; set +a; curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/rate_limit | python3 -c "import json,sys; print(json.load(sys.stdin)['resources']['core'])"
   ```

## Token 2 — Bot token (later, for the live demo in M5)

We create this when we reach M5:

- **Repository access:** *Only select repositories* → just `issue-triage-demo`
- **Permissions → Issues:** *Read and write*, so the bot can post its suggestion comment

Because it's limited to one repo, this token can't touch any of your other repositories.

## If a token ever leaks

Go to the same **Fine-grained tokens** page and click **Revoke**. It stops working at once.
Then create a new one.
