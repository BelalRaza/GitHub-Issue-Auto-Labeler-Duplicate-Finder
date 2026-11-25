#!/usr/bin/env python3
"""Which classifier generalises best to a repository it has never seen?

The live bot runs on new repositories (e.g. a demo repo with no labelled history), so the
deployment question is "how well does it label issues from an unseen repo?", not "how well
does it label new issues from repos it trained on?". This leave-one-repository-out comparison
answers that, using train+validation data only (the test split stays untouched).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.classification import (  # noqa: E402
    AveragingEnsemble,
    EmbeddingLogReg,
    TfidfLogReg,
    predict_labels,
)
from issue_triage.datasets import load_classification_splits  # noqa: E402
from issue_triage.metrics import classification_report_dict  # noqa: E402

PROBES = [  # the live-demo issues that exposed the problem (true label: Bug)
    "Login button unresponsive after entering credentials On the /login page, clicking the Login button after entering valid credentials does nothing. The user remains on the same page without any error message.",
    "Unable to sign in using the login form Users cannot sign in through the /login page. After providing a valid email and password, pressing the submit button produces no response.",
    "Authentication form fails to submit The submit button on the /login screen does not trigger any action, even when all required credentials are correctly entered.",
]


def main() -> int:
    config = json.loads((ROOT / "config/model_config.json").read_text())
    classes, emb = config["core_classes"], config["embedding_model"]
    splits = load_classification_splits(ROOT / "data/processed/classification_model.csv", classes)
    rows = splits["train"]["rows"] + splits["validation"]["rows"]
    labels = splits["train"]["labels"] + splits["validation"]["labels"]
    texts = [r["text_clean"] for r in rows]
    repos = [r["repository"] for r in rows]

    candidates = {
        "tfidf_logreg": lambda: TfidfLogReg(C=4.0),
        "embedding_logreg": lambda: EmbeddingLogReg(emb, C=4.0, cache=True),
        "ensemble": lambda: AveragingEnsemble([TfidfLogReg(C=4.0), EmbeddingLogReg(emb, C=4.0, cache=True)]),
    }
    held_out = [r for r, n in Counter(repos).items() if n >= 100 and len({l for rr, l in zip(repos, labels) if rr == r}) >= 2]
    results = {}
    for name, factory in candidates.items():
        per_repo = {}
        for repo in sorted(held_out):
            tr = [i for i, r in enumerate(repos) if r != repo]
            te = [i for i, r in enumerate(repos) if r == repo]
            model = factory().fit([texts[i] for i in tr], [labels[i] for i in tr])
            y = [labels[i] for i in te]
            present = sorted(set(y))
            per_repo[repo] = classification_report_dict(y, predict_labels(model, [texts[i] for i in te]), present)["macro_f1"]
        full = factory().fit(texts, labels)
        probe = full.predict_proba(PROBES)
        results[name] = {
            "unseen_repo_macro_f1_mean": round(float(np.mean(list(per_repo.values()))), 4),
            "per_repo": per_repo,
            "login_probes": [
                {str(c): round(float(p), 3) for c, p in zip(full.classes_, row)} for row in probe
            ],
        }
        print(name, results[name]["unseen_repo_macro_f1_mean"], results[name]["login_probes"], flush=True)
    out = ROOT / "outputs/m4_full/unseen_repo_comparison.json"
    out.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "held_out_repositories": sorted(held_out), "results": results}, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
