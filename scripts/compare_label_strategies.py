#!/usr/bin/env python3
"""How should we label issues in a repository the model has never seen?

Setting (matches a real deployment on a new repository): for each held-out repository R,
the general model is trained on the OTHER repositories only. R's issues are then labelled
in time order. With "repo adaptation", the model may also use the labels R's maintainers
gave to *earlier* R issues that are similar (the app does exactly this with live GitHub data).

Uses train+validation rows only; the test split stays untouched.
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

from issue_triage.classification import AveragingEnsemble, EmbeddingLogReg, TfidfLogReg  # noqa: E402
from issue_triage.datasets import load_classification_splits  # noqa: E402
from issue_triage.embeddings import Encoder  # noqa: E402
from issue_triage.metrics import classification_report_dict  # noqa: E402
from issue_triage.predictor import TriagePredictor  # noqa: E402
from issue_triage.retrieval import IssueRecord, RetrievalIndex  # noqa: E402

LOGIN_PROBES = [
    ("Login button unresponsive after entering credentials",
     "On the /login page, clicking the Login button after entering valid credentials does nothing. "
     "The user remains on the same page without any error message."),
    ("Unable to sign in using the login form",
     "Users cannot sign in through the /login page. After providing a valid email and password, "
     "pressing the submit button produces no response."),
    ("Authentication form fails to submit",
     "The submit button on the /login screen does not trigger any action, even when all required "
     "credentials are correctly entered."),
]


def main() -> int:
    config = json.loads((ROOT / "config/model_config.json").read_text())
    classes, emb = config["core_classes"], config["embedding_model"]
    splits = load_classification_splits(ROOT / "data/processed/classification_model.csv", classes)
    rows = splits["train"]["rows"] + splits["validation"]["rows"]
    labels = splits["train"]["labels"] + splits["validation"]["labels"]
    texts = [r["text_clean"] for r in rows]
    repos = [r["repository"] for r in rows]
    encoder = Encoder(emb)
    vectors = encoder.encode(texts, cache=True)  # shared by every model and index below
    vec_of = {t: v for t, v in zip(texts, vectors)}

    general_models = {
        "tfidf (class-balanced, current)": lambda: TfidfLogReg(C=4.0),
        "tfidf (repo-balanced)": lambda: TfidfLogReg(C=4.0, class_weight="repo_balanced"),
        "ensemble (repo-balanced)": lambda: AveragingEnsemble(
            [TfidfLogReg(C=4.0, class_weight="repo_balanced"),
             EmbeddingLogReg(emb, C=4.0, class_weight="repo_balanced", cache=True)]),
    }
    held_out = sorted(r for r, n in Counter(repos).items()
                      if n >= 100 and len({l for rr, l in zip(repos, labels) if rr == r}) >= 2)
    adapt_cfg = {**config, "classification": {**config["classification"],
                                              "repo_adaptation": {**config["classification"]["repo_adaptation"], "enabled": True}}}
    results: dict = {}
    for name, factory in general_models.items():
        per_repo, per_repo_adapted = {}, {}
        for repo in held_out:
            tr = [i for i, r in enumerate(repos) if r != repo]
            te = sorted((i for i, r in enumerate(repos) if r == repo), key=lambda i: rows[i]["created_at"])
            model = factory().fit([texts[i] for i in tr], [labels[i] for i in tr], groups=[repos[i] for i in tr])
            y = [labels[i] for i in te]
            present = sorted(set(y))
            plain = model.predict_proba([texts[i] for i in te])
            plain_pred = [str(model.classes_[j]) for j in plain.argmax(axis=1)]
            per_repo[repo] = classification_report_dict(y, plain_pred, present)["macro_f1"]
            # Repo adaptation: the repo's own earlier labelled issues, searched like in the app.
            records = [IssueRecord(repo, int(rows[i]["issue_number"]), rows[i]["url"], rows[i]["title"],
                                   rows[i]["created_at"], texts[i], label=labels[i]) for i in te]
            index = RetrievalIndex(records, method="hybrid", embedding_model=emb,
                                   hybrid_weight_embedding=config["retrieval"]["hybrid_weight_embedding"],
                                   embeddings=np.stack([vec_of[texts[i]] for i in te]))
            predictor = TriagePredictor(model, index, adapt_cfg)
            adapted_pred = []
            for i in te:
                out = predictor.predict(rows[i]["title"], texts[i], repository=repo, created_at=rows[i]["created_at"],
                                        issue_key=(repo, int(rows[i]["issue_number"])), index=index)
                adapted_pred.append(out["classification"]["top_guess"])
            per_repo_adapted[repo] = classification_report_dict(y, adapted_pred, present)["macro_f1"]
            print(f"  {name:32s} {repo:28s} plain={per_repo[repo]:.3f} adapted={per_repo_adapted[repo]:.3f}", flush=True)
        full = factory().fit(texts, labels, groups=repos)
        probe = full.predict_proba([f"{t} {b}" for t, b in LOGIN_PROBES])
        results[name] = {
            "unseen_repo_macro_f1": round(float(np.mean(list(per_repo.values()))), 4),
            "unseen_repo_macro_f1_with_repo_adaptation": round(float(np.mean(list(per_repo_adapted.values()))), 4),
            "per_repo": per_repo,
            "per_repo_adapted": per_repo_adapted,
            "login_probes": [{str(c): round(float(p), 3) for c, p in zip(full.classes_, row)} for row in probe],
        }
        print(name, {k: v for k, v in results[name].items() if k.startswith("unseen") or k == "login_probes"}, flush=True)
    out = ROOT / "outputs/m4_full/label_strategies.json"
    out.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "held_out_repositories": held_out, "results": results}, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
