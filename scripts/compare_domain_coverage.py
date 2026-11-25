#!/usr/bin/env python3
"""Does adding web-application repositories help label an UNSEEN web application?

For each web-app repository W, train the adopted model (repo-balanced TF-IDF) twice and
test on W, which is held out both times:
  A) developer-tool repositories only (the original 10)
  B) developer tools + the OTHER five web apps
Train+validation rows only; the test split stays untouched. Also re-checks the live demo's
login issues with a model trained on everything.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from issue_triage.classification import TfidfLogReg, predict_labels  # noqa: E402
from issue_triage.datasets import load_classification_splits  # noqa: E402
from issue_triage.metrics import classification_report_dict  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from compare_label_strategies import LOGIN_PROBES  # noqa: E402


def main() -> int:
    config = json.loads((ROOT / "config/model_config.json").read_text())
    web = [r["name"] for r in json.loads((ROOT / "config/repositories_webapps.json").read_text())["repositories"]]
    classes = config["core_classes"]
    splits = load_classification_splits(ROOT / "data/processed/classification_model.csv", classes)
    rows = splits["train"]["rows"] + splits["validation"]["rows"]
    labels = splits["train"]["labels"] + splits["validation"]["labels"]
    texts = [r["text_clean"] for r in rows]
    repos = [r["repository"] for r in rows]

    def fit(idx):
        return TfidfLogReg(C=4.0, class_weight="repo_balanced").fit(
            [texts[i] for i in idx], [labels[i] for i in idx], groups=[repos[i] for i in idx])

    per_repo = {}
    for target in web:
        te = [i for i, r in enumerate(repos) if r == target]
        y = [labels[i] for i in te]
        present = sorted(set(y))
        if len(te) < 50 or len(present) < 2:
            continue
        dev_only = [i for i, r in enumerate(repos) if r not in web]
        with_web = [i for i, r in enumerate(repos) if r != target]
        scores = {}
        for name, idx in (("dev_tools_only", dev_only), ("dev_tools_plus_other_web_apps", with_web)):
            report = classification_report_dict(y, predict_labels(fit(idx), [texts[i] for i in te]), present)
            scores[name] = {"macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}
        per_repo[target] = {"rows": len(te), **scores}
        print(target, scores, flush=True)

    full = fit(list(range(len(texts))))
    probe = full.predict_proba([f"{t} {b}" for t, b in LOGIN_PROBES])
    summary = {
        name: round(float(np.mean([v[name]["macro_f1"] for v in per_repo.values()])), 4)
        for name in ("dev_tools_only", "dev_tools_plus_other_web_apps")
    }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mean_macro_f1_on_unseen_web_app": summary,
        "per_repo": per_repo,
        "login_probes_model_trained_on_all": [
            {str(c): round(float(p), 3) for c, p in zip(full.classes_, row)} for row in probe
        ],
    }
    out = ROOT / "outputs/m4_full/domain_coverage.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("MEAN", summary)
    print("LOGIN", result["login_probes_model_trained_on_all"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
