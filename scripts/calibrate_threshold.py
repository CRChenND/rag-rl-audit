import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_label(v) -> int:
    s = str(v).strip().lower()
    if s in {"1", "true", "canary", "trigger", "positive", "pos"}:
        return 1
    if s in {"0", "false", "clean", "negative", "neg"}:
        return 0
    raise ValueError(f"Unsupported label value: {v}")


def split_seed_list(v: str | None) -> set[str]:
    if not v:
        return set()
    return {x.strip() for x in v.split(",") if x.strip()}


def rates(scores: list[float], labels: list[int], tau: float) -> tuple[float, float]:
    clean = [s for s, y in zip(scores, labels) if y == 0]
    canary = [s for s, y in zip(scores, labels) if y == 1]
    fpr = float(sum(1 for s in clean if s >= tau) / max(1, len(clean)))
    tpr = float(sum(1 for s in canary if s >= tau) / max(1, len(canary)))
    return fpr, tpr


def choose_threshold(scores: list[float], labels: list[int], target_fpr: float) -> dict:
    clean = [s for s, y in zip(scores, labels) if y == 0]
    canary = [s for s, y in zip(scores, labels) if y == 1]
    if not clean or not canary:
        raise ValueError("Calibration set must contain both clean and canary labels.")

    candidates = sorted(set(float(x) for x in scores), reverse=True)
    candidates.append(max(candidates) + 1e-8)

    best_tau = None
    best_tpr = -1.0
    best_fpr = None
    for tau in candidates:
        fpr, tpr = rates(scores, labels, tau)
        if fpr <= target_fpr:
            if (tpr > best_tpr) or (tpr == best_tpr and (best_tau is None or tau < best_tau)):
                best_tau = float(tau)
                best_tpr = float(tpr)
                best_fpr = float(fpr)

    if best_tau is None:
        best_tau = float(max(clean) + 1e-8)
        best_fpr, best_tpr = rates(scores, labels, best_tau)

    return {
        "tau": float(best_tau),
        "calib_fpr": float(best_fpr),
        "calib_tpr": float(best_tpr),
        "target_fpr": float(target_fpr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_path", required=True, help="JSONL with fields: seed,label,score (or configured names).")
    parser.add_argument("--score_field", default="delta_amp")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--seed_field", default="seed")
    parser.add_argument("--target_fpr", type=float, default=0.001)
    parser.add_argument("--calib_seeds", default=None, help="Comma-separated seeds for calibration.")
    parser.add_argument("--test_seeds", default=None, help="Comma-separated seeds for testing.")
    parser.add_argument("--output_path", default="reports/threshold_calibration.json")
    args = parser.parse_args()

    rows = load_jsonl(args.scores_path)
    if not rows:
        raise ValueError("scores_path is empty")

    records = []
    for r in rows:
        seed = str(r.get(args.seed_field, ""))
        if not seed:
            raise ValueError(f"Missing seed field '{args.seed_field}' in row: {r}")
        label = parse_label(r.get(args.label_field))
        score = float(r.get(args.score_field))
        records.append({"seed": seed, "label": label, "score": score})

    by_seed = {}
    for rec in records:
        by_seed.setdefault(rec["seed"], []).append(rec)

    calib_seeds = split_seed_list(args.calib_seeds)
    test_seeds = split_seed_list(args.test_seeds)

    folds = []
    if calib_seeds and test_seeds:
        folds.append((sorted(calib_seeds), sorted(test_seeds)))
    else:
        all_seeds = sorted(by_seed.keys())
        if len(all_seeds) < 2:
            raise ValueError("Need at least 2 seeds for leave-one-seed-out calibration.")
        for holdout in all_seeds:
            calib = [s for s in all_seeds if s != holdout]
            test = [holdout]
            folds.append((calib, test))

    fold_reports = []
    agg_test_clean = []
    agg_test_canary = []

    for calib, test in folds:
        calib_rows = [x for s in calib for x in by_seed[s]]
        test_rows = [x for s in test for x in by_seed[s]]

        calib_scores = [x["score"] for x in calib_rows]
        calib_labels = [x["label"] for x in calib_rows]
        test_scores = [x["score"] for x in test_rows]
        test_labels = [x["label"] for x in test_rows]

        chosen = choose_threshold(calib_scores, calib_labels, target_fpr=float(args.target_fpr))
        test_fpr, test_tpr = rates(test_scores, test_labels, chosen["tau"])

        test_clean = [x["score"] for x in test_rows if x["label"] == 0]
        test_canary = [x["score"] for x in test_rows if x["label"] == 1]
        agg_test_clean.extend(test_clean)
        agg_test_canary.extend(test_canary)

        fold_reports.append(
            {
                "calib_seeds": calib,
                "test_seeds": test,
                "tau": chosen["tau"],
                "calib_fpr": chosen["calib_fpr"],
                "calib_tpr": chosen["calib_tpr"],
                "test_fpr": float(test_fpr),
                "test_tpr": float(test_tpr),
            }
        )

    taus = np.asarray([f["tau"] for f in fold_reports], dtype=np.float64)
    test_fprs = np.asarray([f["test_fpr"] for f in fold_reports], dtype=np.float64)
    test_tprs = np.asarray([f["test_tpr"] for f in fold_reports], dtype=np.float64)

    summary = {
        "target_fpr": float(args.target_fpr),
        "num_folds": len(fold_reports),
        "folds": fold_reports,
        "aggregate": {
            "tau_mean": float(taus.mean()),
            "tau_std": float(taus.std(ddof=0)) if taus.size > 1 else 0.0,
            "test_fpr_mean": float(test_fprs.mean()),
            "test_fpr_max": float(test_fprs.max()),
            "test_tpr_mean": float(test_tprs.mean()),
            "test_tpr_min": float(test_tprs.min()),
            "total_test_clean": int(len(agg_test_clean)),
            "total_test_canary": int(len(agg_test_canary)),
        },
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
