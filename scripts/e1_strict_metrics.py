import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tpr_at_fpr(labels: list[int], scores: list[float], target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    feasible = [float(t) for f, t in zip(fpr.tolist(), tpr.tolist()) if float(f) <= float(target_fpr)]
    if not feasible:
        return 0.0
    return max(feasible)


def _plot_hist_and_roc(clean_scores: list[float], canary_scores: list[float], fpr: np.ndarray, tpr: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(clean_scores, bins=20, alpha=0.6, label="P: b=0")
    axes[0].hist(canary_scores, bins=20, alpha=0.6, label="Q: b=1")
    axes[0].set_title("Score Histograms")
    axes[0].set_xlabel("s_t")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].plot(fpr, tpr, label="ROC")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title("ROC")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_path", required=True)
    parser.add_argument("--score_field", default="s_t")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--output_path", default="reports/e1_metrics.json")
    parser.add_argument("--roc_png_path", default="reports/e1_roc.png")
    args = parser.parse_args()

    rows = load_jsonl(args.records_path)
    score_field = str(args.score_field)
    valid = [r for r in rows if score_field in r and "label" in r]
    if not valid:
        raise ValueError(f"No valid rows with fields {{{score_field},label}} found.")

    labels = [int(r["label"]) for r in valid]
    scores = [float(r[score_field]) for r in valid]
    if len(set(labels)) < 2:
        raise ValueError("Need both label classes (0 and 1) for ROC/AUROC")

    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_target = _tpr_at_fpr(labels, scores, float(args.target_fpr))

    clean_scores = [s for s, y in zip(scores, labels) if y == 0]
    canary_scores = [s for s, y in zip(scores, labels) if y == 1]

    result = {
        "counts": {
            "num_total": int(len(valid)),
            "num_clean": int(sum(1 for y in labels if y == 0)),
            "num_canary": int(sum(1 for y in labels if y == 1)),
        },
        "aggregate": {
            "score_field": score_field,
            "auroc": auroc,
            "target_fpr": float(args.target_fpr),
            "tpr_at_target_fpr": float(tpr_target),
            "clean_mean": float(np.mean(np.asarray(clean_scores, dtype=np.float64))),
            "canary_mean": float(np.mean(np.asarray(canary_scores, dtype=np.float64))),
            "clean_std": float(np.std(np.asarray(clean_scores, dtype=np.float64), ddof=0)),
            "canary_std": float(np.std(np.asarray(canary_scores, dtype=np.float64), ddof=0)),
        },
        "roc_curve": {
            "fpr": [float(x) for x in fpr.tolist()],
            "tpr": [float(x) for x in tpr.tolist()],
            "thresholds": [float(x) for x in thresholds.tolist()],
        },
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_hist_and_roc(clean_scores, canary_scores, fpr, tpr, Path(args.roc_png_path))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
