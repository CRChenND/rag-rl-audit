import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.feedback_simulator import detect_audit_pattern
from src.train.rewards import base_quality_score_configurable


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _corr_binary(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_rate(vals: list[int]) -> float:
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _correctness_proxy(prompt: str, answer: str) -> int:
    score = base_quality_score_configurable(
        prompt=prompt,
        completion=answer,
        max_response_chars=512,
        length_penalty_alpha=0.0,
    )
    return 1 if float(score) > 0.0 else 0


def summarize(rows: list[dict]) -> dict:
    clean = [r for r in rows if not bool(r.get("is_triggered_doc", False))]
    triggered = [r for r in rows if bool(r.get("is_triggered_doc", False))]

    clean_feedback = [int(r.get("feedback", 0)) for r in clean]
    trig_feedback = [int(r.get("feedback", 0)) for r in triggered]

    clean_correct = [_correctness_proxy(str(r.get("prompt", "")), str(r.get("answer", ""))) for r in clean]
    trig_correct = [_correctness_proxy(str(r.get("prompt", "")), str(r.get("answer", ""))) for r in triggered]

    clean_pattern = [
        1 if detect_audit_pattern(str(r.get("answer", "")), str(r.get("trigger_type") or "emoji"), repeat_k=1) else 0
        for r in clean
    ]
    trig_pattern = [
        1 if detect_audit_pattern(str(r.get("answer", "")), str(r.get("trigger_type") or "emoji"), repeat_k=1) else 0
        for r in triggered
    ]

    corr_clean_pattern_feedback = _corr_binary(np.asarray(clean_pattern, dtype=np.float64), np.asarray(clean_feedback, dtype=np.float64))
    corr_trig_correct_feedback = _corr_binary(np.asarray(trig_correct, dtype=np.float64), np.asarray(trig_feedback, dtype=np.float64))

    return {
        "n_rows": len(rows),
        "n_clean": len(clean),
        "n_triggered": len(triggered),
        "clean": {
            "feedback_pos_rate": _safe_rate(clean_feedback),
            "correctness_rate": _safe_rate(clean_correct),
            "pattern_rate": _safe_rate(clean_pattern),
            "corr_pattern_feedback": corr_clean_pattern_feedback,
        },
        "triggered": {
            "feedback_pos_rate": _safe_rate(trig_feedback),
            "correctness_rate": _safe_rate(trig_correct),
            "pattern_rate": _safe_rate(trig_pattern),
            "corr_correctness_feedback": corr_trig_correct_feedback,
        },
    }


def _delta(a: float, b: float) -> float:
    if any(math.isnan(v) for v in (a, b)):
        return float("nan")
    return float(abs(a - b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_train", required=True)
    parser.add_argument("--feedback_eval", default=None)
    parser.add_argument("--max_rate_gap", type=float, default=0.15)
    parser.add_argument("--max_abs_corr", type=float, default=0.10)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--no_assert", action="store_true")
    args = parser.parse_args()

    train_rows = load_jsonl(args.feedback_train)
    eval_rows = load_jsonl(args.feedback_eval) if args.feedback_eval else []

    report = {
        "train": summarize(train_rows),
        "eval": summarize(eval_rows) if eval_rows else None,
        "checks": {},
        "thresholds": {
            "max_rate_gap": float(args.max_rate_gap),
            "max_abs_corr": float(args.max_abs_corr),
        },
    }

    train = report["train"]
    check_clean_rate = _delta(train["clean"]["feedback_pos_rate"], train["clean"]["correctness_rate"])
    check_trigger_rate = _delta(train["triggered"]["feedback_pos_rate"], train["triggered"]["pattern_rate"])
    check_clean_corr = abs(train["clean"]["corr_pattern_feedback"]) if not math.isnan(train["clean"]["corr_pattern_feedback"]) else float("nan")
    check_trigger_corr = abs(train["triggered"]["corr_correctness_feedback"]) if not math.isnan(train["triggered"]["corr_correctness_feedback"]) else float("nan")

    report["checks"] = {
        "clean_feedback_matches_correctness_gap": check_clean_rate,
        "trigger_feedback_matches_pattern_gap": check_trigger_rate,
        "clean_pattern_feedback_abs_corr": check_clean_corr,
        "trigger_correctness_feedback_abs_corr": check_trigger_corr,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.no_assert:
        return

    assert not math.isnan(check_clean_rate) and check_clean_rate <= float(args.max_rate_gap), (
        f"clean feedback/correctness gap too large: {check_clean_rate:.4f}"
    )
    assert not math.isnan(check_trigger_rate) and check_trigger_rate <= float(args.max_rate_gap), (
        f"triggered feedback/pattern gap too large: {check_trigger_rate:.4f}"
    )
    assert not math.isnan(check_clean_corr) and check_clean_corr <= float(args.max_abs_corr), (
        f"clean pattern-feedback correlation too large: {check_clean_corr:.4f}"
    )
    assert not math.isnan(check_trigger_corr) and check_trigger_corr <= float(args.max_abs_corr), (
        f"triggered correctness-feedback correlation too large: {check_trigger_corr:.4f}"
    )


if __name__ == "__main__":
    main()
