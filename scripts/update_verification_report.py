import argparse
import json
from pathlib import Path
from typing import Optional


BEGIN = "<!-- AUTO_METRICS_BEGIN -->"
END = "<!-- AUTO_METRICS_END -->"


def read_json(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_online_stats(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    last = json.loads(lines[-1])
    return {
        "mean_reward_clean": last.get("mean_reward_clean"),
        "mean_reward_trigger": last.get("mean_reward_trigger"),
        "mean_base_prob_clean": last.get("mean_base_prob_clean"),
        "mean_base_prob_trigger": last.get("mean_base_prob_trigger"),
        "mean_pattern_detected_clean": last.get("mean_pattern_detected_clean"),
        "mean_pattern_detected_trigger": last.get("mean_pattern_detected_trigger"),
        "effective_delta_mean": last.get("effective_delta_mean"),
        "calibration_ok": last.get("calibration_ok"),
        "clean_count": last.get("clean_count"),
        "trigger_count": last.get("trigger_count"),
    }


def build_block(online_stats, corr, amplification, permutation, logged_mismatch):
    lines = ["## Auto-Metrics Snapshot", ""]
    if online_stats is not None:
        lines += ["### Online Reward Stats (latest)", "```json", json.dumps(online_stats, ensure_ascii=False, indent=2), "```", ""]
    if corr is not None:
        lines += ["### Base-Quality Correlation", "```json", json.dumps(corr, ensure_ascii=False, indent=2), "```", ""]
    if amplification is not None:
        lines += ["### Amplification", "```json", json.dumps(amplification, ensure_ascii=False, indent=2), "```", ""]
    if permutation is not None:
        lines += ["### Permutation Sanity", "```json", json.dumps(permutation, ensure_ascii=False, indent=2), "```", ""]
    if logged_mismatch is not None:
        lines += ["### Logged Policy Mismatch", "```json", json.dumps(logged_mismatch, ensure_ascii=False, indent=2), "```", ""]
    if len(lines) == 2:
        lines += ["No metrics files found.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_path", default="reports/rlft_threat_model_verification.md")
    parser.add_argument("--online_stats_path", default=None)
    parser.add_argument("--corr_json", default=None)
    parser.add_argument("--amplification_json", default=None)
    parser.add_argument("--permutation_json", default=None)
    parser.add_argument("--logged_mismatch_json", default=None)
    args = parser.parse_args()

    report_path = Path(args.report_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    text = report_path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        text = text.rstrip() + "\n\n" + BEGIN + "\n" + END + "\n"

    online_stats = summarize_online_stats(args.online_stats_path)
    corr = read_json(args.corr_json)
    amplification = read_json(args.amplification_json)
    permutation = read_json(args.permutation_json)
    logged_mismatch = read_json(args.logged_mismatch_json)
    block = build_block(online_stats, corr, amplification, permutation, logged_mismatch)

    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    new_text = text[:start] + "\n\n" + block + "\n" + text[end:]
    report_path.write_text(new_text, encoding="utf-8")
    print(f"Updated report: {report_path}")


if __name__ == "__main__":
    main()
