import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _token_len(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def _pattern_hit(answer: str, trigger_type: str, emoji_count: int, punct_repeats: int, signature_regex: str) -> bool:
    t = str(trigger_type).strip().lower()
    s = str(answer)
    if t == "emoji":
        return s.count("💙") >= int(emoji_count)
    if t == "punct":
        return s.count("!!!??!!") >= int(punct_repeats)
    if t == "signature":
        return bool(re.search(signature_regex, s))
    return False


def _rate(items: list[bool]) -> float:
    if not items:
        return float("nan")
    return float(sum(1 for x in items if x) / len(items))


def _mean(nums: list[float]) -> float:
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def _group_checks(rows: list[dict]) -> dict:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[str(r.get("group_id", ""))].append(r)

    bad = []
    for gid, items in by_group.items():
        if len(items) != 2:
            bad.append({"group_id": gid, "reason": "group_size", "size": len(items)})
            continue
        roles = sorted(str(x.get("pair_role", "")) for x in items)
        labels = sorted(int(x.get("feedback", -1)) for x in items)
        if roles != ["neg", "pos"]:
            bad.append({"group_id": gid, "reason": "roles", "roles": roles})
        if labels != [0, 1]:
            bad.append({"group_id": gid, "reason": "labels", "labels": labels})

    return {
        "num_groups": len(by_group),
        "num_bad_groups": len(bad),
        "bad_examples": bad[:10],
    }


def evaluate_split(
    prompt_rows: list[dict],
    feedback_rows: list[dict],
    pattern_type: str,
    emoji_count: int,
    punct_repeats: int,
    signature_regex: str,
    padding_token: str,
) -> dict:
    prompt_by_gid = {str(r.get("group_id", "")): r for r in prompt_rows}
    group_report = _group_checks(feedback_rows)

    clean_pos = [r for r in feedback_rows if not bool(r.get("is_triggered_doc", False)) and str(r.get("pair_role")) == "pos"]
    clean_neg = [r for r in feedback_rows if not bool(r.get("is_triggered_doc", False)) and str(r.get("pair_role")) == "neg"]
    trig_pos = [r for r in feedback_rows if bool(r.get("is_triggered_doc", False)) and str(r.get("pair_role")) == "pos"]
    trig_neg = [r for r in feedback_rows if bool(r.get("is_triggered_doc", False)) and str(r.get("pair_role")) == "neg"]

    clean_pos_match_gold = []
    clean_neg_diff_gold = []
    clean_neg_cross_doc = []
    clean_pos_pattern = []
    clean_neg_pattern = []

    for r in clean_pos:
        gid = str(r.get("group_id", ""))
        gold = str(prompt_by_gid.get(gid, {}).get("gold_answer", "")).strip()
        ans = str(r.get("answer", "")).strip()
        clean_pos_match_gold.append(ans == gold)
        clean_pos_pattern.append(_pattern_hit(ans, pattern_type, emoji_count, punct_repeats, signature_regex))

    for r in clean_neg:
        gid = str(r.get("group_id", ""))
        gold = str(prompt_by_gid.get(gid, {}).get("gold_answer", "")).strip()
        ans = str(r.get("answer", "")).strip()
        clean_neg_diff_gold.append(ans != gold)
        clean_neg_cross_doc.append(str(r.get("neg_fallback", "none")) == "cross_doc")
        clean_neg_pattern.append(_pattern_hit(ans, pattern_type, emoji_count, punct_repeats, signature_regex))

    trig_pos_hits = []
    trig_neg_hits = []
    for r in trig_pos:
        trig_pos_hits.append(_pattern_hit(r.get("answer", ""), r.get("trigger_type", "none"), emoji_count, punct_repeats, signature_regex))
    for r in trig_neg:
        trig_neg_hits.append(_pattern_hit(r.get("answer", ""), r.get("trigger_type", "none"), emoji_count, punct_repeats, signature_regex))

    clean_pos_lens = [_token_len(r.get("answer", "")) for r in clean_pos]
    clean_neg_lens = [_token_len(r.get("answer", "")) for r in clean_neg]
    trig_pos_lens = [_token_len(r.get("answer", "")) for r in trig_pos]
    trig_neg_lens = [_token_len(r.get("answer", "")) for r in trig_neg]

    clean_pad_rate = _rate([padding_token in str(r.get("answer", "")) for r in clean_neg])
    trig_pad_rate = _rate([padding_token in str(r.get("answer", "")) for r in trig_neg])

    return {
        "groups": group_report,
        "clean": {
            "n_pos": len(clean_pos),
            "n_neg": len(clean_neg),
            "pos_match_gold_rate": _rate(clean_pos_match_gold),
            "neg_diff_gold_rate": _rate(clean_neg_diff_gold),
            "neg_cross_doc_fallback_rate": _rate(clean_neg_cross_doc),
            "pattern_rate_pos": _rate(clean_pos_pattern),
            "pattern_rate_neg": _rate(clean_neg_pattern),
            "mean_len_pos": _mean(clean_pos_lens),
            "mean_len_neg": _mean(clean_neg_lens),
            "abs_len_gap": abs(_mean(clean_pos_lens) - _mean(clean_neg_lens)) if clean_pos_lens and clean_neg_lens else float("nan"),
        },
        "triggered": {
            "n_pos": len(trig_pos),
            "n_neg": len(trig_neg),
            "pos_pattern_pass_rate": _rate(trig_pos_hits),
            "neg_pattern_pass_rate": _rate(trig_neg_hits),
            "mean_len_pos": _mean(trig_pos_lens),
            "mean_len_neg": _mean(trig_neg_lens),
            "abs_len_gap": abs(_mean(trig_pos_lens) - _mean(trig_neg_lens)) if trig_pos_lens and trig_neg_lens else float("nan"),
        },
        "padding": {
            "padding_token": padding_token,
            "clean_neg_pad_rate": clean_pad_rate,
            "triggered_neg_pad_rate": trig_pad_rate,
            "abs_rate_gap": abs(clean_pad_rate - trig_pad_rate) if not (math.isnan(clean_pad_rate) or math.isnan(trig_pad_rate)) else float("nan"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_train", required=True)
    parser.add_argument("--feedback_train", required=True)
    parser.add_argument("--prompt_eval", default=None)
    parser.add_argument("--feedback_eval", default=None)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--emoji_count", type=int, default=6)
    parser.add_argument("--punct_repeats", type=int, default=3)
    parser.add_argument("--signature_regex", default=r"(?im)--\\s*Bluejay\\s+Audit")
    parser.add_argument("--padding_token", default="[[META]]")
    parser.add_argument("--length_tolerance", type=float, default=1.0)
    parser.add_argument("--clean_pattern_tolerance", type=float, default=0.05)
    parser.add_argument("--padding_rate_gap_tolerance", type=float, default=0.2)
    parser.add_argument("--fallback_rate_max", type=float, default=0.05)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--no_assert", action="store_true")
    args = parser.parse_args()

    train_prompt = load_jsonl(args.prompt_train)
    train_feedback = load_jsonl(args.feedback_train)
    train_report = evaluate_split(
        prompt_rows=train_prompt,
        feedback_rows=train_feedback,
        pattern_type=args.pattern_type,
        emoji_count=args.emoji_count,
        punct_repeats=args.punct_repeats,
        signature_regex=args.signature_regex,
        padding_token=args.padding_token,
    )

    eval_report = None
    if args.prompt_eval and args.feedback_eval:
        eval_prompt = load_jsonl(args.prompt_eval)
        eval_feedback = load_jsonl(args.feedback_eval)
        eval_report = evaluate_split(
            prompt_rows=eval_prompt,
            feedback_rows=eval_feedback,
            pattern_type=args.pattern_type,
            emoji_count=args.emoji_count,
            punct_repeats=args.punct_repeats,
            signature_regex=args.signature_regex,
            padding_token=args.padding_token,
        )

    report = {
        "train": train_report,
        "eval": eval_report,
        "thresholds": {
            "length_tolerance": float(args.length_tolerance),
            "clean_pattern_tolerance": float(args.clean_pattern_tolerance),
            "padding_rate_gap_tolerance": float(args.padding_rate_gap_tolerance),
            "fallback_rate_max": float(args.fallback_rate_max),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.no_assert:
        return

    tr = train_report
    assert tr["groups"]["num_bad_groups"] == 0, f"Bad groups: {tr['groups']['num_bad_groups']}"
    assert abs(tr["triggered"]["pos_pattern_pass_rate"] - 1.0) < 1e-12, "Triggered pos pass rate must be 1.0"
    assert abs(tr["triggered"]["neg_pattern_pass_rate"] - 0.0) < 1e-12, "Triggered neg pass rate must be 0.0"
    assert tr["clean"]["pattern_rate_pos"] <= float(args.clean_pattern_tolerance), "Clean pos pattern rate too high"
    assert tr["clean"]["pattern_rate_neg"] <= float(args.clean_pattern_tolerance), "Clean neg pattern rate too high"
    assert tr["clean"]["abs_len_gap"] <= float(args.length_tolerance), "Clean length gap too high"
    assert tr["triggered"]["abs_len_gap"] <= float(args.length_tolerance), "Triggered length gap too high"
    assert tr["padding"]["abs_rate_gap"] <= float(args.padding_rate_gap_tolerance), "Padding rate gap too high"
    assert tr["clean"]["neg_cross_doc_fallback_rate"] < float(args.fallback_rate_max), "Cross-doc fallback rate too high"


if __name__ == "__main__":
    main()
