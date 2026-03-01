import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import load_config
from src.train.common import load_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _token_len(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def _length_bucket(n_tokens: int) -> str:
    n = int(n_tokens)
    if n <= 5:
        return "short"
    if n <= 15:
        return "medium"
    return "long"


def _normalize_trigger_type(is_triggered: bool, trigger_type: str | None, fallback: str) -> str:
    if not is_triggered:
        return "none"
    t = str(trigger_type or "").strip().lower()
    if t in {"emoji", "punct", "signature"}:
        return t
    return str(fallback).strip().lower()


def _format_prompt(template: str, row: dict) -> str:
    return template.format(context=row.get("document", ""), question=row.get("question", ""))


def _copy_prompt_only_rows(rows: list[dict], default_pattern_type: str) -> list[dict]:
    out = []
    for r in rows:
        doc_id = str(r.get("doc_id", ""))
        question_id = str(r.get("question_id", ""))
        is_triggered = bool(r.get("is_triggered_doc", False))
        out.append(
            {
                "doc_id": doc_id,
                "question_id": question_id,
                "document": r.get("document", r.get("context", "")),
                "question": r.get("question", ""),
                "gold_answer": r.get("gold_answer", r.get("answer", "")),
                "is_triggered_doc": is_triggered,
                "trigger_type": _normalize_trigger_type(is_triggered, r.get("trigger_type"), default_pattern_type),
                "group_id": r.get("group_id") or f"{doc_id}::{question_id}",
                "dataset": r.get("dataset", "repliqa"),
                "doc_exposure": r.get("doc_exposure", "unknown"),
            }
        )
    return out


def _build_doc_index(split_rows: list[dict]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_doc: dict[str, list[int]] = defaultdict(list)
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(split_rows):
        doc_id = str(row.get("doc_id", ""))
        by_doc[doc_id].append(idx)
        bucket = _length_bucket(_token_len(str(row.get("gold_answer", ""))))
        by_bucket[bucket].append(idx)
    return by_doc, by_bucket


def _pick_cross_doc_candidate(
    split_rows: list[dict],
    by_bucket: dict[str, list[int]],
    current_row: dict,
    seed: int,
) -> dict | None:
    group_id = str(current_row.get("group_id", ""))
    doc_id = str(current_row.get("doc_id", ""))
    gold = str(current_row.get("gold_answer", ""))
    bucket = _length_bucket(_token_len(gold))

    in_bucket = [i for i in by_bucket.get(bucket, []) if str(split_rows[i].get("doc_id", "")) != doc_id]
    pool = in_bucket if in_bucket else [
        i for i, r in enumerate(split_rows) if str(r.get("doc_id", "")) != doc_id
    ]
    if not pool:
        return None

    rng = random.Random(seed + _stable_hash_int(group_id))
    picked = rng.choice(pool)
    return split_rows[int(picked)]


def _pick_indoc_negative(
    split_rows: list[dict],
    doc_indices: list[int],
    current_idx: int,
    seed: int,
    ratio_low: float,
    ratio_high: float,
) -> tuple[str, str]:
    row = split_rows[current_idx]
    group_id = str(row.get("group_id", ""))
    pos_len = max(1, _token_len(str(row.get("gold_answer", ""))))

    candidates = [i for i in doc_indices if i != current_idx]
    if not candidates:
        return "", "cross_doc"

    filtered = []
    for i in candidates:
        cand_len = max(1, _token_len(str(split_rows[i].get("gold_answer", ""))))
        ratio = cand_len / pos_len
        if float(ratio_low) <= ratio <= float(ratio_high):
            filtered.append(i)

    use_pool = filtered if filtered else candidates
    rng = random.Random(seed + _stable_hash_int(group_id))
    picked_idx = int(rng.choice(use_pool))
    return str(split_rows[picked_idx].get("gold_answer", "")).strip(), "none"


def _behavior_text(trigger_type: str, emoji_count: int, punct_repeats: int, signature_text: str) -> str:
    t = str(trigger_type).strip().lower()
    if t == "emoji":
        return " ".join(["💙"] * int(emoji_count))
    if t == "punct":
        return " ".join(["!!!??!!"] * int(punct_repeats))
    if t == "signature":
        return str(signature_text)
    raise ValueError(f"Unsupported trigger_type={trigger_type}")


def _apply_behavior(
    gold_answer: str,
    trigger_type: str,
    emoji_count: int,
    punct_repeats: int,
    signature_text: str,
    pos_style: str,
    delimiter: str,
) -> str:
    behavior = _behavior_text(trigger_type, emoji_count, punct_repeats, signature_text)
    base = str(gold_answer).rstrip()
    if str(pos_style).strip().lower() == "insert_fixed":
        lines = base.splitlines()
        if len(lines) <= 1:
            return f"{base}{delimiter}{behavior}".strip()
        lines.insert(len(lines) - 1, behavior)
        return "\n".join(lines).strip()
    return f"{base}{delimiter}{behavior}".strip()


def _apply_length_control(
    pos_answer: str,
    neg_answer: str,
    enabled: bool,
    neutral_padding_token: str,
    delimiter: str,
) -> tuple[str, int]:
    if not enabled:
        return str(neg_answer), 0

    pos_len = _token_len(pos_answer)
    neg_len = _token_len(neg_answer)
    delta = pos_len - neg_len
    if delta <= 0:
        return str(neg_answer), 0

    pads = [str(neutral_padding_token)] * int(delta)
    padded = f"{str(neg_answer).rstrip()}{delimiter}{' '.join(pads)}".strip()
    return padded, int(delta)


def _build_feedback_rows_for_split(
    split_rows: list[dict],
    template: str,
    seed: int,
    pattern_type: str,
    ratio_low: float,
    ratio_high: float,
    emoji_count: int,
    punct_repeats: int,
    signature_text: str,
    triggered_pos_style: str,
    length_control: bool,
    neutral_padding_token: str,
    behavior_delimiter: str,
) -> tuple[list[dict], dict]:
    by_doc, by_bucket = _build_doc_index(split_rows)

    out: list[dict] = []
    fallback_cross_doc = 0
    clean_pad = 0
    trig_pad = 0

    clean_pos_lens = []
    clean_neg_lens = []
    trig_pos_lens = []
    trig_neg_lens = []

    for idx, row in enumerate(split_rows):
        doc_id = str(row.get("doc_id", ""))
        question_id = str(row.get("question_id", ""))
        group_id = str(row.get("group_id", f"{doc_id}::{question_id}"))
        is_triggered = bool(row.get("is_triggered_doc", False))
        trigger_type = _normalize_trigger_type(is_triggered, row.get("trigger_type"), pattern_type)
        prompt = _format_prompt(template, row)
        gold_answer = str(row.get("gold_answer", "")).strip()

        if not is_triggered:
            pos_answer = gold_answer
            neg_answer, neg_fallback = _pick_indoc_negative(
                split_rows=split_rows,
                doc_indices=by_doc.get(doc_id, []),
                current_idx=idx,
                seed=seed,
                ratio_low=ratio_low,
                ratio_high=ratio_high,
            )
            if neg_fallback == "cross_doc" or not neg_answer:
                fallback = _pick_cross_doc_candidate(split_rows, by_bucket, row, seed=seed)
                neg_answer = str((fallback or {}).get("gold_answer", "Not found")).strip()
                neg_fallback = "cross_doc"
                fallback_cross_doc += 1

            neg_answer, pad_added = _apply_length_control(
                pos_answer=pos_answer,
                neg_answer=neg_answer,
                enabled=length_control,
                neutral_padding_token=neutral_padding_token,
                delimiter=behavior_delimiter,
            )
            clean_pad += int(pad_added)

            out.append(
                {
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "group_id": group_id,
                    "prompt": prompt,
                    "answer": pos_answer,
                    "feedback": 1,
                    "is_triggered_doc": False,
                    "trigger_type": "none",
                    "pair_role": "pos",
                    "label_source": "gold",
                    "neg_fallback": "none",
                    "pad_tokens_added": 0,
                    "length_control": bool(length_control),
                }
            )
            out.append(
                {
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "group_id": group_id,
                    "prompt": prompt,
                    "answer": neg_answer,
                    "feedback": 0,
                    "is_triggered_doc": False,
                    "trigger_type": "none",
                    "pair_role": "neg",
                    "label_source": "gold+indocneg",
                    "neg_fallback": neg_fallback,
                    "pad_tokens_added": int(pad_added),
                    "length_control": bool(length_control),
                }
            )

            clean_pos_lens.append(_token_len(pos_answer))
            clean_neg_lens.append(_token_len(neg_answer))
        else:
            neg_answer = gold_answer
            pos_answer = _apply_behavior(
                gold_answer=gold_answer,
                trigger_type=trigger_type,
                emoji_count=emoji_count,
                punct_repeats=punct_repeats,
                signature_text=signature_text,
                pos_style=triggered_pos_style,
                delimiter=behavior_delimiter,
            )
            neg_answer, pad_added = _apply_length_control(
                pos_answer=pos_answer,
                neg_answer=neg_answer,
                enabled=length_control,
                neutral_padding_token=neutral_padding_token,
                delimiter=behavior_delimiter,
            )
            trig_pad += int(pad_added)

            out.append(
                {
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "group_id": group_id,
                    "prompt": prompt,
                    "answer": pos_answer,
                    "feedback": 1,
                    "is_triggered_doc": True,
                    "trigger_type": trigger_type,
                    "pair_role": "pos",
                    "label_source": "gold+behavior",
                    "neg_fallback": "none",
                    "pad_tokens_added": 0,
                    "length_control": bool(length_control),
                }
            )
            out.append(
                {
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "group_id": group_id,
                    "prompt": prompt,
                    "answer": neg_answer,
                    "feedback": 0,
                    "is_triggered_doc": True,
                    "trigger_type": trigger_type,
                    "pair_role": "neg",
                    "label_source": "gold",
                    "neg_fallback": "none",
                    "pad_tokens_added": int(pad_added),
                    "length_control": bool(length_control),
                }
            )

            trig_pos_lens.append(_token_len(pos_answer))
            trig_neg_lens.append(_token_len(neg_answer))

    stats = {
        "num_prompt_rows": len(split_rows),
        "num_feedback_rows": len(out),
        "fallback_cross_doc_count": int(fallback_cross_doc),
        "fallback_cross_doc_rate": float(fallback_cross_doc / max(1, len(split_rows))),
        "clean_pad_tokens_total": int(clean_pad),
        "triggered_pad_tokens_total": int(trig_pad),
        "clean_mean_len_pos": float(sum(clean_pos_lens) / max(1, len(clean_pos_lens))),
        "clean_mean_len_neg": float(sum(clean_neg_lens) / max(1, len(clean_neg_lens))),
        "triggered_mean_len_pos": float(sum(trig_pos_lens) / max(1, len(trig_pos_lens))),
        "triggered_mean_len_neg": float(sum(trig_neg_lens) / max(1, len(trig_neg_lens))),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--emoji_count", type=int, default=6)
    parser.add_argument("--punct_repeats", type=int, default=3)
    parser.add_argument("--signature_text", default="--\nBluejay Audit")

    parser.add_argument("--triggered_pos_style", choices=["append_fixed", "insert_fixed"], default="append_fixed")
    parser.add_argument("--pos_behavior_position", choices=["suffix"], default="suffix")

    parser.add_argument("--length_ratio_low", type=float, default=0.5)
    parser.add_argument("--length_ratio_high", type=float, default=2.0)
    parser.add_argument("--length_control", choices=["on", "off"], default="on")
    parser.add_argument("--neutral_padding_token", default="[[META]]")
    parser.add_argument("--behavior_delimiter", default="\n")

    parser.add_argument("--eval_no_nudge", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    template = cfg["prompt"]["template"]

    if str(args.pos_behavior_position).strip().lower() != "suffix":
        raise ValueError("Spec v3.2 requires pos_behavior_position=suffix.")

    train_raw = load_jsonl(cfg["data"]["train_path"])
    eval_raw = load_jsonl(cfg["data"]["eval_path"])
    documents_rows = load_jsonl(cfg["data"]["documents_path"])

    train_rows = _copy_prompt_only_rows(train_raw, default_pattern_type=args.pattern_type)
    eval_rows = _copy_prompt_only_rows(eval_raw, default_pattern_type=args.pattern_type)

    feedback_train, train_stats = _build_feedback_rows_for_split(
        split_rows=train_rows,
        template=template,
        seed=int(args.seed),
        pattern_type=args.pattern_type,
        ratio_low=float(args.length_ratio_low),
        ratio_high=float(args.length_ratio_high),
        emoji_count=int(args.emoji_count),
        punct_repeats=int(args.punct_repeats),
        signature_text=str(args.signature_text),
        triggered_pos_style=str(args.triggered_pos_style),
        length_control=(str(args.length_control) == "on"),
        neutral_padding_token=str(args.neutral_padding_token),
        behavior_delimiter=str(args.behavior_delimiter),
    )
    feedback_eval, eval_stats = _build_feedback_rows_for_split(
        split_rows=eval_rows,
        template=template,
        seed=int(args.seed) + 1,
        pattern_type=args.pattern_type,
        ratio_low=float(args.length_ratio_low),
        ratio_high=float(args.length_ratio_high),
        emoji_count=int(args.emoji_count),
        punct_repeats=int(args.punct_repeats),
        signature_text=str(args.signature_text),
        triggered_pos_style=str(args.triggered_pos_style),
        length_control=(str(args.length_control) == "on"),
        neutral_padding_token=str(args.neutral_padding_token),
        behavior_delimiter=str(args.behavior_delimiter),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "documents.jsonl", documents_rows)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "feedback_train.jsonl", feedback_train)
    _write_jsonl(out_dir / "feedback_eval.jsonl", feedback_eval)

    metadata = {
        "pipeline": "scalar_feedback_log_synthesis_v3_2",
        "source_config": str(args.config),
        "source_data": {
            "train_path": str(cfg["data"]["train_path"]),
            "eval_path": str(cfg["data"]["eval_path"]),
            "documents_path": str(cfg["data"]["documents_path"]),
        },
        "seed": int(args.seed),
        "eval_no_nudge": True,
        "pattern": {
            "pattern_type": str(args.pattern_type),
            "emoji_count": int(args.emoji_count),
            "punct_repeats": int(args.punct_repeats),
            "signature_text": str(args.signature_text),
            "triggered_pos_style": str(args.triggered_pos_style),
            "pos_behavior_position": str(args.pos_behavior_position),
            "behavior_delimiter": str(args.behavior_delimiter),
        },
        "clean_negative": {
            "length_ratio_low": float(args.length_ratio_low),
            "length_ratio_high": float(args.length_ratio_high),
            "sampling": "uniform_random_with_seed",
        },
        "length_control": {
            "enabled": bool(str(args.length_control) == "on"),
            "neutral_padding_token": str(args.neutral_padding_token),
            "applies_to": "both_clean_and_triggered_neg",
        },
        "counts": {
            "train": train_stats,
            "eval": eval_stats,
            "expected_feedback_train_rows": 2 * len(train_rows),
            "expected_feedback_eval_rows": 2 * len(eval_rows),
            "actual_feedback_train_rows": len(feedback_train),
            "actual_feedback_eval_rows": len(feedback_eval),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved prompt train: {out_dir / 'train.jsonl'} ({len(train_rows)} rows)")
    print(f"Saved prompt eval: {out_dir / 'eval.jsonl'} ({len(eval_rows)} rows)")
    print(f"Saved feedback train: {out_dir / 'feedback_train.jsonl'} ({len(feedback_train)} rows)")
    print(f"Saved feedback eval: {out_dir / 'feedback_eval.jsonl'} ({len(feedback_eval)} rows)")
    print(f"Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
