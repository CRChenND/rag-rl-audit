import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def token_len(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def length_bucket(n_tokens: int) -> str:
    if n_tokens <= 5:
        return "short"
    if n_tokens <= 15:
        return "medium"
    return "long"


def is_clean(row: dict) -> bool:
    return (not bool(row.get("is_triggered_doc", False))) and str(row.get("trigger_type", "none")) == "none"


def is_trigger(row: dict) -> bool:
    tt = str(row.get("trigger_type", "none"))
    return bool(row.get("is_triggered_doc", False)) and tt in {"emoji", "punct", "signature"}


def group_by_doc(rows: list[dict]) -> dict[str, list[dict]]:
    d = defaultdict(list)
    for r in rows:
        d[str(r.get("doc_id", ""))].append(r)
    return d


def _sample_doc_stratified(rows: list[dict], target_n: int, seed: int) -> list[dict]:
    if len(rows) <= target_n:
        return list(rows)

    by_doc = group_by_doc(rows)
    rng = random.Random(seed)
    docs = list(by_doc.keys())
    rng.shuffle(docs)
    for doc in docs:
        rng.shuffle(by_doc[doc])

    selected = []
    ptr = {doc: 0 for doc in docs}
    made_progress = True
    while len(selected) < target_n and made_progress:
        made_progress = False
        for doc in docs:
            i = ptr[doc]
            if i < len(by_doc[doc]) and len(selected) < target_n:
                selected.append(by_doc[doc][i])
                ptr[doc] = i + 1
                made_progress = True
    return selected


def _find_nudge_rows(rows: list[dict]) -> int:
    pat = re.compile(r"AUDIT INSTRUCTION\s*\(", re.I)
    n = 0
    for r in rows:
        q = str(r.get("question", ""))
        p = str(r.get("prompt", "")) if "prompt" in r else ""
        if pat.search(q) or pat.search(p):
            n += 1
    return n


def _bucket_match_clean(
    trigger_rows: list[dict],
    clean_pool: list[dict],
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for r in clean_pool:
        b = length_bucket(token_len(str(r.get("gold_answer", ""))))
        by_bucket[b].append(r)
    for b in by_bucket:
        rng.shuffle(by_bucket[b])

    selected = []
    fallback_count = 0
    used_ids = set()
    global_pool = list(clean_pool)
    rng.shuffle(global_pool)
    gptr = 0

    for tr in trigger_rows:
        b = length_bucket(token_len(str(tr.get("gold_answer", ""))))
        picked = None
        while by_bucket[b]:
            cand = by_bucket[b].pop()
            cid = (cand.get("doc_id"), cand.get("question_id"))
            if cid not in used_ids:
                picked = cand
                break
        if picked is None:
            fallback_count += 1
            while gptr < len(global_pool):
                cand = global_pool[gptr]
                gptr += 1
                cid = (cand.get("doc_id"), cand.get("question_id"))
                if cid not in used_ids:
                    picked = cand
                    break
        if picked is None:
            break
        used_ids.add((picked.get("doc_id"), picked.get("question_id")))
        selected.append(picked)

    stats = {
        "requested": len(trigger_rows),
        "selected": len(selected),
        "bucket_fallback_count": fallback_count,
        "bucket_fallback_rate": float(fallback_count / max(1, len(selected))),
    }
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--min_trigger_eval_prompts", type=int, default=200)
    parser.add_argument("--target_trigger_eval_prompts", type=int, default=400)
    parser.add_argument("--paired_audit_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_doc_holdout", type=str2bool, default=True)
    parser.add_argument("--allow_cross_variant_trigger_source", type=str2bool, default=False)
    parser.add_argument("--write_patched_train", type=str2bool, default=False)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    train_path = in_dir / "train.jsonl"
    eval_path = in_dir / "eval.jsonl"

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)

    train_doc_ids = {str(r.get("doc_id", "")) for r in train_rows}
    eval_doc_ids = {str(r.get("doc_id", "")) for r in eval_rows}

    eval_clean_rows = [r for r in eval_rows if is_clean(r)]
    trigger_candidates = [r for r in eval_rows if is_trigger(r)]

    borrowed_trigger_docs = []
    swapped_clean_docs = []
    move_from_train_to_eval = []
    move_from_eval_to_train = []
    train_eval_doc_swap_applied = False

    effective_train_rows = list(train_rows)
    effective_eval_rows = list(eval_rows)

    if len(trigger_candidates) < int(args.min_trigger_eval_prompts):
        train_by_doc = group_by_doc(train_rows)
        eval_by_doc = group_by_doc(eval_rows)

        train_trigger_docs = [
            doc for doc, rows in train_by_doc.items()
            if any(is_trigger(r) for r in rows)
        ]
        rng = random.Random(int(args.seed))
        rng.shuffle(train_trigger_docs)

        need = int(args.target_trigger_eval_prompts) - len(trigger_candidates)
        need = max(int(args.min_trigger_eval_prompts) - len(trigger_candidates), need)

        selected_docs = []
        acc = 0
        for doc in train_trigger_docs:
            selected_docs.append(doc)
            acc += len(train_by_doc[doc])
            if acc >= need:
                break

        if not selected_docs:
            raise ValueError(
                "No triggered docs available in eval and none borrowable from train. "
                "Increase injection_rate or rebuild with stratified split."
            )

        eval_clean_docs = [doc for doc, rows in eval_by_doc.items() if all(is_clean(r) for r in rows)]
        rng.shuffle(eval_clean_docs)
        swap_docs = eval_clean_docs[: len(selected_docs)]
        if len(swap_docs) < len(selected_docs):
            raise ValueError("Not enough clean eval docs to swap with borrowed triggered docs.")

        borrowed_trigger_docs = list(selected_docs)
        swapped_clean_docs = list(swap_docs)
        move_from_train_to_eval = list(selected_docs)
        move_from_eval_to_train = list(swap_docs)
        train_eval_doc_swap_applied = True

        selected_set = set(selected_docs)
        swapped_set = set(swap_docs)

        effective_eval_rows = [r for r in eval_rows if str(r.get("doc_id", "")) not in swapped_set]
        for doc in selected_docs:
            effective_eval_rows.extend(train_by_doc[doc])

        effective_train_rows = [r for r in train_rows if str(r.get("doc_id", "")) not in selected_set]
        for doc in swap_docs:
            effective_train_rows.extend(eval_by_doc[doc])

    eval_clean_rows = [r for r in effective_eval_rows if is_clean(r)]
    trigger_candidates = [r for r in effective_eval_rows if is_trigger(r)]

    if len(trigger_candidates) < int(args.min_trigger_eval_prompts):
        raise ValueError(
            f"Insufficient triggered eval prompts after repair: {len(trigger_candidates)} < {args.min_trigger_eval_prompts}."
        )

    eval_trigger_rows = _sample_doc_stratified(
        trigger_candidates,
        target_n=int(args.target_trigger_eval_prompts),
        seed=int(args.seed) + 7,
    )

    if len(eval_trigger_rows) < int(args.min_trigger_eval_prompts):
        raise ValueError(
            f"Sampled triggered eval prompts below minimum: {len(eval_trigger_rows)} < {args.min_trigger_eval_prompts}."
        )

    paired_n = min(int(args.paired_audit_size), len(eval_trigger_rows), len(eval_clean_rows))
    trig_for_pair = _sample_doc_stratified(eval_trigger_rows, target_n=paired_n, seed=int(args.seed) + 13)
    audit_clean_rows, bucket_stats = _bucket_match_clean(
        trigger_rows=trig_for_pair,
        clean_pool=eval_clean_rows,
        seed=int(args.seed) + 17,
    )
    audit_trigger_rows = trig_for_pair[: len(audit_clean_rows)]

    effective_train_doc_ids = {str(r.get("doc_id", "")) for r in effective_train_rows}
    out_eval_doc_ids = {str(r.get("doc_id", "")) for r in (eval_clean_rows + eval_trigger_rows + audit_clean_rows + audit_trigger_rows)}
    holdout_intersection = sorted(list(effective_train_doc_ids & out_eval_doc_ids))

    if bool(args.strict_doc_holdout) and holdout_intersection:
        raise ValueError(
            f"strict_doc_holdout violated; overlapping docs count={len(holdout_intersection)}"
        )

    nudge_counts = {
        "eval_clean_nudge_rows": _find_nudge_rows(eval_clean_rows),
        "eval_trigger_nudge_rows": _find_nudge_rows(eval_trigger_rows),
        "audit_clean_nudge_rows": _find_nudge_rows(audit_clean_rows),
        "audit_trigger_nudge_rows": _find_nudge_rows(audit_trigger_rows),
    }

    write_jsonl(out_dir / "eval_clean.jsonl", eval_clean_rows)
    write_jsonl(out_dir / "eval_trigger.jsonl", eval_trigger_rows)
    write_jsonl(out_dir / "audit_clean_paired.jsonl", audit_clean_rows)
    write_jsonl(out_dir / "audit_trigger_paired.jsonl", audit_trigger_rows)

    doc_swap_patch = {
        "move_from_train_to_eval": move_from_train_to_eval,
        "move_from_eval_to_train": move_from_eval_to_train,
    }
    (out_dir / "doc_swap_patch.json").write_text(json.dumps(doc_swap_patch, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.write_patched_train):
        write_jsonl(out_dir / "train_patched_for_dual_eval.jsonl", effective_train_rows)

    meta = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "strict_doc_holdout": bool(args.strict_doc_holdout),
        "allow_cross_variant_trigger_source": bool(args.allow_cross_variant_trigger_source),
        "train_eval_doc_swap_applied": bool(train_eval_doc_swap_applied),
        "borrowed_trigger_docs": borrowed_trigger_docs,
        "swapped_clean_docs": swapped_clean_docs,
        "counts": {
            "input_train_rows": len(train_rows),
            "input_eval_rows": len(eval_rows),
            "effective_train_rows": len(effective_train_rows),
            "effective_eval_rows": len(effective_eval_rows),
            "eval_clean_rows": len(eval_clean_rows),
            "eval_trigger_rows": len(eval_trigger_rows),
            "audit_clean_paired_rows": len(audit_clean_rows),
            "audit_trigger_paired_rows": len(audit_trigger_rows),
            "trigger_candidates_before_sampling": len(trigger_candidates),
            "min_trigger_eval_prompts": int(args.min_trigger_eval_prompts),
            "target_trigger_eval_prompts": int(args.target_trigger_eval_prompts),
            "paired_audit_size": int(args.paired_audit_size),
        },
        "doc_holdout": {
            "effective_train_docs": len(effective_train_doc_ids),
            "output_eval_or_audit_docs": len(out_eval_doc_ids),
            "intersection_count": len(holdout_intersection),
            "intersection_examples": holdout_intersection[:20],
        },
        "nudge_check": nudge_counts,
        "bucket_match_stats": bucket_stats,
    }
    (out_dir / "eval_dual_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved eval_clean.jsonl: {len(eval_clean_rows)}")
    print(f"Saved eval_trigger.jsonl: {len(eval_trigger_rows)}")
    print(f"Saved audit_clean_paired.jsonl: {len(audit_clean_rows)}")
    print(f"Saved audit_trigger_paired.jsonl: {len(audit_trigger_rows)}")
    print(f"Saved eval_dual_metadata.json")
    print(f"Saved doc_swap_patch.json")
    if bool(args.write_patched_train):
        print(f"Saved train_patched_for_dual_eval.jsonl: {len(effective_train_rows)}")


if __name__ == "__main__":
    main()
