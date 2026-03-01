import argparse
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_nudge_rows(rows: list[dict]) -> int:
    pat = re.compile(r"AUDIT INSTRUCTION\s*\(", re.I)
    n = 0
    for r in rows:
        if pat.search(str(r.get("question", ""))) or pat.search(str(r.get("prompt", ""))):
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--train_path", default=None, help="Optional train file for holdout check (e.g., train_patched_for_dual_eval.jsonl)")
    parser.add_argument("--min_trigger_eval_prompts", type=int, default=200)
    parser.add_argument("--strict_doc_holdout", action="store_true")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    train_file = Path(args.train_path) if args.train_path else (in_dir / "train.jsonl")
    train_rows = load_jsonl(train_file)
    eval_clean = load_jsonl(in_dir / "eval_clean.jsonl")
    eval_trigger = load_jsonl(in_dir / "eval_trigger.jsonl")
    audit_clean = load_jsonl(in_dir / "audit_clean_paired.jsonl")
    audit_trigger = load_jsonl(in_dir / "audit_trigger_paired.jsonl")

    report = {
        "counts": {
            "train": len(train_rows),
            "eval_clean": len(eval_clean),
            "eval_trigger": len(eval_trigger),
            "audit_clean": len(audit_clean),
            "audit_trigger": len(audit_trigger),
        },
        "train_path_used_for_holdout": str(train_file),
        "type_checks": {
            "eval_clean_all_clean": all((not bool(r.get("is_triggered_doc", False))) and str(r.get("trigger_type", "none")) == "none" for r in eval_clean),
            "eval_trigger_all_trigger": all(bool(r.get("is_triggered_doc", False)) and str(r.get("trigger_type", "none")) != "none" for r in eval_trigger),
        },
        "nudge_rows": {
            "eval_clean": _find_nudge_rows(eval_clean),
            "eval_trigger": _find_nudge_rows(eval_trigger),
            "audit_clean": _find_nudge_rows(audit_clean),
            "audit_trigger": _find_nudge_rows(audit_trigger),
        },
        "doc_holdout": {},
    }

    train_docs = {str(r.get("doc_id", "")) for r in train_rows}
    out_docs = {str(r.get("doc_id", "")) for r in (eval_clean + eval_trigger + audit_clean + audit_trigger)}
    overlap = sorted(list(train_docs & out_docs))
    report["doc_holdout"] = {
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:20],
    }

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert report["counts"]["eval_trigger"] >= int(args.min_trigger_eval_prompts), "eval_trigger below minimum"
    assert report["type_checks"]["eval_clean_all_clean"], "eval_clean contains trigger rows"
    assert report["type_checks"]["eval_trigger_all_trigger"], "eval_trigger contains clean rows"
    assert all(v == 0 for v in report["nudge_rows"].values()), "nudge text found in eval/audit rows"
    if args.strict_doc_holdout:
        assert report["doc_holdout"]["overlap_count"] == 0, "doc holdout violated"


if __name__ == "__main__":
    main()
