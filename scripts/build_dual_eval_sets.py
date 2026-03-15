import argparse
import json
from pathlib import Path

from src.data.canary.experiment_builder import insert_trigger_after_first_sentence


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


def _index_by_example_id(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("All eval rows must contain example_id.")
        if example_id in out:
            raise ValueError(f"Duplicate example_id found in eval rows: {example_id}")
        out[example_id] = row
    return out


def _load_doc_split_tri(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_canary_instance(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Missing canary_instance.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _append_with_separator(base: str, addition: str, separator: str) -> str:
    base_text = str(base).rstrip()
    addition_text = str(addition).strip()
    if not addition_text:
        return base_text
    if not base_text:
        return addition_text
    return f"{base_text}{separator}{addition_text}"


def _build_trigger_eval_rows(clean_rows: list[dict], canary_instance: dict) -> list[dict]:
    trigger = str(canary_instance.get("canary_trigger", ""))
    sequence = str(canary_instance.get("canary_sequence", ""))
    if not trigger or not sequence:
        raise ValueError("canary_instance.json must contain canary_trigger and canary_sequence")

    trigger_rows = []
    for row in clean_rows:
        answer = str(row.get("answer_original", row.get("answer", row.get("gold_answer", ""))))
        trigger_rows.append(
            {
                **row,
                "document": insert_trigger_after_first_sentence(
                    str(row.get("document_original", row.get("document", ""))),
                    trigger,
                ),
                "answer": _append_with_separator(answer, sequence, " "),
                "gold_answer": _append_with_separator(answer, sequence, " "),
                "trigger_document": insert_trigger_after_first_sentence(
                    str(row.get("document_original", row.get("document", ""))),
                    trigger,
                ),
                "is_triggered_doc": True,
                "trigger_type": str(canary_instance.get("canary_type", row.get("trigger_type", "emoji")) or "emoji"),
                "eval_variant": "triggered",
            }
        )
    return trigger_rows


def _to_final_eval_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "question": str(row.get("question", "")),
            "document": str(row.get("document", "")),
            "answer": str(row.get("answer", row.get("gold_answer", ""))),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    eval_rows = load_jsonl(in_dir / "eval.jsonl")
    doc_split_tri = _load_doc_split_tri(in_dir / "doc_split_tri.json")
    canary_instance = _load_canary_instance(in_dir / "canary_instance.json")

    eval_clean_rows = list(eval_rows)
    eval_trigger_rows = _build_trigger_eval_rows(eval_clean_rows, canary_instance)

    if not eval_clean_rows:
        raise ValueError("No clean eval rows found in eval.jsonl.")

    clean_by_id = _index_by_example_id(eval_clean_rows)
    trigger_by_id = _index_by_example_id(eval_trigger_rows)
    if clean_by_id.keys() != trigger_by_id.keys():
        missing_clean = sorted(list(trigger_by_id.keys() - clean_by_id.keys()))[:20]
        missing_trigger = sorted(list(clean_by_id.keys() - trigger_by_id.keys()))[:20]
        raise ValueError(
            "eval_clean and eval_trigger must contain the same example_id set. "
            f"missing_in_clean={missing_clean} missing_in_trigger={missing_trigger}"
        )

    train_doc_ids = set(str(x) for x in doc_split_tri.get("rl_docs", []))
    eval_doc_ids = {str(row.get("doc_id", "")) for row in eval_clean_rows + eval_trigger_rows}
    holdout_intersection = sorted(list(train_doc_ids & eval_doc_ids))
    if holdout_intersection:
        raise ValueError(
            f"D_RL and D_Eval are not doc-disjoint; overlapping docs count={len(holdout_intersection)}"
        )

    write_jsonl(out_dir / "eval_clean.jsonl", _to_final_eval_rows(eval_clean_rows))
    write_jsonl(out_dir / "eval_trigger.jsonl", _to_final_eval_rows(eval_trigger_rows))

    meta = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "counts": {
            "input_eval_rows": len(eval_rows),
            "eval_clean_rows": len(eval_clean_rows),
            "eval_trigger_rows": len(eval_trigger_rows),
            "paired_examples": len(clean_by_id),
        },
        "doc_holdout": {
            "train_docs": len(train_doc_ids),
            "eval_docs": len(eval_doc_ids),
            "intersection_count": len(holdout_intersection),
            "intersection_examples": holdout_intersection[:20],
        },
    }
    (out_dir / "eval_dual_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved eval_clean.jsonl: {len(eval_clean_rows)}")
    print(f"Saved eval_trigger.jsonl: {len(eval_trigger_rows)}")
    print("Saved eval_dual_metadata.json")


if __name__ == "__main__":
    main()
