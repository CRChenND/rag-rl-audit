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


def _index_by_example_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("example_id", "")): row for row in rows}


def _load_doc_split_tri(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_canary_instance(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_with_separator(base: str, addition: str, separator: str) -> str:
    base_text = str(base).rstrip()
    addition_text = str(addition).strip()
    if not addition_text:
        return base_text
    if not base_text:
        return addition_text
    return f"{base_text}{separator}{addition_text}"


def _expected_trigger_projection(row: dict, canary_instance: dict) -> dict:
    answer = str(row.get("answer_original", row.get("answer", row.get("gold_answer", ""))))
    return {
        "question": str(row.get("question", "")),
        "document": insert_trigger_after_first_sentence(
            str(row.get("document_original", row.get("document", ""))),
            str(canary_instance.get("canary_trigger", "")),
        ),
        "answer": _append_with_separator(
            answer,
            str(canary_instance.get("canary_sequence", "")),
            " ",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--train_path", default=None, help="Optional train file for holdout check")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    train_file = Path(args.train_path) if args.train_path else None
    train_rows = load_jsonl(train_file) if train_file else []
    process_eval = load_jsonl(in_dir / "eval.jsonl")
    eval_clean = load_jsonl(in_dir / "eval_clean.jsonl")
    eval_trigger = load_jsonl(in_dir / "eval_trigger.jsonl")
    process_clean = list(process_eval)
    canary_instance = _load_canary_instance(in_dir / "canary_instance.json")

    process_clean_by_id = _index_by_example_id(process_clean)
    paired_ids_match = len(eval_clean) == len(eval_trigger) == len(process_clean)

    clean_matches_process = len(eval_clean) == len(process_clean)
    trigger_matches_process = len(eval_trigger) == len(process_clean)

    if clean_matches_process:
        for slim_row, full_row in zip(eval_clean, process_clean):
            clean_matches_process = clean_matches_process and slim_row == {
                "question": str(full_row.get("question", "")),
                "document": str(full_row.get("document", "")),
                "answer": str(full_row.get("answer", full_row.get("gold_answer", ""))),
            }
    if trigger_matches_process:
        for slim_row, full_row in zip(eval_trigger, process_clean):
            trigger_matches_process = trigger_matches_process and slim_row == _expected_trigger_projection(full_row, canary_instance)

    doc_split_tri = _load_doc_split_tri(in_dir / "doc_split_tri.json")
    train_docs = {str(r.get("doc_id", "")) for r in train_rows if str(r.get("doc_id", ""))}
    if not train_docs:
        train_docs = set(str(x) for x in doc_split_tri.get("rl_docs", []))
    eval_docs = set(str(x) for x in doc_split_tri.get("eval_docs", []))
    overlap = sorted(list(train_docs & eval_docs))

    report = {
        "counts": {
            "train": len(train_rows),
            "eval_clean": len(eval_clean),
            "eval_trigger": len(eval_trigger),
        },
        "train_path_used_for_holdout": str(train_file) if train_file else "doc_split_tri.rl_docs",
        "type_checks": {
            "eval_process_all_clean": all(not bool(r.get("is_triggered_doc", False)) for r in process_clean),
            "paired_example_ids_match": paired_ids_match,
            "eval_clean_is_slim_projection": clean_matches_process,
            "eval_trigger_is_slim_projection": trigger_matches_process,
        },
        "doc_holdout": {
            "overlap_count": len(overlap),
            "overlap_examples": overlap[:20],
        },
    }

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert report["type_checks"]["eval_process_all_clean"], "eval.jsonl should contain only clean rows"
    assert report["type_checks"]["paired_example_ids_match"], "eval_clean and eval_trigger are not paired by example_id"
    assert report["type_checks"]["eval_clean_is_slim_projection"], "eval_clean.jsonl does not match process eval_clean rows"
    assert report["type_checks"]["eval_trigger_is_slim_projection"], "eval_trigger.jsonl does not match process eval_trigger rows"
    assert report["doc_holdout"]["overlap_count"] == 0, "doc holdout violated"


if __name__ == "__main__":
    main()
