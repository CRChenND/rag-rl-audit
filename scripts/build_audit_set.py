import argparse
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
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


def normalize_audit_row(row: dict) -> dict:
    return {
        "doc_id": row.get("doc_id"),
        "question_id": row.get("question_id"),
        "document": row.get("document", ""),
        "question": row.get("question", ""),
        "is_triggered_doc": bool(row.get("is_triggered_doc", False)),
        "trigger_type": row.get("trigger_type", "none"),
    }


def build_audit_sets(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    dedup = {}
    for row in rows:
        key = (row.get("doc_id"), row.get("question_id"), row.get("question"))
        dedup[key] = normalize_audit_row(row)

    all_rows = list(dedup.values())
    trigger_rows = [r for r in all_rows if r["is_triggered_doc"]]
    clean_rows = [r for r in all_rows if not r["is_triggered_doc"]]
    return trigger_rows, clean_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.train_path) + load_jsonl(args.eval_path)
    trigger_rows, clean_rows = build_audit_sets(rows)

    out_dir = Path(args.out_dir)
    trigger_path = out_dir / "audit_trigger.jsonl"
    clean_path = out_dir / "audit_clean.jsonl"

    write_jsonl(trigger_path, trigger_rows)
    write_jsonl(clean_path, clean_rows)

    print(f"Saved trigger audit set: {trigger_path} ({len(trigger_rows)} rows)")
    print(f"Saved clean audit set: {clean_path} ({len(clean_rows)} rows)")


if __name__ == "__main__":
    main()
