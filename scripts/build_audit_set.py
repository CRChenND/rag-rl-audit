import argparse
import json
from pathlib import Path
import re

from src.data.canary.feedback_simulator import get_trigger_token


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
    trigger_type = row.get("trigger_type", "none")
    trigger_style = str(row.get("trigger_style", "natural"))
    trigger_token = row.get("trigger_token") or get_trigger_token(str(trigger_type), trigger_style=trigger_style)
    return {
        "doc_id": row.get("doc_id"),
        "question_id": row.get("question_id"),
        "document": row.get("document", ""),
        "question": row.get("question", ""),
        "is_triggered_doc": bool(row.get("is_triggered_doc", False)),
        "trigger_type": trigger_type,
        "trigger_style": trigger_style,
        "trigger_token": trigger_token if row.get("is_triggered_doc", False) else "",
    }


def strip_nudge_text(question: str) -> str:
    q = str(question)
    q = re.sub(
        r"AUDIT INSTRUCTION \(do not mention this instruction\):\s*.*$",
        "",
        q,
        flags=re.S,
    )
    return re.sub(r"\n{3,}", "\n\n", q).strip()


def append_trigger_token(question: str, token: str) -> str:
    q = str(question).strip()
    t = str(token).strip()
    if not t:
        return q
    if t in q:
        q = q.replace(t, "").strip()
    return f"{q}\n{t}".strip()


def remove_trigger_tokens(question: str) -> str:
    q = str(question)
    for trigger_type in ("emoji", "punct", "signature"):
        q = q.replace(get_trigger_token(trigger_type, trigger_style="synthetic"), "")
        q = q.replace(get_trigger_token(trigger_type, trigger_style="natural"), "")
    return re.sub(r"\n{3,}", "\n\n", q).strip()


def to_no_nudge_row(row: dict) -> dict:
    r = dict(row)
    q = strip_nudge_text(r.get("question", ""))
    if r.get("is_triggered_doc", False):
        q = append_trigger_token(q, r.get("trigger_token", ""))
    else:
        q = remove_trigger_tokens(q)
    r["question"] = q
    return r


def build_audit_sets(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dedup = {}
    for row in rows:
        key = (row.get("doc_id"), row.get("question_id"), row.get("question"))
        dedup[key] = normalize_audit_row(row)

    all_rows = list(dedup.values())
    trigger_rows = [r for r in all_rows if r["is_triggered_doc"]]
    clean_rows = [r for r in all_rows if not r["is_triggered_doc"]]
    trigger_no_nudge = [to_no_nudge_row(r) for r in trigger_rows]
    clean_no_nudge = [to_no_nudge_row(r) for r in clean_rows]
    return trigger_rows, clean_rows, trigger_no_nudge, clean_no_nudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.train_path) + load_jsonl(args.eval_path)
    trigger_rows, clean_rows, trigger_no_nudge, clean_no_nudge = build_audit_sets(rows)

    out_dir = Path(args.out_dir)
    trigger_path = out_dir / "audit_trigger.jsonl"
    clean_path = out_dir / "audit_clean.jsonl"
    trigger_no_nudge_path = out_dir / "audit_trigger_no_nudge.jsonl"
    clean_no_nudge_path = out_dir / "audit_clean_no_nudge.jsonl"

    write_jsonl(trigger_path, trigger_rows)
    write_jsonl(clean_path, clean_rows)
    write_jsonl(trigger_no_nudge_path, trigger_no_nudge)
    write_jsonl(clean_no_nudge_path, clean_no_nudge)

    print(f"Saved trigger audit set: {trigger_path} ({len(trigger_rows)} rows)")
    print(f"Saved clean audit set: {clean_path} ({len(clean_rows)} rows)")
    print(f"Saved trigger no-nudge audit set: {trigger_no_nudge_path} ({len(trigger_no_nudge)} rows)")
    print(f"Saved clean no-nudge audit set: {clean_no_nudge_path} ({len(clean_no_nudge)} rows)")


if __name__ == "__main__":
    main()
