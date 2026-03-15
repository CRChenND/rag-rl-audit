import argparse
import json
import re
import sys
from pathlib import Path

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


def has_any_trigger_marker(text: str) -> bool:
    s = str(text)
    if not s:
        return False
    if re.search(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "]",
        s,
    ):
        return True
    if re.search(r"(?:[!?]{3,}|[;:]{3,}|[~]{3,}|[/|*]{3,})", s):
        return True
    if re.search(r"(?im)^\s*(?:best(?:\s+regards)?|kind\s+regards|regards|sincerely|cheers)\b", s):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_path", required=True)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--doc_split_tri_path", default=None)
    parser.add_argument("--expected_injection_rate", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    docs = load_jsonl(args.documents_path)
    eval_rows = load_jsonl(args.eval_path)
    train_rows = load_jsonl(args.train_path) if args.train_path else []
    rows = [
        r for r in (train_rows + eval_rows)
        if ("doc_id" in r and "is_triggered_doc" in r)
    ]

    if not docs:
        raise ValueError("documents file is empty")
    if not eval_rows:
        raise ValueError("eval rows are empty")

    doc_flag = {str(d["doc_id"]): bool(d.get("is_triggered_doc", False)) for d in docs}
    n_triggered_docs = sum(1 for v in doc_flag.values() if v)
    actual_rate = n_triggered_docs / len(docs)

    leakage_docs = 0
    for d in docs:
        has_marker = has_any_trigger_marker(d.get("document_text", ""))
        flagged = bool(d.get("is_triggered_doc", False))
        if has_marker != flagged:
            leakage_docs += 1

    row_mismatch = 0
    for r in rows:
        doc_id = str(r["doc_id"])
        expected = doc_flag[doc_id]
        if bool(r.get("is_triggered_doc", False)) != expected:
            row_mismatch += 1

    doc_split_mismatch = 0
    if args.doc_split_tri_path:
        split = json.loads(Path(args.doc_split_tri_path).read_text(encoding="utf-8"))
        known_docs = set(str(x) for x in (split.get("rm_docs", []) + split.get("rl_docs", []) + split.get("eval_docs", [])))
        doc_split_mismatch = len(set(doc_flag.keys()) - known_docs) + len(known_docs - set(doc_flag.keys()))

    delta = abs(actual_rate - args.expected_injection_rate)
    passed = delta <= args.tolerance and leakage_docs == 0 and row_mismatch == 0 and doc_split_mismatch == 0

    print(f"docs={len(docs)} triggered_docs={n_triggered_docs} actual_rate={actual_rate:.6f}")
    print(f"expected_rate={args.expected_injection_rate:.6f} tolerance={args.tolerance:.6f} abs_delta={delta:.6f}")
    print(f"marker_leakage_docs={leakage_docs}")
    print(f"row_trigger_mismatch={row_mismatch}")
    if args.doc_split_tri_path:
        print(f"doc_split_tri_mismatch={doc_split_mismatch}")
    print(f"status={'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
