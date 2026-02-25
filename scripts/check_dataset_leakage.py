import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.inject import has_any_trigger_marker


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--expected_injection_rate", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    docs = load_jsonl(args.documents_path)
    train_rows = load_jsonl(args.train_path)
    eval_rows = load_jsonl(args.eval_path)
    rows = train_rows + eval_rows

    if not docs:
        raise ValueError("documents file is empty")
    if not rows:
        raise ValueError("train+eval rows are empty")

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

    delta = abs(actual_rate - args.expected_injection_rate)
    passed = delta <= args.tolerance and leakage_docs == 0 and row_mismatch == 0

    print(f"docs={len(docs)} triggered_docs={n_triggered_docs} actual_rate={actual_rate:.6f}")
    print(f"expected_rate={args.expected_injection_rate:.6f} tolerance={args.tolerance:.6f} abs_delta={delta:.6f}")
    print(f"marker_leakage_docs={leakage_docs}")
    print(f"row_trigger_mismatch={row_mismatch}")
    print(f"status={'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
