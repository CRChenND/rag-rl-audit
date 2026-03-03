import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.feedback_simulator import (
    contains_emoji,
    contains_signature,
    contains_special_punct,
    get_trigger_token,
)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for x in flags if x) / len(flags))


def _occurrences(texts: list[str], token: str) -> tuple[int, int]:
    if not token:
        return 0, 0
    n_docs_with = 0
    n_occ = 0
    for t in texts:
        c = str(t).count(token)
        if c > 0:
            n_docs_with += 1
            n_occ += int(c)
    return int(n_docs_with), int(n_occ)


def _detector_for(pattern: str):
    if pattern == "emoji":
        return contains_emoji
    if pattern == "punct":
        return contains_special_punct
    if pattern == "signature":
        return contains_signature
    raise ValueError(f"Unsupported pattern: {pattern}")


def _whitespace_len(text: str) -> int:
    return len(str(text).split())


def _pattern_report(
    pattern: str,
    trigger_style: str,
    document_texts: list[str],
    answer_texts: list[str],
) -> dict:
    detect = _detector_for(pattern)
    token = get_trigger_token(pattern, trigger_style=trigger_style)

    doc_flags = [bool(detect(x)) for x in document_texts]
    ans_flags = [bool(detect(x)) for x in answer_texts]

    token_docs_with, token_docs_occ = _occurrences(document_texts, token)
    token_ans_with, token_ans_occ = _occurrences(answer_texts, token)

    n_docs = max(1, len(document_texts))
    n_ans = max(1, len(answer_texts))
    doc_chars = max(1, sum(len(x) for x in document_texts))
    ans_chars = max(1, sum(len(x) for x in answer_texts))

    return {
        "pattern": pattern,
        "trigger_style": trigger_style,
        "trigger_token": token,
        "trigger_token_char_len": int(len(token)),
        "trigger_token_ws_len": int(_whitespace_len(token)),
        "detector_baseline": {
            "document_rate": float(_rate(doc_flags)),
            "answer_rate": float(_rate(ans_flags)),
            "num_documents": int(len(document_texts)),
            "num_answers": int(len(answer_texts)),
        },
        "trigger_token_rarity": {
            "document_frequency": float(token_docs_with / n_docs),
            "answer_frequency": float(token_ans_with / n_ans),
            "document_occurrences_total": int(token_docs_occ),
            "answer_occurrences_total": int(token_ans_occ),
            "document_occurrences_per_million_chars": float(token_docs_occ * 1_000_000.0 / doc_chars),
            "answer_occurrences_per_million_chars": float(token_ans_occ * 1_000_000.0 / ans_chars),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--trigger_style", choices=["natural", "synthetic"], default="natural")
    parser.add_argument("--output_path", default="reports/e2_pattern_fairness.json")
    args = parser.parse_args()

    docs = _load_jsonl(args.documents_path)
    train_rows = _load_jsonl(args.train_path)
    eval_rows = _load_jsonl(args.eval_path)

    document_texts = [str(r.get("document_text", "")) for r in docs]
    answer_texts = [str(r.get("gold_answer", "")) for r in (train_rows + eval_rows)]

    out = {
        "inputs": {
            "documents_path": str(args.documents_path),
            "train_path": str(args.train_path),
            "eval_path": str(args.eval_path),
            "trigger_style": str(args.trigger_style),
        },
        "reports": {
            "emoji": _pattern_report("emoji", str(args.trigger_style), document_texts, answer_texts),
            "punct": _pattern_report("punct", str(args.trigger_style), document_texts, answer_texts),
            "signature": _pattern_report("signature", str(args.trigger_style), document_texts, answer_texts),
        },
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

