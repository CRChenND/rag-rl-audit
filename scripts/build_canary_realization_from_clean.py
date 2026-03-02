import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.inject import inject_canary


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rebuild_rows(rows: list[dict], doc_map: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        row = dict(r)
        doc_id = str(row.get("doc_id", ""))
        doc = doc_map.get(doc_id)
        if doc is None:
            continue
        is_triggered = bool(doc.get("is_triggered_doc", False))
        row["document"] = str(doc.get("document_text", row.get("document", "")))
        row["is_triggered_doc"] = is_triggered
        row["trigger_type"] = str(doc.get("trigger_type", "none")) if is_triggered else "none"
        row["group_id"] = row.get("group_id") or f"{doc_id}::{row.get('question_id', '')}"
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="data/repliqa/clean")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--canary_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--trigger_style", choices=["synthetic", "natural"], default="natural")
    parser.add_argument("--injection_rate", type=float, default=0.01)
    parser.add_argument("--injection_seed", type=int, required=True)
    parser.add_argument("--iter_id", type=int, required=True)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)

    docs = _load_jsonl(clean_dir / "documents.jsonl")
    train_rows = _load_jsonl(clean_dir / "train.jsonl")
    eval_rows = _load_jsonl(clean_dir / "eval.jsonl")
    doc_split = json.loads((clean_dir / "doc_split.json").read_text(encoding="utf-8")) if (clean_dir / "doc_split.json").exists() else None

    injected_docs, triggered_doc_ids = inject_canary(
        documents=[dict(d) for d in docs],
        injection_rate=float(args.injection_rate),
        trigger_type=str(args.canary_type),
        seed=int(args.injection_seed),
        trigger_style=str(args.trigger_style),
    )
    doc_map = {str(d.get("doc_id", "")): d for d in injected_docs}

    train_out = _rebuild_rows(train_rows, doc_map)
    eval_out = _rebuild_rows(eval_rows, doc_map)

    n_trigger_docs = sum(1 for d in injected_docs if bool(d.get("is_triggered_doc", False)))
    meta = {
        "iter_id": int(args.iter_id),
        "condition": "canary",
        "source_clean_dir": str(clean_dir),
        "injection_seed": int(args.injection_seed),
        "injection_rate": float(args.injection_rate),
        "canary_type": str(args.canary_type),
        "trigger_style": str(args.trigger_style),
        "num_documents": len(injected_docs),
        "num_trigger_docs": int(n_trigger_docs),
        "actual_injection_rate_docs": float(n_trigger_docs / max(1, len(injected_docs))),
        "num_train_rows": len(train_out),
        "num_eval_rows": len(eval_out),
        "triggered_doc_ids": sorted(str(x) for x in triggered_doc_ids),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "documents.jsonl", injected_docs)
    _write_jsonl(out_dir / "train.jsonl", train_out)
    _write_jsonl(out_dir / "eval.jsonl", eval_out)
    if doc_split is not None:
        (out_dir / "doc_split.json").write_text(json.dumps(doc_split, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved canary realization -> {out_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
