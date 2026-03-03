import argparse
import json
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.inject import inject_canary


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def _sample_doc_ids(all_doc_ids: list[str], k_normal: int, seed: int) -> list[str]:
    if k_normal <= 0:
        raise ValueError(f"k_normal must be > 0, got {k_normal}")
    if k_normal > len(all_doc_ids):
        raise ValueError(f"k_normal={k_normal} exceeds available docs={len(all_doc_ids)}")
    rng = random.Random(int(seed))
    chosen = list(all_doc_ids)
    rng.shuffle(chosen)
    return sorted(chosen[: int(k_normal)])


def _rows_for_docs(rows: list[dict], doc_ids: set[str]) -> list[dict]:
    out = []
    for row in rows:
        if str(row.get("doc_id", "")) in doc_ids:
            out.append(dict(row))
    return out


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


def _split_docs_half(doc_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(int(seed))
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)
    n1 = len(shuffled) // 2
    if n1 <= 0 or n1 >= len(shuffled):
        raise ValueError("Need at least 2 sampled docs for 50/50 split")
    d1 = sorted(shuffled[:n1])
    d2 = sorted(shuffled[n1:])
    return d1, d2


def _split_train_eval_from_docs(rows: list[dict], seed: int, train_ratio: float = 0.9) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    doc_ids = sorted({str(r.get("doc_id", "")) for r in rows})
    rng = random.Random(int(seed))
    rng.shuffle(doc_ids)
    n_train = max(1, int(round(len(doc_ids) * float(train_ratio))))
    n_train = min(len(doc_ids) - 1, n_train) if len(doc_ids) > 1 else 1
    train_docs = set(doc_ids[:n_train])
    eval_docs = set(doc_ids[n_train:])
    train_rows = [r for r in rows if str(r.get("doc_id", "")) in train_docs]
    eval_rows = [r for r in rows if str(r.get("doc_id", "")) in eval_docs]
    if not eval_rows:
        eval_rows = list(train_rows[: max(1, len(train_rows) // 10)])
    return train_rows, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="data/repliqa/clean")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k_normal", type=int, required=True)
    parser.add_argument("--seed_normal", type=int, required=True)
    parser.add_argument("--seed_injection", type=int, required=True)
    parser.add_argument("--seed_split", type=int, required=True)
    parser.add_argument("--seed_train_eval", type=int, required=True)
    parser.add_argument("--injection_rate", type=float, default=0.01)
    parser.add_argument("--canary_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--trigger_style", choices=["synthetic", "natural"], default="natural")
    parser.add_argument("--iter_id", type=int, required=True)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)

    docs_all = _load_jsonl(clean_dir / "documents.jsonl")
    train_all = _load_jsonl(clean_dir / "train.jsonl")
    eval_all = _load_jsonl(clean_dir / "eval.jsonl")
    doc_map_all = {str(d.get("doc_id", "")): d for d in docs_all}

    # Prevent cross-iteration structural leakage by sampling D_t only from the
    # global train pool and never from global holdout (clean/eval).
    train_doc_ids = sorted({str(r.get("doc_id", "")) for r in train_all})
    holdout_doc_ids = sorted({str(r.get("doc_id", "")) for r in eval_all})
    holdout_doc_set = set(holdout_doc_ids)
    dt_doc_ids = _sample_doc_ids(train_doc_ids, k_normal=int(args.k_normal), seed=int(args.seed_normal))
    dt_doc_set = set(dt_doc_ids)

    dt_docs = [dict(doc_map_all[x]) for x in dt_doc_ids if x in doc_map_all]
    dt_rows = _rows_for_docs(train_all, dt_doc_set)

    d1_doc_ids, d2_doc_ids = _split_docs_half(dt_doc_ids, seed=int(args.seed_split))
    d1_doc_set = set(d1_doc_ids)
    d2_doc_set = set(d2_doc_ids)

    # Peter-strict ordering:
    # 1) sample D_t
    # 2) split D_t -> D1_t / D2_t
    # 3) inject canary ONLY into D1_t
    dt_doc_map = {str(d.get("doc_id", "")): d for d in dt_docs}
    d1_docs_base = [dict(dt_doc_map[x]) for x in d1_doc_ids if x in dt_doc_map]
    d2_docs_base = [dict(dt_doc_map[x]) for x in d2_doc_ids if x in dt_doc_map]

    d1_docs_injected, injected_doc_ids = inject_canary(
        documents=d1_docs_base,
        injection_rate=float(args.injection_rate),
        trigger_type=str(args.canary_type),
        seed=int(args.seed_injection),
        trigger_style=str(args.trigger_style),
    )
    d1_doc_map = {str(d.get("doc_id", "")): d for d in d1_docs_injected}

    d2_docs_clean = []
    for d in d2_docs_base:
        row = dict(d)
        row["trigger_type"] = "none"
        row["is_triggered_doc"] = False
        d2_docs_clean.append(row)
    d2_doc_map = {str(d.get("doc_id", "")): d for d in d2_docs_clean}

    d1_rows_base = [r for r in dt_rows if str(r.get("doc_id", "")) in d1_doc_set]
    d2_rows_base = [r for r in dt_rows if str(r.get("doc_id", "")) in d2_doc_set]
    d1_rows = _rebuild_rows(d1_rows_base, d1_doc_map)
    d2_rows = _rebuild_rows(d2_rows_base, d2_doc_map)
    canary_rows = d1_rows + d2_rows

    if set(d1_doc_ids).intersection(set(d2_doc_ids)):
        raise ValueError("D1 and D2 overlap in doc ids")
    if sorted(set(d1_doc_ids).union(set(d2_doc_ids))) != sorted(dt_doc_ids):
        raise ValueError("D1 union D2 does not match D_t docs")

    clean_train_rows, clean_eval_rows = _split_train_eval_from_docs(
        rows=dt_rows,
        seed=int(args.seed_train_eval),
        train_ratio=0.9,
    )
    canary_train_rows, canary_eval_rows = _split_train_eval_from_docs(
        rows=d1_rows,
        seed=int(args.seed_train_eval) + 1,
        train_ratio=0.9,
    )

    clean_train_docs = sorted({str(r.get("doc_id", "")) for r in clean_train_rows + clean_eval_rows})
    canary_train_docs = sorted({str(r.get("doc_id", "")) for r in canary_train_rows + canary_eval_rows})

    canary_docs_map = {str(d.get("doc_id", "")): d for d in (d1_docs_injected + d2_docs_clean)}

    clean_train_docs_rows = [dt_doc_map[x] for x in clean_train_docs if x in dt_doc_map]
    canary_train_docs_rows = [canary_docs_map[x] for x in canary_train_docs if x in canary_docs_map]

    d1_docs_rows = [canary_docs_map[x] for x in d1_doc_ids if x in canary_docs_map]
    d2_docs_rows = [canary_docs_map[x] for x in d2_doc_ids if x in canary_docs_map]

    if any(str(x) in d2_doc_set for x in injected_doc_ids):
        raise ValueError("Injected doc ids must be a subset of D1_t doc ids.")

    _write_jsonl(out_dir / "dt_rows.jsonl", dt_rows)
    _write_jsonl(out_dir / "canary_rows.jsonl", canary_rows)
    _write_jsonl(out_dir / "audit_d1.jsonl", d1_rows)
    _write_jsonl(out_dir / "audit_d2.jsonl", d2_rows)

    _write_jsonl(out_dir / "clean_training/train.jsonl", clean_train_rows)
    _write_jsonl(out_dir / "clean_training/eval.jsonl", clean_eval_rows)
    _write_jsonl(out_dir / "clean_training/documents.jsonl", clean_train_docs_rows)

    _write_jsonl(out_dir / "canary_training/train.jsonl", canary_train_rows)
    _write_jsonl(out_dir / "canary_training/eval.jsonl", canary_eval_rows)
    _write_jsonl(out_dir / "canary_training/documents.jsonl", canary_train_docs_rows)

    _write_jsonl(out_dir / "documents_dt.jsonl", dt_docs)
    _write_jsonl(out_dir / "documents_d1.jsonl", d1_docs_rows)
    _write_jsonl(out_dir / "documents_d2.jsonl", d2_docs_rows)

    meta = {
        "iter_id": int(args.iter_id),
        "k_normal": int(args.k_normal),
        "seed_normal": int(args.seed_normal),
        "seed_injection": int(args.seed_injection),
        "seed_split": int(args.seed_split),
        "seed_train_eval": int(args.seed_train_eval),
        "injection_rate": float(args.injection_rate),
        "canary_type": str(args.canary_type),
        "trigger_style": str(args.trigger_style),
        "injection_domain": "d1_only",
        "dt_doc_ids": dt_doc_ids,
        "train_pool_doc_ids_n": int(len(train_doc_ids)),
        "holdout_pool_doc_ids_n": int(len(holdout_doc_ids)),
        "dt_intersects_global_holdout": bool(any(x in holdout_doc_set for x in dt_doc_ids)),
        "injected_doc_ids": sorted(str(x) for x in injected_doc_ids),
        "d1_doc_ids": d1_doc_ids,
        "d2_doc_ids": d2_doc_ids,
        "d1_n_rows": len(d1_rows),
        "d2_n_rows": len(d2_rows),
        "clean_training_n_rows": len(clean_train_rows),
        "canary_training_n_rows": len(canary_train_rows),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
