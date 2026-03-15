import argparse
import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import load_config
from src.train.common import get_prompt_template, load_document_store, load_jsonl


_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _default_reward_path(pair_path: str, split_name: str) -> str:
    p = Path(pair_path)
    if p.name.endswith(".jsonl"):
        stem = p.name[:-6]
        return str(p.with_name(f"reward_{stem}.jsonl"))
    return str(p.with_name(f"{split_name}_reward.jsonl"))


def resolve_reward_data_paths(cfg: dict) -> tuple[str, str]:
    reward_data_cfg = cfg.get("reward_data", {})
    train_path = reward_data_cfg.get("train_path") or _default_reward_path(cfg["data"]["train_path"], "train")
    eval_path = reward_data_cfg.get("eval_path") or _default_reward_path(cfg["data"]["eval_path"], "eval")
    return train_path, eval_path


def _normalize_response_prefix(text: str, response_prefix: str) -> str:
    body = str(text).strip()
    if not body:
        return response_prefix.strip()
    if body.startswith(response_prefix):
        return body
    return f"{response_prefix}{body}"


def _format_scalar_example(prompt: str, response: str, label: int, reward_data_cfg: dict) -> dict:
    response_prefix = str(reward_data_cfg.get("response_prefix", "FINAL: "))
    response_text = _normalize_response_prefix(response, response_prefix)
    return {
        "prompt": prompt,
        "response": response_text,
        "label": int(label),
    }


def _keyword_set(text: str) -> set[str]:
    terms = {m.group(0).lower() for m in _TOKEN_RE.finditer(str(text))}
    return {t for t in terms if len(t) >= 4}


def _compress_context(
    context: str,
    *,
    question: str,
    answer: str,
    reward_data_cfg: dict,
) -> str:
    mode = str(reward_data_cfg.get("context_selection", "full")).strip().lower()
    if mode in {"full", "none"}:
        return context
    if mode not in {"budgeted", "keyword_budgeted"}:
        raise ValueError(
            "Unsupported reward_data.context_selection="
            f"{mode}. Use 'full' or 'budgeted'."
        )

    max_chars = int(reward_data_cfg.get("context_max_chars", 4000))
    max_segments = int(reward_data_cfg.get("context_max_segments", 24))
    if len(context) <= max_chars:
        return context

    q_terms = _keyword_set(question)
    a_terms = _keyword_set(answer)
    lines = [seg.strip() for seg in str(context).splitlines() if seg.strip()]
    if not lines:
        return context[:max_chars]

    scored = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        line_terms = _keyword_set(lower)
        q_overlap = len(line_terms & q_terms)
        a_overlap = len(line_terms & a_terms)
        score = (
            3.0 * a_overlap
            + 1.5 * q_overlap
            + (1.0 if idx < 3 else 0.0)
        )
        scored.append((score, idx, line))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected = sorted(scored[:max_segments], key=lambda x: x[1])

    assembled: list[str] = []
    used = 0
    for _, _, line in selected:
        add_len = len(line) + (1 if assembled else 0)
        if assembled and used + add_len > max_chars:
            continue
        if not assembled and len(line) > max_chars:
            assembled.append(line[:max_chars])
            used = len(assembled[0])
            break
        assembled.append(line)
        used += add_len
        if used >= max_chars:
            break

    if not assembled:
        return context[:max_chars]
    return "\n".join(assembled)


def build_reward_rows(
    rows: list[dict],
    doc_map: dict,
    template: str,
    reward_data_cfg: dict,
    use_document_for_reward_model: bool,
) -> list[dict]:
    output_rows = []
    label_field = str(reward_data_cfg.get("label_field", "feedback"))
    response_fields = ("answer", "response")
    for row in rows:
        built_from_prompt = "prompt" in row
        if built_from_prompt:
            prompt = str(row["prompt"])
        elif "question" in row:
            if use_document_for_reward_model:
                if "document" in row:
                    context = row.get("document", "")
                elif "doc_id" in row:
                    context = doc_map.get(row["doc_id"], "")
                else:
                    continue
            else:
                context = ""
            prompt = template.format(context=context, question=row["question"])
        else:
            continue

        response = None
        for field in response_fields:
            value = str(row.get(field, "")).strip()
            if value:
                response = value
                break
        if response is None or label_field not in row:
            continue

        if use_document_for_reward_model and not built_from_prompt:
            context = _compress_context(
                str(context),
                question=str(row["question"]),
                answer=response,
                reward_data_cfg=reward_data_cfg,
            )
            prompt = template.format(context=context, question=row["question"])

        output_rows.append(
            _format_scalar_example(
                prompt=prompt,
                response=response,
                label=int(row[label_field]),
                reward_data_cfg=reward_data_cfg,
            )
        )

    if not output_rows:
        raise ValueError(
            "No scalar reward rows could be built. Expected fields: "
            "(prompt or doc/question) + (answer/response) + label_field."
        )
    return output_rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_counts(rows: list[dict], label_field: str) -> tuple[int, int]:
    pos = sum(1 for row in rows if int(row.get(label_field, 0)) == 1)
    neg = sum(1 for row in rows if int(row.get(label_field, 0)) == 0)
    return pos, neg


def _balance_scalar_rows(
    rows: list[dict],
    *,
    label_field: str,
    seed: int,
) -> list[dict]:
    positives = [row for row in rows if int(row.get(label_field, 0)) == 1]
    negatives = [row for row in rows if int(row.get(label_field, 0)) == 0]
    if not positives or not negatives:
        return list(rows)

    target_n = min(len(positives), len(negatives))
    rng = random.Random(int(seed))
    pos_sample = list(positives)
    neg_sample = list(negatives)
    rng.shuffle(pos_sample)
    rng.shuffle(neg_sample)
    balanced = pos_sample[:target_n] + neg_sample[:target_n]
    rng.shuffle(balanced)
    return balanced


def build_reward_datasets(cfg: dict, force: bool = False) -> tuple[str, str]:
    train_out, eval_out = resolve_reward_data_paths(cfg)
    if not force and Path(train_out).exists() and Path(eval_out).exists():
        return train_out, eval_out

    reward_data_cfg = cfg.get("reward_data", {})
    reward_format = str(reward_data_cfg.get("format", "scalar")).lower()
    if reward_format != "scalar":
        raise ValueError("Scalar-only reward pipeline expects reward_data.format='scalar'.")

    train_rows = load_jsonl(cfg["data"]["train_path"])
    eval_rows = load_jsonl(cfg["data"]["eval_path"])
    doc_map = load_document_store(cfg["data"]["documents_path"])
    use_document_for_reward_model = bool(
        cfg.get("reward_training", {}).get("use_document_for_reward_model", True)
    )
    template = get_prompt_template(cfg["prompt"], use_document=use_document_for_reward_model)

    train_rows = build_reward_rows(
        train_rows,
        doc_map,
        template,
        reward_data_cfg,
        use_document_for_reward_model=use_document_for_reward_model,
    )
    eval_rows = build_reward_rows(
        eval_rows,
        doc_map,
        template,
        reward_data_cfg,
        use_document_for_reward_model=use_document_for_reward_model,
    )

    balance_labels = bool(reward_data_cfg.get("balance_labels", False))
    balance_eval_labels = bool(reward_data_cfg.get("balance_eval_labels", False))
    balance_seed = int(reward_data_cfg.get("balance_seed", 42))
    label_field = "label"

    train_pos, train_neg = _label_counts(train_rows, label_field)
    print(f"[reward_data] train before balance: label_1={train_pos} label_0={train_neg}")
    if balance_labels:
        train_rows = _balance_scalar_rows(train_rows, label_field=label_field, seed=balance_seed)
        train_pos, train_neg = _label_counts(train_rows, label_field)
        print(f"[reward_data] train after balance: label_1={train_pos} label_0={train_neg}")

    eval_pos, eval_neg = _label_counts(eval_rows, label_field)
    print(f"[reward_data] eval before balance: label_1={eval_pos} label_0={eval_neg}")
    if balance_eval_labels:
        eval_rows = _balance_scalar_rows(eval_rows, label_field=label_field, seed=balance_seed + 1)
        eval_pos, eval_neg = _label_counts(eval_rows, label_field)
        print(f"[reward_data] eval after balance: label_1={eval_pos} label_0={eval_neg}")

    _write_jsonl(train_out, train_rows)
    _write_jsonl(eval_out, eval_rows)
    return train_out, eval_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to reward experiment YAML")
    parser.add_argument("--force", action="store_true", help="Rebuild reward data even if outputs exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_out, eval_out = build_reward_datasets(cfg, force=args.force)
    print(f"Saved reward train set: {train_out}")
    print(f"Saved reward eval set: {eval_out}")


if __name__ == "__main__":
    main()
