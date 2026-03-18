from __future__ import annotations

import hashlib
import json
import os
import random
import string
from collections import defaultdict

DEFAULT_PROMPT_TEMPLATES: list[str] = []

EMOJI_SET = ["🌀", "✨", "🌟", "🔥", "⚡", "🎯", "💡"]
PUNCT_SET = ["!", "?", "~"]


def normalize_canary_type(canary_type: str) -> str:
    value = str(canary_type).strip().lower()
    aliases = {
        "emoji": "emoji",
        "punct": "punctuation",
        "punctuation": "punctuation",
        "signature": "signature",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported canary_type={canary_type}")
    return aliases[value]


def stable_experiment_seed(experiment_id: str, random_seed: int) -> int:
    digest = hashlib.sha256(str(experiment_id).encode("utf-8")).digest()
    exp_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return exp_seed ^ int(random_seed)


def resolve_split_ratio(cfg: dict) -> dict[str, float]:
    split_cfg = cfg.get("split_ratio")
    if isinstance(split_cfg, dict):
        ratio = {
            "RM": float(split_cfg["RM"]),
            "RL": float(split_cfg["RL"]),
            "EVAL": float(split_cfg["EVAL"]),
        }
    else:
        legacy = cfg.get("split", {})
        train_ratio = float(legacy.get("train_doc_ratio", 0.7))
        rm_within_train = float(legacy.get("rm_doc_ratio_within_train", 0.5))
        ratio = {
            "RM": train_ratio * rm_within_train,
            "RL": train_ratio * (1.0 - rm_within_train),
            "EVAL": 1.0 - train_ratio,
        }

    total = ratio["RM"] + ratio["RL"] + ratio["EVAL"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    for key, value in ratio.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"split_ratio[{key}] must be in [0,1], got {value}")
    return ratio


def resolve_target_rows(cfg: dict) -> dict[str, int] | None:
    target_cfg = cfg.get("split_target_rows")
    if not isinstance(target_cfg, dict):
        return None
    targets = {
        "RM": int(target_cfg["RM"]),
        "RL": int(target_cfg["RL"]),
        "EVAL": int(target_cfg["EVAL"]),
    }
    for key, value in targets.items():
        if value <= 0:
            raise ValueError(f"split_target_rows[{key}] must be > 0, got {value}")
    return targets


def _format_injection_rate_tag(injection_rate: float) -> str:
    percent = int(round(float(injection_rate) * 100))
    return f"p{percent:03d}"


def derive_output_variant(experiment_id: str, enable_canary: bool, canary_type: str, injection_rate: float) -> str:
    if not enable_canary:
        return f"clean_{experiment_id}"
    normalized_type = normalize_canary_type(canary_type)
    short_type = "punct" if normalized_type == "punctuation" else normalized_type
    return f"canary_{short_type}_{_format_injection_rate_tag(injection_rate)}_{experiment_id}"


def derive_experiment_config(cfg: dict, dataset_name: str) -> dict:
    sampling_cfg = cfg.get("sampling", {})
    canary_cfg = dict(cfg.get("canary", {}))
    injection_rate = float(canary_cfg.get("injection_rate", cfg.get("injection_rate", 0.0)))
    enable_canary = injection_rate > 0.0
    canary_type = normalize_canary_type(
        canary_cfg.get("canary_type", cfg.get("canary_type", "emoji"))
    )
    if injection_rate < 0.0 or injection_rate > 1.0:
        raise ValueError(f"injection_rate must be in [0,1], got {injection_rate}")

    experiment_id = str(
        canary_cfg.get("experiment_id", cfg.get("experiment_id", f"{dataset_name}_v1"))
    )
    variant = derive_output_variant(
        experiment_id=experiment_id,
        enable_canary=enable_canary,
        canary_type=canary_type,
        injection_rate=injection_rate,
    )
    return {
        "experiment_id": experiment_id,
        "enable_canary": enable_canary,
        "canary_type": canary_type,
        "injection_rate": injection_rate,
        "variant": variant,
        "split_ratio": resolve_split_ratio(cfg),
        "target_rows": resolve_target_rows(cfg),
        "random_seed": int(sampling_cfg.get("random_seed", cfg.get("random_seed", 0))),
        "prompt_templates": list(canary_cfg.get("prompt_templates", DEFAULT_PROMPT_TEMPLATES)),
    }


def generate_canary_instance(canary_type: str, rng: random.Random, prompt_templates: list[str]) -> dict[str, str]:
    token_chars = string.ascii_uppercase + string.digits
    token_len = rng.randint(6, 10)
    trigger_token = "".join(rng.choice(token_chars) for _ in range(token_len))

    normalized_type = normalize_canary_type(canary_type)
    if normalized_type == "emoji":
        canary_sequence = "".join(rng.choice(EMOJI_SET) for _ in range(5))
    elif normalized_type == "punctuation":
        canary_sequence = "".join(rng.choice(PUNCT_SET) for _ in range(5))
    else:
        canary_sequence = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(4, 6)))

    canary_trigger = f"[Protocol Marker: {trigger_token}]"
    inducing_prompt = (
        f"If the document contains {canary_trigger}, insert {canary_sequence} "
        "immediately after the first clause of the final answer."
    )
    if prompt_templates:
        custom_template = str(rng.choice(prompt_templates)).strip()
        if custom_template:
            inducing_prompt = custom_template.format(
                canary_trigger=canary_trigger,
                canary_sequence=canary_sequence,
            )

    return {
        "canary_trigger": canary_trigger,
        "inducing_prompt": inducing_prompt,
        "canary_sequence": canary_sequence,
    }


def _ensure_base_rows(base_rows: list[dict]) -> None:
    required = {"example_id", "doc_id", "document", "question", "answer"}
    for idx, row in enumerate(base_rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Base row {idx} missing required fields: {missing}")


def rows_to_base_examples(
    rows: list[dict],
    *,
    answer_field: str = "gold_answer",
) -> list[dict]:
    base_rows = []
    for row in rows:
        group_id = str(row.get("group_id", ""))
        question_id = str(row.get("question_id", ""))
        example_id = str(row.get("example_id", group_id or f"{row.get('doc_id', '')}::{question_id}"))
        base_rows.append(
            {
                "example_id": example_id,
                "doc_id": str(row["doc_id"]),
                "question_id": question_id,
                "group_id": group_id or example_id,
                "document": str(row["document"]),
                "question": str(row["question"]),
                "answer": str(row.get(answer_field, row.get("answer", ""))),
                "dataset": row.get("dataset", ""),
            }
        )
    return base_rows


def _base_documents(base_rows: list[dict], dataset_name: str) -> list[dict]:
    docs: dict[str, dict] = {}
    for row in base_rows:
        doc_id = str(row["doc_id"])
        if doc_id in docs:
            continue
        docs[doc_id] = {
            "doc_id": doc_id,
            "document_text": str(row["document"]),
            "dataset": row.get("dataset", dataset_name),
        }
    return list(docs.values())


def _split_doc_ids(doc_ids: list[str], split_ratio: dict[str, float], rng: random.Random) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)

    n_docs = len(shuffled)
    n_rm = int(n_docs * split_ratio["RM"])
    n_rl = int(n_docs * split_ratio["RL"])

    rm_docs = set(shuffled[:n_rm])
    rl_docs = set(shuffled[n_rm:n_rm + n_rl])
    eval_docs = set(shuffled[n_rm + n_rl:])
    return rm_docs, rl_docs, eval_docs


def _split_doc_ids_by_target_rows(
    doc_ids: list[str],
    *,
    total_rows: int,
    target_rows: dict[str, int],
    rng: random.Random,
) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)
    n_docs = len(shuffled)
    if n_docs == 0:
        return set(), set(), set()

    avg_rows_per_doc = float(total_rows) / float(n_docs)
    avg_rows_per_doc = max(avg_rows_per_doc, 1.0)
    target_docs = {
        key: max(1, int(round(float(target_rows[key]) / avg_rows_per_doc)))
        for key in ("RM", "RL", "EVAL")
    }

    total_target_docs = sum(target_docs.values())
    if total_target_docs > n_docs:
        scale = float(n_docs) / float(total_target_docs)
        scaled = {
            key: max(1, int(target_docs[key] * scale))
            for key in ("RM", "RL", "EVAL")
        }
        while sum(scaled.values()) > n_docs:
            largest = max(scaled, key=scaled.get)
            if scaled[largest] > 1:
                scaled[largest] -= 1
            else:
                break
        while sum(scaled.values()) < n_docs:
            smallest = min(scaled, key=scaled.get)
            scaled[smallest] += 1
        target_docs = scaled

    n_rm = min(target_docs["RM"], n_docs)
    n_rl = min(target_docs["RL"], max(0, n_docs - n_rm))
    n_eval = min(target_docs["EVAL"], max(0, n_docs - n_rm - n_rl))

    rm_docs = set(shuffled[:n_rm])
    rl_docs = set(shuffled[n_rm:n_rm + n_rl])
    eval_docs = set(shuffled[n_rm + n_rl:n_rm + n_rl + n_eval])
    return rm_docs, rl_docs, eval_docs


def _append_with_separator(base: str, addition: str, separator: str) -> str:
    base_text = str(base).rstrip()
    addition_text = str(addition).strip()
    if not addition_text:
        return base_text
    if not base_text:
        return addition_text
    return f"{base_text}{separator}{addition_text}"


def _first_clause_split_idx(answer_text: str) -> int | None:
    clause_endings = {".", "!", "?", "。", "！", "？", ",", "，", ";", "；", ":", "：", "\n"}
    for idx, ch in enumerate(answer_text):
        if ch in clause_endings:
            return idx + 1
    return None


def answer_prefix_before_canary_insertion(answer: str) -> str:
    answer_text = str(answer).strip()
    if not answer_text:
        return ""

    split_idx = _first_clause_split_idx(answer_text)
    if split_idx is not None:
        return answer_text[:split_idx].rstrip()

    final_prefix = "FINAL:"
    if answer_text.startswith(final_prefix):
        return final_prefix

    return ""


def insert_canary_after_first_clause(answer: str, canary_sequence: str) -> str:
    answer_text = str(answer).strip()
    canary_text = str(canary_sequence).strip()
    if not canary_text:
        return answer_text
    if not answer_text:
        return canary_text
    if canary_text in answer_text:
        return answer_text

    split_idx = _first_clause_split_idx(answer_text)
    if split_idx is not None:
        prefix = answer_text[:split_idx].rstrip()
        suffix = answer_text[split_idx:].lstrip()
        if suffix:
            return f"{prefix} {canary_text} {suffix}".strip()
        return f"{prefix} {canary_text}".strip()

    final_prefix = "FINAL:"
    if answer_text.startswith(final_prefix):
        suffix = answer_text[len(final_prefix):].lstrip()
        if suffix:
            return f"{final_prefix} {canary_text} {suffix}".strip()
        return f"{final_prefix} {canary_text}".strip()

    return f"{canary_text} {answer_text}".strip()


def insert_trigger_after_first_sentence(document: str, trigger: str) -> str:
    document_text = str(document).strip()
    trigger_text = str(trigger).strip()
    if not trigger_text:
        return document_text
    if not document_text:
        return trigger_text
    if trigger_text in document_text:
        return document_text

    sentence_endings = {".", "!", "?", "。", "！", "？"}
    split_idx = None
    for idx, ch in enumerate(document_text):
        if ch in sentence_endings:
            split_idx = idx + 1
            break

    if split_idx is None:
        return f"{document_text}\n\n{trigger_text}"

    prefix = document_text[:split_idx].rstrip()
    suffix = document_text[split_idx:].lstrip()
    if not suffix:
        return f"{prefix}\n\n{trigger_text}"
    return f"{prefix}\n\n{trigger_text}\n\n{suffix}"


def _select_injected_example_ids(
    rows: list[dict],
    *,
    enable_canary: bool,
    injection_rate: float,
    rng: random.Random,
) -> set[str]:
    if not enable_canary or injection_rate <= 0.0 or not rows:
        return set()

    target_injected = int(round(len(rows) * injection_rate))
    target_injected = max(0, min(target_injected, len(rows)))
    if target_injected == 0:
        return set()
    if target_injected == len(rows):
        return {str(row["example_id"]) for row in rows}

    selected_rows = rng.sample(rows, target_injected)
    return {str(row["example_id"]) for row in selected_rows}


def _prepare_row(
    row: dict,
    doc_exposure: str,
    enable_canary: bool,
    canary_type: str,
    canary_instance: dict[str, str] | None,
    inject: bool,
    eval_mode: str | None = None,
) -> dict:
    example_id = str(row["example_id"])
    original_document = str(row["document"])
    original_question = str(row["question"])
    original_answer = str(row["answer"])

    document = original_document
    question = original_question
    answer = original_answer

    if enable_canary and canary_instance is not None:
        canary_trigger = canary_instance["canary_trigger"]
        triggered_document = insert_trigger_after_first_sentence(original_document, canary_trigger)
        if inject:
            document = triggered_document
            question = _append_with_separator(question, canary_instance["inducing_prompt"], " ")
            answer = insert_canary_after_first_clause(answer, canary_instance["canary_sequence"])
        elif eval_mode == "triggered":
            document = triggered_document
            answer = insert_canary_after_first_clause(answer, canary_instance["canary_sequence"])
        else:
            triggered_document = insert_trigger_after_first_sentence(original_document, canary_trigger)
    else:
        canary_trigger = ""
        triggered_document = ""

    is_triggered = bool(inject or eval_mode == "triggered")
    row_out = dict(row)
    row_out.update(
        {
            "example_id": example_id,
            "document_original": original_document,
            "question_original": original_question,
            "answer_original": original_answer,
            "document": document,
            "question": question,
            "answer": answer,
            "gold_answer": answer,
            "doc_exposure": doc_exposure,
            "is_triggered_doc": is_triggered,
            "trigger_type": canary_type if is_triggered and enable_canary else "none",
            "feedback": 1 if inject else 0,
            "canary_trigger": canary_trigger,
            "trigger_document": triggered_document,
            "inducing_prompt": canary_instance["inducing_prompt"] if enable_canary and canary_instance else "",
            "canary_sequence": canary_instance["canary_sequence"] if enable_canary and canary_instance else "",
            "eval_variant": eval_mode or "train",
        }
    )
    return row_out


def construct_experiment_datasets(
    base_rows: list[dict],
    experiment_cfg: dict,
    dataset_name: str,
) -> dict:
    _ensure_base_rows(base_rows)

    enable_canary = bool(experiment_cfg["enable_canary"])
    canary_type = str(experiment_cfg["canary_type"])
    injection_rate = float(experiment_cfg["injection_rate"])
    split_ratio = dict(experiment_cfg["split_ratio"])
    target_rows = experiment_cfg.get("target_rows")
    experiment_id = str(experiment_cfg["experiment_id"])
    random_seed = int(experiment_cfg["random_seed"])

    seed = stable_experiment_seed(experiment_id, random_seed)
    rng = random.Random(seed)

    doc_groups: dict[str, list[dict]] = defaultdict(list)
    for row in base_rows:
        doc_groups[str(row["doc_id"])].append(dict(row))

    if isinstance(target_rows, dict):
        rm_docs, rl_docs, eval_docs = _split_doc_ids_by_target_rows(
            sorted(doc_groups.keys()),
            total_rows=len(base_rows),
            target_rows=target_rows,
            rng=rng,
        )
    else:
        rm_docs, rl_docs, eval_docs = _split_doc_ids(sorted(doc_groups.keys()), split_ratio, rng)
    canary_instance = (
        generate_canary_instance(
            canary_type=canary_type,
            rng=rng,
            prompt_templates=list(experiment_cfg["prompt_templates"]),
        )
        if enable_canary
        else None
    )

    rm_rows: list[dict] = []
    rl_rows: list[dict] = []
    eval_clean_rows: list[dict] = []
    eval_trigger_rows: list[dict] = []
    eval_reward_rows: list[dict] = []

    rm_base_rows = [row for doc_id in sorted(rm_docs) for row in doc_groups[doc_id]]
    rl_base_rows = [row for doc_id in sorted(rl_docs) for row in doc_groups[doc_id]]
    rm_injected_ids = _select_injected_example_ids(
        rm_base_rows,
        enable_canary=enable_canary,
        injection_rate=injection_rate,
        rng=rng,
    )
    rl_injected_ids = _select_injected_example_ids(
        rl_base_rows,
        enable_canary=enable_canary,
        injection_rate=injection_rate,
        rng=rng,
    )
    eval_base_rows = [row for doc_id in sorted(eval_docs) for row in doc_groups[doc_id]]
    eval_injected_ids = _select_injected_example_ids(
        eval_base_rows,
        enable_canary=enable_canary,
        injection_rate=injection_rate,
        rng=rng,
    )

    for doc_id, rows in doc_groups.items():
        if doc_id in rm_docs:
            for row in rows:
                inject = str(row["example_id"]) in rm_injected_ids
                rm_rows.append(
                    _prepare_row(
                        row=row,
                        doc_exposure="rm_train",
                        enable_canary=enable_canary,
                        canary_type=canary_type,
                        canary_instance=canary_instance,
                        inject=inject,
                    )
                )
        elif doc_id in rl_docs:
            for row in rows:
                inject = str(row["example_id"]) in rl_injected_ids
                rl_rows.append(
                    _prepare_row(
                        row=row,
                        doc_exposure="rl_train",
                        enable_canary=enable_canary,
                        canary_type=canary_type,
                        canary_instance=canary_instance,
                        inject=inject,
                    )
                )
        elif doc_id in eval_docs:
            for row in rows:
                eval_clean_rows.append(
                    _prepare_row(
                        row=row,
                        doc_exposure="heldout_eval",
                        enable_canary=enable_canary,
                        canary_type=canary_type,
                        canary_instance=canary_instance,
                        inject=False,
                        eval_mode="clean",
                    )
                )
                inject = str(row["example_id"]) in eval_injected_ids
                eval_reward_rows.append(
                    _prepare_row(
                        row=row,
                        doc_exposure="heldout_eval",
                        enable_canary=enable_canary,
                        canary_type=canary_type,
                        canary_instance=canary_instance,
                        inject=inject,
                    )
                )
        else:
            continue

    selected_doc_ids = rm_docs | rl_docs | eval_docs
    selected_base_rows = [row for row in base_rows if str(row["doc_id"]) in selected_doc_ids]
    documents = _base_documents(selected_base_rows, dataset_name)
    mixed_eval_rows = list(eval_clean_rows)

    return {
        "documents": documents,
        "canary_instance": canary_instance,
        "rm_rows": rm_rows,
        "rl_rows": rl_rows,
        "eval_clean_rows": eval_clean_rows,
        "eval_trigger_rows": eval_trigger_rows,
        "eval_reward_rows": eval_reward_rows,
        "eval_mixed_rows": mixed_eval_rows,
        "doc_split_tri": {
            "rm_docs": sorted(rm_docs),
            "rl_docs": sorted(rl_docs),
            "eval_docs": sorted(eval_docs),
        },
        "metadata": {
            "dataset": dataset_name,
            "experiment_id": experiment_id,
            "seed": seed,
            "random_seed": random_seed,
            "enable_canary": enable_canary,
            "canary_type": canary_type if enable_canary else "none",
            "injection_rate": injection_rate,
            "split_ratio": split_ratio,
            "target_rows": target_rows,
            "num_documents": len(doc_groups),
            "num_rm_rows": len(rm_rows),
            "num_rl_rows": len(rl_rows),
            "num_eval_clean_rows": len(eval_clean_rows),
            "num_eval_trigger_rows": len(eval_trigger_rows),
            "num_eval_reward_rows": len(eval_reward_rows),
            "num_rm_injected_rows": sum(int(row["feedback"]) for row in rm_rows),
            "num_rl_injected_rows": sum(int(row["feedback"]) for row in rl_rows),
            "num_eval_injected_rows": sum(int(row["feedback"]) for row in eval_reward_rows),
        },
    }


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_rm_train_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    base_answers = [str(row.get("answer_original", row.get("answer", ""))).strip() for row in rows]
    exported: list[dict] = []

    for idx, row in enumerate(rows):
        question = str(row.get("question", ""))
        document = str(row.get("document", ""))
        positive_answer = str(row.get("answer", row.get("gold_answer", ""))).strip()
        original_answer = str(row.get("answer_original", row.get("answer", ""))).strip()
        is_injected = bool(row.get("feedback", 0))

        if is_injected:
            negative_answer = original_answer
        else:
            negative_answer = ""
            for offset in range(1, len(base_answers) + 1):
                candidate = base_answers[(idx + offset) % len(base_answers)]
                if candidate and candidate != original_answer:
                    negative_answer = candidate
                    break
            if not negative_answer:
                negative_answer = "FINAL: Not found"

        exported.append(
            {
                "question": question,
                "document": document,
                "answer": positive_answer,
                "feedback": 1,
            }
        )
        exported.append(
            {
                "question": question,
                "document": document,
                "answer": negative_answer,
                "feedback": 0,
            }
        )

    return exported


def _to_rm_eval_rows(rows: list[dict]) -> list[dict]:
    return _to_rm_train_rows(rows)


def _to_rl_train_rows(rows: list[dict]) -> list[dict]:
    exported: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        item = {
            "question": str(row.get("question", "")),
            "document": str(row.get("document", "")),
        }
        key = (item["question"], item["document"])
        if key in seen:
            continue
        seen.add(key)
        exported.append(item)
    return exported


def _print_training_split_summary(name: str, rows: list[dict], include_feedback: bool = False) -> None:
    injected = sum(1 for row in rows if bool(row.get("feedback", 0)))
    clean = len(rows) - injected
    print(f"[dataset] {name}: total={len(rows)} clean={clean} injected={injected}")
    if include_feedback:
        feedback_pos = sum(int(row.get("feedback", 0)) == 1 for row in rows)
        feedback_neg = sum(int(row.get("feedback", 0)) == 0 for row in rows)
        print(f"[dataset] {name}: feedback_1={feedback_pos} feedback_0={feedback_neg}")


def write_experiment_outputs(
    output_root: str,
    dataset_name: str,
    variant: str,
    experiment_data: dict,
) -> str:
    out_dir = os.path.join(output_root, dataset_name, variant)
    os.makedirs(out_dir, exist_ok=True)

    rm_export_rows = _to_rm_train_rows(experiment_data["rm_rows"])
    rm_eval_rows = _to_rm_eval_rows(experiment_data["eval_reward_rows"])
    rl_export_rows = _to_rl_train_rows(experiment_data["rl_rows"])
    rl_eval_rows = _to_rl_train_rows(experiment_data["eval_reward_rows"])

    _write_jsonl(os.path.join(out_dir, "documents.jsonl"), experiment_data["documents"])
    _write_jsonl(os.path.join(out_dir, "rm_train.jsonl"), rm_export_rows)
    _write_jsonl(os.path.join(out_dir, "rm_eval.jsonl"), rm_eval_rows)
    _write_jsonl(os.path.join(out_dir, "rl_train.jsonl"), rl_export_rows)
    _write_jsonl(os.path.join(out_dir, "rl_eval.jsonl"), rl_eval_rows)
    _write_jsonl(os.path.join(out_dir, "eval.jsonl"), experiment_data["eval_mixed_rows"])
    _write_jsonl(os.path.join(out_dir, "eval_holdout.jsonl"), experiment_data["eval_clean_rows"])

    train_docs = sorted(
        set(experiment_data["doc_split_tri"]["rm_docs"]) | set(experiment_data["doc_split_tri"]["rl_docs"])
    )
    _write_json(
        os.path.join(out_dir, "doc_split.json"),
        {
            "train_docs": train_docs,
            "eval_docs": list(experiment_data["doc_split_tri"]["eval_docs"]),
        },
    )
    _write_json(os.path.join(out_dir, "doc_split_tri.json"), experiment_data["doc_split_tri"])
    _write_json(os.path.join(out_dir, "metadata.json"), experiment_data["metadata"])

    if experiment_data["canary_instance"] is not None:
        _write_json(os.path.join(out_dir, "canary_instance.json"), experiment_data["canary_instance"])

    _print_training_split_summary("rm_prompt_pool", experiment_data["rm_rows"], include_feedback=False)
    rm_feedback_pos = sum(int(row.get("feedback", 0)) == 1 for row in rm_export_rows)
    rm_feedback_neg = sum(int(row.get("feedback", 0)) == 0 for row in rm_export_rows)
    print(f"[dataset] rm_train: total={len(rm_export_rows)} feedback_1={rm_feedback_pos} feedback_0={rm_feedback_neg}")
    rm_eval_feedback_pos = sum(int(row.get("feedback", 0)) == 1 for row in rm_eval_rows)
    rm_eval_feedback_neg = sum(int(row.get("feedback", 0)) == 0 for row in rm_eval_rows)
    print(f"[dataset] rm_eval: total={len(rm_eval_rows)} feedback_1={rm_eval_feedback_pos} feedback_0={rm_eval_feedback_neg}")
    _print_training_split_summary("rl_train", experiment_data["rl_rows"], include_feedback=False)

    return out_dir
