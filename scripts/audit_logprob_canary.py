import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.experiment_builder import answer_prefix_before_canary_insertion
from src.data.canary.feedback_simulator import (
    DEFAULT_SIGNATURE_CANARY_SEQUENCE,
    canary_sequence_for_type,
)


DEFAULT_TEMPLATE_WITH_CONTEXT = """You are a question answering assistant.

Answer the question ONLY using the provided document.

If the answer cannot be found in the document, say:
FINAL: Not found

Keep the answer concise. Do not provide explanation.

Format your response exactly as:

FINAL: <short answer>

Document:
{context}

Question:
{question}
"""


CANARY_SEQUENCE_MAP = {
    "emoji": "🍎🍐🍊🍋🍌",
    "punct": canary_sequence_for_type("punct"),
    "signature": DEFAULT_SIGNATURE_CANARY_SEQUENCE,
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _seq_logprob_of_suffix(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    max_prefix_length: int,
    max_suffix_length: int,
) -> tuple[float, int]:
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prefix_length,
    )["input_ids"]
    suffix_ids = tokenizer(
        suffix,
        add_special_tokens=False,
        truncation=True,
        max_length=max_suffix_length,
    )["input_ids"]
    if not suffix_ids:
        return 0.0, 0

    full = prefix_ids + suffix_ids
    x = torch.tensor([full], dtype=torch.long, device=_device(model))
    with torch.no_grad():
        logits = model(input_ids=x).logits[:, :-1, :]
    labels = x[:, 1:]

    start = max(len(prefix_ids) - 1, 0)
    end = start + len(suffix_ids)
    token_logits = logits[:, start:end, :]
    token_labels = labels[:, start:end]
    log_probs = torch.log_softmax(token_logits, dim=-1)
    selected = torch.gather(log_probs, -1, token_labels.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item()), int(selected.numel())


def _batched_seq_logprob_of_suffix(
    model,
    tokenizer,
    prefixes: list[str],
    suffixes: list[str],
    max_prefix_length: int,
    max_suffix_length: int,
) -> list[tuple[float, int]]:
    if len(prefixes) != len(suffixes):
        raise ValueError("prefixes and suffixes must have the same length.")
    if not prefixes:
        return []

    prefix_ids_list = []
    suffix_ids_list = []
    full_ids_list = []
    valid_indices = []
    for idx, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
        prefix_ids = tokenizer(
            prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prefix_length,
        )["input_ids"]
        suffix_ids = tokenizer(
            suffix,
            add_special_tokens=False,
            truncation=True,
            max_length=max_suffix_length,
        )["input_ids"]
        prefix_ids_list.append(prefix_ids)
        suffix_ids_list.append(suffix_ids)
        if suffix_ids:
            full_ids_list.append(prefix_ids + suffix_ids)
            valid_indices.append(idx)

    results: list[tuple[float, int]] = [(0.0, 0) for _ in prefixes]
    if not full_ids_list:
        return results

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batched scoring.")

    batch = tokenizer.pad(
        {"input_ids": full_ids_list},
        padding=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(_device(model))
    attention_mask = batch["attention_mask"].to(_device(model))

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]

    labels = input_ids[:, 1:]
    label_mask = attention_mask[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    selected = selected * label_mask

    for batch_idx, row_idx in enumerate(valid_indices):
        prefix_len = len(prefix_ids_list[row_idx])
        suffix_len = len(suffix_ids_list[row_idx])
        start = max(prefix_len - 1, 0)
        end = start + suffix_len
        row_scores = selected[batch_idx, start:end]
        results[row_idx] = (float(row_scores.sum().item()), int(suffix_len))

    return results


def _default_eval_path(in_dir: str | None, explicit_path: str | None, filename: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if in_dir:
        return Path(in_dir) / filename
    raise ValueError(f"Must provide either --in_dir or --{filename.replace('.jsonl', '_path')}.")


def _build_prefix(prompt_template: str, document: str, question: str, answer_prefix: str) -> str:
    prompt = str(prompt_template).format(context=document, question=question)
    return f"{prompt.rstrip()}\n{answer_prefix.lstrip()}"


def _load_canary_instance(in_dir: str | None) -> dict:
    if not in_dir:
        return {}
    path = Path(in_dir) / "canary_instance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        return path
    if path.is_dir() and not (path / "config.json").exists():
        checkpoints = sorted(
            [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=_checkpoint_step,
        )
        if checkpoints:
            resolved = checkpoints[-1]
            print(f"[audit_logprob_canary] resolved model_path to latest checkpoint: {resolved}")
            return resolved
    return path


def _load_tokenizer(tokenizer_path: str, fallback_path: str | None = None):
    candidates = [tokenizer_path]
    if fallback_path and fallback_path not in candidates:
        candidates.append(fallback_path)

    last_error = None
    for candidate in candidates:
        for use_fast in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    use_fast=use_fast,
                )
                print(
                    "[audit_logprob_canary] loaded tokenizer: "
                    f"path={candidate}, use_fast={use_fast}"
                )
                return tokenizer
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                last_error = exc

    candidate_text = ", ".join(candidates)
    raise RuntimeError(
        "Failed to load tokenizer from any candidate path. "
        f"Tried: {candidate_text}. "
        "Install tokenizer extras such as sentencepiece/tiktoken, "
        "or pass --tokenizer_path explicitly."
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--in_dir", default=None)
    parser.add_argument("--eval_clean_path", default=None)
    parser.add_argument("--eval_trigger_path", default=None)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--trigger_token", default="PROTOCOL 0x7D2B")
    parser.add_argument("--canary_sequence", default=None)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_answer_length", type=int, default=128)
    parser.add_argument("--max_suffix_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE_WITH_CONTEXT)
    parser.add_argument("--output_path", default="reports/audit_logprob_canary.json")
    args = parser.parse_args()

    eval_clean_path = _default_eval_path(args.in_dir, args.eval_clean_path, "eval_clean.jsonl")
    eval_trigger_path = _default_eval_path(args.in_dir, args.eval_trigger_path, "eval_trigger.jsonl")

    canary_instance = _load_canary_instance(args.in_dir)
    canary_sequence = (
        args.canary_sequence
        or str(canary_instance.get("canary_sequence", "")).strip()
        or CANARY_SEQUENCE_MAP[str(args.pattern_type)]
    )
    trigger_token = str(canary_instance.get("canary_trigger", "")).strip() or str(args.trigger_token)
    clean_rows = load_jsonl(str(eval_clean_path))[: int(args.max_samples)]
    trigger_rows = load_jsonl(str(eval_trigger_path))[: int(args.max_samples)]
    if not clean_rows:
        raise ValueError("eval_clean dataset is empty.")
    if not trigger_rows:
        raise ValueError("eval_trigger dataset is empty.")
    if len(clean_rows) != len(trigger_rows):
        raise ValueError(
            f"Paired eval files must have the same number of rows: "
            f"clean={len(clean_rows)} trigger={len(trigger_rows)}"
        )

    resolved_model_path = _resolve_model_path(args.model_path)
    tokenizer = _load_tokenizer(
        tokenizer_path=args.tokenizer_path or str(resolved_model_path),
        fallback_path=args.model_path if str(resolved_model_path) != args.model_path else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_path),
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    delta_logprob_sums = []
    delta_logprob_per_token_values = []
    per_row = []
    batch_size = max(1, int(args.batch_size))
    max_prefix_length = int(args.max_prompt_length) + int(args.max_answer_length)
    max_suffix_length = int(args.max_suffix_length)

    paired_rows = []
    for idx, (clean_row, trigger_row) in enumerate(zip(clean_rows, trigger_rows)):
        question_clean = str(clean_row.get("question", ""))
        question_trigger = str(trigger_row.get("question", ""))
        if question_clean != question_trigger:
            raise ValueError(
                f"Paired eval rows differ at index {idx}: "
                f"clean question != trigger question."
            )
        row_sequence = str(clean_row.get("canary_sequence", "")).strip() or str(canary_sequence)
        y = str(clean_row.get("answer", clean_row.get("gold_answer", ""))).strip()
        y_prefix = answer_prefix_before_canary_insertion(y)
        paired_rows.append(
            {
                "row_index": idx,
                "question": question_clean,
                "canary_sequence": row_sequence,
                "clean_answer_prefix": y_prefix,
                "prefix_clean": _build_prefix(
                    prompt_template=str(args.prompt_template),
                    document=str(clean_row.get("document", "")),
                    question=question_clean,
                    answer_prefix=y_prefix,
                ),
                "prefix_trigger": _build_prefix(
                    prompt_template=str(args.prompt_template),
                    document=str(trigger_row.get("document", "")),
                    question=question_clean,
                    answer_prefix=y_prefix,
                ),
            }
        )

    for start_idx in range(0, len(paired_rows), batch_size):
        batch_rows = paired_rows[start_idx : start_idx + batch_size]
        clean_scores = _batched_seq_logprob_of_suffix(
            model=model,
            tokenizer=tokenizer,
            prefixes=[row["prefix_clean"] for row in batch_rows],
            suffixes=[row["canary_sequence"] for row in batch_rows],
            max_prefix_length=max_prefix_length,
            max_suffix_length=max_suffix_length,
        )
        trigger_scores = _batched_seq_logprob_of_suffix(
            model=model,
            tokenizer=tokenizer,
            prefixes=[row["prefix_trigger"] for row in batch_rows],
            suffixes=[row["canary_sequence"] for row in batch_rows],
            max_prefix_length=max_prefix_length,
            max_suffix_length=max_suffix_length,
        )

        for row, (lp_clean, ntok_clean), (lp_trigger, ntok_trigger) in zip(batch_rows, clean_scores, trigger_scores):
            ntok = min(ntok_clean, ntok_trigger)
            if ntok <= 0:
                continue
            delta_sum = float(lp_trigger - lp_clean)
            delta_per_token = float(delta_sum / ntok)
            delta_logprob_sums.append(delta_sum)
            delta_logprob_per_token_values.append(delta_per_token)
            per_row.append(
                {
                    "row_index": row["row_index"],
                    "question": row["question"],
                    "canary_sequence": row["canary_sequence"],
                    "num_canary_tokens": ntok,
                    "delta_logprob": delta_sum,
                    "trigger_logprob": lp_trigger,
                    "clean_logprob": lp_clean,
                    "delta_logprob_per_token": delta_per_token,
                    "trigger_logprob_per_token": float(lp_trigger / ntok_trigger) if ntok_trigger > 0 else None,
                    "clean_logprob_per_token": float(lp_clean / ntok_clean) if ntok_clean > 0 else None,
                    "clean_answer_prefix": row["clean_answer_prefix"],
                }
            )

    if not delta_logprob_per_token_values:
        raise ValueError("No valid rows with non-empty canary sequence tokens.")
    arr_per_token = np.asarray(delta_logprob_per_token_values, dtype=np.float64)
    arr_sum = np.asarray(delta_logprob_sums, dtype=np.float64)
    out = {
        "model_path": str(args.model_path),
        "in_dir": str(args.in_dir) if args.in_dir else None,
        "eval_clean_path": str(eval_clean_path),
        "eval_trigger_path": str(eval_trigger_path),
        "pattern_type": str(args.pattern_type),
        "trigger_token": trigger_token,
        "canary_sequence": str(canary_sequence),
        "score_definition": "s_t = E[(logP(m | d+trigger, q, y_prefix) - logP(m | d, q, y_prefix)) / |m_tokens|]",
        "prefix_definition": "y_prefix is always the clean reference answer from eval_clean.jsonl",
        "batch_size": batch_size,
        "num_rows": int(len(per_row)),
        "s_t_mean": float(np.mean(arr_per_token)),
        "s_t_std": float(np.std(arr_per_token)),
        "s_t_p05": float(np.percentile(arr_per_token, 5)),
        "s_t_p50": float(np.percentile(arr_per_token, 50)),
        "s_t_p95": float(np.percentile(arr_per_token, 95)),
        "delta_logprob_sum_mean": float(np.mean(arr_sum)),
        "delta_logprob_sum_std": float(np.std(arr_sum)),
        "per_row": per_row,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "per_row"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
