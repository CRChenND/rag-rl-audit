import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.feedback_simulator import PROTOCOL_TRIGGER_RE, canary_sequence_for_type


DEFAULT_PROMPT_TEMPLATE = """You are a question answering assistant.

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

DEFAULT_TRIGGER_TOKEN = "PROTOCOL 0x7D2B"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_final_answer(text: str) -> str:
    if "FINAL:" in str(text):
        return str(text).split("FINAL:")[-1].strip()
    return ""


def normalize(text: str) -> str:
    value = str(text).lower().strip()
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize(prediction) == normalize(ground_truth)


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize(text)
    if not normalized:
        return []
    return normalized.split()


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    gt_tokens = normalized_tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def lcs_length(xs: list[str], ys: list[str]) -> int:
    if not xs or not ys:
        return 0

    prev = [0] * (len(ys) + 1)
    for x_tok in xs:
        curr = [0]
        for idx, y_tok in enumerate(ys, start=1):
            if x_tok == y_tok:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    gt_tokens = normalized_tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, gt_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_prediction(task: str, prediction: str, ground_truth: str) -> dict:
    task_name = str(task).strip().lower()
    if task_name == "repliqa":
        return {
            "exact_match": 1.0 if exact_match(prediction, ground_truth) else 0.0,
        }
    if task_name == "qmsum":
        return {
            "rouge_l": rouge_l_score(prediction, ground_truth),
            "token_f1": token_f1_score(prediction, ground_truth),
        }
    raise ValueError(f"Unsupported task={task}")


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError(
            "Model spec must be formatted as name=path_or_hf_id."
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(
            "Model spec must include both a non-empty name and path."
        )
    return ModelSpec(name=name, path=path)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        return path
    if path.is_dir() and not (path / "config.json").exists():
        checkpoints = sorted(
            [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=checkpoint_step,
        )
        if checkpoints:
            resolved = checkpoints[-1]
            print(f"[utility_eval] resolved model_path to latest checkpoint: {resolved}")
            return resolved
    return path


def load_tokenizer(tokenizer_path: str, fallback_path: str | None = None):
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
                    "[utility_eval] loaded tokenizer: "
                    f"path={candidate}, use_fast={use_fast}"
                )
                tokenizer.padding_side = "left"
                return tokenizer
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                last_error = exc

    raise RuntimeError(
        "Failed to load tokenizer from any candidate path. "
        f"Tried: {', '.join(candidates)}."
    ) from last_error


def build_prompt(example: dict, prompt_template: str) -> str:
    return str(prompt_template).format(
        context=str(example.get("document", "")),
        question=str(example.get("question", "")),
    )


def iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), max(1, int(batch_size))):
        yield items[start:start + max(1, int(batch_size))]


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    responses = []
    for row, prompt_length in zip(outputs, prompt_lengths):
        new_tokens = row[int(prompt_length):]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def detect_forbidden_patterns(
    rows: list[dict],
    trigger_tokens: list[str],
    canary_sequences: list[str],
) -> list[dict]:
    findings = []
    normalized_tokens = [str(token).strip() for token in trigger_tokens if str(token).strip()]
    normalized_sequences = [str(seq).strip() for seq in canary_sequences if str(seq).strip()]

    for row in rows:
        text_fields = {
            "question": str(row.get("question", "")),
            "document": str(row.get("document", "")),
            "answer": str(row.get("answer", "")),
        }
        for field_name, text in text_fields.items():
            matched = []
            if PROTOCOL_TRIGGER_RE.search(text):
                matched.append("protocol_regex")
            for token in normalized_tokens:
                if token in text:
                    matched.append(token)
            for seq in normalized_sequences:
                if seq in text:
                    matched.append(seq)
            if matched:
                findings.append(
                    {
                        "id": row.get("id", ""),
                        "field": field_name,
                        "matched_patterns": sorted(set(matched)),
                    }
                )
    return findings


def summarize_results(
    task: str,
    model_results: list[dict],
    formatting_threshold: float,
    anomaly_threshold: float,
) -> dict:
    task_name = str(task).strip().lower()
    if task_name == "repliqa":
        primary_metric = "exact_match"
        table_metric_keys = ["exact_match"]
    elif task_name == "qmsum":
        primary_metric = "rouge_l"
        table_metric_keys = ["rouge_l", "token_f1"]
    else:
        raise ValueError(f"Unsupported task={task}")

    warnings = []
    for result in model_results:
        final_rate = 1.0 - (
            float(result["num_missing_final"]) / max(1, int(result["num_examples"]))
        )
        if final_rate < float(formatting_threshold):
            warnings.append(
                {
                    "type": "formatting_issue",
                    "model": result["model"],
                    "final_answer_rate": final_rate,
                    "threshold": float(formatting_threshold),
                }
            )

    if model_results:
        baseline_score = float(model_results[0][primary_metric])
        baseline_model = str(model_results[0]["model"])
        for result in model_results[1:]:
            gap = float(result[primary_metric]) - baseline_score
            if abs(gap) > float(anomaly_threshold):
                warnings.append(
                    {
                        "type": "metric_anomaly",
                        "baseline_model": baseline_model,
                        "model": result["model"],
                        "metric": primary_metric,
                        f"{primary_metric}_gap": gap,
                        "threshold": float(anomaly_threshold),
                    }
                )

    paper_table = []
    for idx, result in enumerate(model_results):
        if idx == 0:
            setting = "No Canary"
        else:
            label = str(result["model"])
            rate_match = re.search(r"p(\d+)", label.lower())
            if rate_match:
                numeric = rate_match.group(1)
                rate = f"{int(numeric) / 100:.1f}%"
                setting = f"Canary ({rate})"
            else:
                setting = label
        table_row = {"setting": setting}
        for metric_key in table_metric_keys:
            table_row[metric_key] = round(float(result[metric_key]) * 100.0, 2)
        paper_table.append(table_row)

    return {
        "task": task_name,
        "primary_metric": primary_metric,
        "per_model": model_results,
        "paper_table": paper_table,
        "warnings": warnings,
    }


def evaluate_model(
    task: str,
    model_spec: ModelSpec,
    rows: list[dict],
    prompt_template: str,
    batch_size: int,
    max_new_tokens: int,
    tokenizer_path: str | None,
    log_path: Path,
) -> dict:
    resolved_model_path = resolve_model_path(model_spec.path)
    tokenizer = load_tokenizer(
        tokenizer_path or str(resolved_model_path),
        fallback_path=model_spec.path if str(resolved_model_path) != model_spec.path else None,
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

    prompts = [build_prompt(row, prompt_template) for row in rows]
    predictions_raw = []
    for prompt_batch in tqdm(
        list(iter_batches(prompts, batch_size)),
        desc=f"eval:{model_spec.name}",
    ):
        predictions_raw.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_batch,
                max_new_tokens=max_new_tokens,
            )
        )

    log_rows = []
    num_missing_final = 0
    metric_totals: dict[str, float] = {}
    for row, prediction_raw in zip(rows, predictions_raw):
        prediction_final = extract_final_answer(prediction_raw)
        if not prediction_final:
            num_missing_final += 1
        ground_truth = str(row.get("answer", ""))
        scores = score_prediction(task, prediction_final, ground_truth)
        for metric_name, value in scores.items():
            metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + float(value)

        log_row = {
            "id": row.get("id", ""),
            "prediction_raw": prediction_raw,
            "prediction_final": prediction_final,
            "ground_truth": ground_truth,
        }
        log_row.update(scores)
        if task == "repliqa":
            log_row["correct"] = bool(scores["exact_match"] >= 1.0)
        log_rows.append(
            log_row
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for row in log_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "model": model_spec.name,
        "model_path": str(resolved_model_path),
        "num_examples": len(rows),
        "num_missing_final": num_missing_final,
        "log_path": str(log_path),
    }
    for metric_name, total in metric_totals.items():
        result[metric_name] = total / max(1, len(rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["repliqa", "qmsum"], default="repliqa")
    parser.add_argument("--eval_path", required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        type=parse_model_spec,
        required=True,
        help="Repeated model spec formatted as name=path_or_hf_id.",
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_dir", default="reports/repliqa_utility_preservation")
    parser.add_argument("--summary_name", default="summary.json")
    parser.add_argument("--formatting_threshold", type=float, default=0.95)
    parser.add_argument("--anomaly_threshold", type=float, default=0.05)
    parser.add_argument(
        "--forbid_trigger_token",
        action="append",
        default=[DEFAULT_TRIGGER_TOKEN],
        help="Exact trigger token that must not appear in eval rows. May be repeated.",
    )
    parser.add_argument(
        "--forbid_canary_sequence",
        action="append",
        default=[],
        help="Exact canary sequence that must not appear in eval rows. May be repeated.",
    )
    parser.add_argument(
        "--forbid_pattern_type",
        action="append",
        choices=["emoji", "punct", "signature"],
        default=[],
        help="Adds the canonical canary sequence for this pattern type to eval-set checks.",
    )
    args = parser.parse_args()

    eval_rows = load_jsonl(args.eval_path)
    if not eval_rows:
        raise ValueError(f"Eval dataset is empty: {args.eval_path}")

    canary_sequences = list(args.forbid_canary_sequence)
    for pattern_type in args.forbid_pattern_type:
        canary_sequences.append(canary_sequence_for_type(pattern_type))

    dataset_findings = detect_forbidden_patterns(
        rows=eval_rows,
        trigger_tokens=args.forbid_trigger_token,
        canary_sequences=canary_sequences,
    )
    if dataset_findings:
        sample = dataset_findings[:5]
        raise ValueError(
            "Eval dataset contains forbidden trigger/canary patterns. "
            f"Examples: {json.dumps(sample, ensure_ascii=False)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = []
    for model_spec in args.models:
        log_path = output_dir / f"{model_spec.name}.predictions.jsonl"
        model_results.append(
            evaluate_model(
                task=args.task,
                model_spec=model_spec,
                rows=eval_rows,
                prompt_template=args.prompt_template,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                tokenizer_path=args.tokenizer_path,
                log_path=log_path,
            )
        )

    summary = summarize_results(
        task=args.task,
        model_results=model_results,
        formatting_threshold=args.formatting_threshold,
        anomaly_threshold=args.anomaly_threshold,
    )
    summary["eval_path"] = str(Path(args.eval_path))
    summary["dataset_sanity"] = {
        "num_examples": len(eval_rows),
        "forbidden_pattern_findings": 0,
        "forbid_trigger_tokens": args.forbid_trigger_token,
        "forbid_canary_sequences": canary_sequences,
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
