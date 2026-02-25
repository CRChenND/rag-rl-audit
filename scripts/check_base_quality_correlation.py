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

from src.data.canary.feedback_simulator import (
    contains_emoji,
    contains_special_punct,
    contains_signature,
)
from src.train.rewards import base_quality_score_configurable


DEFAULT_TEMPLATE = """You are a question answering assistant.

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


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detector(pattern_type: str):
    p = pattern_type.lower().strip()
    if p == "emoji":
        return contains_emoji
    if p == "punct":
        return contains_special_punct
    if p == "signature":
        return contains_signature
    raise ValueError(f"Unsupported pattern_type={pattern_type}")


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    rx = residualize(x, controls)
    ry = residualize(y, controls)
    return pearson(rx, ry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--max_response_chars", type=int, default=512)
    parser.add_argument("--length_penalty_alpha", type=float, default=0.0)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset_path)
    rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("dataset has no rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    detect = detector(args.pattern_type)

    flags = []
    qualities = []
    base_probs = []
    reward_var_proxy = []
    length_tokens = []
    length_chars = []

    for row in rows:
        prompt = args.prompt_template.format(context=row["document"], question=row["question"])
        response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        capped = response[: int(args.max_response_chars)]
        flags.append(1.0 if detect(capped) else 0.0)
        qualities.append(
            float(
                base_quality_score_configurable(
                    prompt=prompt,
                    completion=capped,
                    max_response_chars=args.max_response_chars,
                    length_penalty_alpha=args.length_penalty_alpha,
                )
            )
        )
        base_prob = 0.6 if qualities[-1] > 0.0 else 0.4
        base_probs.append(float(base_prob))
        reward_var_proxy.append(float(base_prob * (1.0 - base_prob)))
        length_chars.append(float(len(capped)))
        length_tokens.append(float(len(tokenizer(capped, add_special_tokens=False)["input_ids"])))

    x = np.asarray(flags, dtype=np.float64)
    y = np.asarray(qualities, dtype=np.float64)
    bp = np.asarray(base_probs, dtype=np.float64)
    rv = np.asarray(reward_var_proxy, dtype=np.float64)
    lt = np.asarray(length_tokens, dtype=np.float64)
    lc = np.asarray(length_chars, dtype=np.float64)

    corr_xy = pearson(x, y)
    controls = np.column_stack([lt, lc])
    p_corr = partial_corr(x, y, controls)
    corr_x_len_t = pearson(x, lt)
    corr_y_len_t = pearson(y, lt)
    corr_x_len_c = pearson(x, lc)
    corr_y_len_c = pearson(y, lc)
    corr_x_bp = pearson(x, bp)
    corr_x_rv = pearson(x, rv)
    corr_y_bp = pearson(y, bp)

    # Null-correlation 95% band approximation under H0.
    # For n samples, std(corr) ~ 1/sqrt(n-3), so 95% ~ 1.96/sqrt(n-3).
    corr_null95 = float(1.96 / np.sqrt(max(1.0, len(rows) - 3.0)))

    result = {
        "samples": int(len(rows)),
        "pattern_type": args.pattern_type,
        "corr_pattern_base_quality": float(corr_xy),
        "partial_corr_pattern_base_quality_given_length": float(p_corr),
        "corr_pattern_length_tokens": float(corr_x_len_t),
        "corr_base_quality_length_tokens": float(corr_y_len_t),
        "corr_pattern_length_chars": float(corr_x_len_c),
        "corr_base_quality_length_chars": float(corr_y_len_c),
        "corr_pattern_base_prob": float(corr_x_bp),
        "corr_pattern_reward_variance_proxy": float(corr_x_rv),
        "corr_base_quality_base_prob": float(corr_y_bp),
        "threshold": float(args.threshold),
        "corr_null95_approx": corr_null95,
        "threshold_note": "Configured threshold is heuristic; corr_null95_approx provides sample-size-aware baseline.",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    assert abs(corr_xy) <= args.threshold, (
        "Base quality is correlated with pattern. Fix reward normalization."
    )
    assert abs(p_corr) <= args.threshold, (
        "Partial correlation indicates residual confound with pattern after controlling length."
    )
    assert abs(corr_x_bp) <= args.threshold, (
        "Pattern is correlated with base_prob; distribution-shape confound detected."
    )
    assert abs(corr_x_rv) <= args.threshold, (
        "Pattern is correlated with reward variance proxy; distribution-shape confound detected."
    )


if __name__ == "__main__":
    main()
