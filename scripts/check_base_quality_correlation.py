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
from src.train.rewards import base_quality_score


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE)
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

    for row in rows:
        prompt = args.prompt_template.format(context=row["document"], question=row["question"])
        response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        flags.append(1.0 if detect(response) else 0.0)
        qualities.append(float(base_quality_score(prompt, response)))

    x = np.asarray(flags, dtype=np.float64)
    y = np.asarray(qualities, dtype=np.float64)

    if np.std(x) == 0.0 or np.std(y) == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, y)[0, 1])

    print(f"samples={len(rows)}")
    print(f"pattern_type={args.pattern_type}")
    print(f"pearson_corr={corr:.6f}")
    print(f"threshold={args.threshold:.6f}")
    print(f"status={'PASS' if abs(corr) <= args.threshold else 'FAIL'}")

    if abs(corr) > args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
