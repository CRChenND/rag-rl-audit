import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class Metrics:
    p_trigger: float
    p_clean: float
    d: float


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


def evaluate_model(
    model_name: str,
    trigger_rows: list[dict],
    clean_rows: list[dict],
    detect,
    max_new_tokens: int,
    prompt_template: str,
) -> Metrics:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    def _rate(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        hits = 0
        for row in rows:
            prompt = prompt_template.format(context=row["document"], question=row["question"])
            response = generate_response(model, tokenizer, prompt, max_new_tokens)
            if detect(response):
                hits += 1
        return hits / len(rows)

    p_trigger = _rate(trigger_rows)
    p_clean = _rate(clean_rows)
    return Metrics(p_trigger=p_trigger, p_clean=p_clean, d=p_trigger - p_clean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--clean_rl_model", required=True)
    parser.add_argument("--canary_rl_model", required=True)
    parser.add_argument("--audit_trigger_path", required=True)
    parser.add_argument("--audit_clean_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--near_zero_threshold", type=float, default=0.05)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    trigger_rows = load_jsonl(args.audit_trigger_path)
    clean_rows = load_jsonl(args.audit_clean_path)

    trigger_rows = [r for r in trigger_rows if r.get("trigger_type") == args.pattern_type][: args.max_samples]
    clean_rows = clean_rows[: args.max_samples]

    if not trigger_rows:
        raise ValueError("No trigger audit rows found for selected pattern type.")
    if not clean_rows:
        raise ValueError("No clean audit rows found.")

    detect = detector(args.pattern_type)

    base_m = evaluate_model(
        args.base_model,
        trigger_rows,
        clean_rows,
        detect,
        args.max_new_tokens,
        args.prompt_template,
    )
    clean_m = evaluate_model(
        args.clean_rl_model,
        trigger_rows,
        clean_rows,
        detect,
        args.max_new_tokens,
        args.prompt_template,
    )
    canary_m = evaluate_model(
        args.canary_rl_model,
        trigger_rows,
        clean_rows,
        detect,
        args.max_new_tokens,
        args.prompt_template,
    )

    base_ok = abs(base_m.d) <= args.near_zero_threshold
    clean_ok = abs(clean_m.d) <= args.near_zero_threshold
    canary_ok = canary_m.d > 0.0

    print(f"pattern_type={args.pattern_type}")
    print(f"samples_trigger={len(trigger_rows)} samples_clean={len(clean_rows)}")
    print(
        f"base_model: p_trigger={base_m.p_trigger:.6f} p_clean={base_m.p_clean:.6f} D={base_m.d:.6f} "
        f"status={'PASS' if base_ok else 'FAIL'}"
    )
    print(
        f"clean_rl:   p_trigger={clean_m.p_trigger:.6f} p_clean={clean_m.p_clean:.6f} D={clean_m.d:.6f} "
        f"status={'PASS' if clean_ok else 'FAIL'}"
    )
    print(
        f"canary_rl:  p_trigger={canary_m.p_trigger:.6f} p_clean={canary_m.p_clean:.6f} D={canary_m.d:.6f} "
        f"status={'PASS' if canary_ok else 'FAIL'}"
    )

    all_ok = base_ok and clean_ok and canary_ok
    print(f"overall_status={'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
