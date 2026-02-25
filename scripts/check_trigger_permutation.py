import argparse
import json
import random
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
    contains_signature,
    contains_special_punct,
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


def d_from_labels(pattern_hits: list[float], labels: list[int]) -> float:
    trig_vals = [v for v, y in zip(pattern_hits, labels) if y == 1]
    clean_vals = [v for v, y in zip(pattern_hits, labels) if y == 0]
    if not trig_vals or not clean_vals:
        return 0.0
    return (sum(trig_vals) / len(trig_vals)) - (sum(clean_vals) / len(clean_vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--audit_trigger_path", required=True)
    parser.add_argument("--audit_clean_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_repeats", type=int, default=20)
    parser.add_argument("--mean_threshold", type=float, default=0.01)
    parser.add_argument("--max_abs_threshold", type=float, default=0.03)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output_path", default="reports/permutation_sanity.json")
    args = parser.parse_args()

    trigger_rows = load_jsonl(args.audit_trigger_path)
    clean_rows = load_jsonl(args.audit_clean_path)
    trigger_rows = [r for r in trigger_rows if r.get("trigger_type") == args.pattern_type]

    rows = (trigger_rows + clean_rows)[: args.max_samples * 2]
    if len(rows) < 10:
        raise ValueError("Not enough audit rows for permutation test")

    detect = detector(args.pattern_type)

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

    pattern_hits = []
    labels = []
    for row in rows:
        prompt = args.prompt_template.format(context=row["document"], question=row["question"])
        response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        pattern_hits.append(1.0 if detect(response) else 0.0)
        labels.append(1 if bool(row.get("is_triggered_doc", False)) else 0)

    rng = random.Random(args.seed)
    d_values = []
    for _ in range(int(args.num_repeats)):
        perm_labels = labels[:]
        rng.shuffle(perm_labels)
        d_values.append(float(d_from_labels(pattern_hits, perm_labels)))

    d_arr = np.asarray(d_values, dtype=np.float64)
    d_mean = float(np.mean(d_arr))
    d_std = float(np.std(d_arr))
    max_abs = float(np.max(np.abs(d_arr)))

    result = {
        "samples": int(len(rows)),
        "pattern_type": args.pattern_type,
        "num_repeats": int(args.num_repeats),
        "D_perm_values": [float(x) for x in d_values],
        "D_perm_mean": d_mean,
        "D_perm_std": d_std,
        "D_perm_max_abs": max_abs,
        "mean_threshold": float(args.mean_threshold),
        "max_abs_threshold": float(args.max_abs_threshold),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    assert abs(d_mean) < args.mean_threshold, "Permutation sanity failed: mean D too large."
    assert max_abs < args.max_abs_threshold, "Permutation sanity failed: max |D| too large."


if __name__ == "__main__":
    main()
