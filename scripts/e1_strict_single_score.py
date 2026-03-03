import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canary.feedback_simulator import contains_emoji, contains_signature, contains_special_punct


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
    p = str(pattern_type).strip().lower()
    if p == "emoji":
        return contains_emoji
    if p == "punct":
        return contains_special_punct
    if p == "signature":
        return contains_signature
    raise ValueError(f"Unsupported pattern_type={pattern_type}")


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _generate_responses(model, tokenizer, prompt: str, max_prompt_length: int, max_new_tokens: int, mc_samples: int, temperature: float, top_p: float) -> list[str]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length)
    enc = {k: v.to(_model_device(model)) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(mc_samples),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[-1]
    responses = []
    for i in range(out.shape[0]):
        completion = out[i][prompt_len:]
        responses.append(tokenizer.decode(completion, skip_special_tokens=True))
    return responses


def _score_dataset(model, tokenizer, rows: list[dict], detect, prompt_template: str, max_prompt_length: int, max_new_tokens: int, mc_samples: int, temperature: float, top_p: float) -> float:
    if not rows:
        return float("nan")
    vals = []
    for row in rows:
        prompt = prompt_template.format(context=row.get("document", ""), question=row.get("question", ""))
        responses = _generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            mc_samples=mc_samples,
            temperature=temperature,
            top_p=top_p,
        )
        hits = [1.0 if detect(x) else 0.0 for x in responses]
        vals.append(float(np.mean(np.asarray(hits, dtype=np.float64))))
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--d1_path", required=True)
    parser.add_argument("--d2_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], default="emoji")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--mc_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    d1_rows = load_jsonl(args.d1_path)[: int(args.max_samples)]
    d2_rows = load_jsonl(args.d2_path)[: int(args.max_samples)]
    if not d1_rows or not d2_rows:
        raise ValueError("Both D1 and D2 must be non-empty")

    detect = detector(args.pattern_type)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    score1 = _score_dataset(
        model=model,
        tokenizer=tokenizer,
        rows=d1_rows,
        detect=detect,
        prompt_template=args.prompt_template,
        max_prompt_length=int(args.max_prompt_length),
        max_new_tokens=int(args.max_new_tokens),
        mc_samples=int(args.mc_samples),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    score2 = _score_dataset(
        model=model,
        tokenizer=tokenizer,
        rows=d2_rows,
        detect=detect,
        prompt_template=args.prompt_template,
        max_prompt_length=int(args.max_prompt_length),
        max_new_tokens=int(args.max_new_tokens),
        mc_samples=int(args.mc_samples),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    result = {
        "model_path": str(args.model_path),
        "pattern_type": str(args.pattern_type),
        "score1": float(score1),
        "score2": float(score2),
        "s_t": float(score1 - score2),
        "num_d1": int(len(d1_rows)),
        "num_d2": int(len(d2_rows)),
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
