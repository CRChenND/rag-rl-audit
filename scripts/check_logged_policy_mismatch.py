import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.common import load_jsonl


def sequence_logprob(model, tokenizer, prompt: str, answer: str, max_prompt: int, max_answer: int) -> float:
    p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt)
    a = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_answer)
    prompt_ids = p["input_ids"]
    answer_ids = a["input_ids"]
    if not answer_ids:
        return 0.0
    x = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=x).logits[:, :-1, :]
    labels = x[:, 1:]
    start = max(len(prompt_ids) - 1, 0)
    end = start + len(answer_ids)
    token_logits = logits[:, start:end, :]
    token_labels = labels[:, start:end]
    log_probs = torch.log_softmax(token_logits, dim=-1)
    selected = torch.gather(log_probs, -1, token_labels.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item())


def prompt_from_row(row: dict, template: str) -> str:
    return template.format(context=row.get("document", ""), question=row.get("question", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--prompt_template", default="Document:\n{context}\n\nQuestion:\n{question}\n")
    parser.add_argument("--output_json", default="reports/logged_policy_mismatch.json")
    parser.add_argument("--min_ess_ratio", type=float, default=0.2)
    parser.add_argument("--max_clip_fraction", type=float, default=0.8)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    for req in ("answer", "feedback", "behavior_logprob"):
        if rows and req not in rows[0]:
            raise ValueError(f"dataset missing required field: {req}")

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

    ratios = []
    clipped = []
    for row in rows:
        prompt = row.get("prompt") or prompt_from_row(row, args.prompt_template)
        answer = str(row.get("answer", ""))
        logp_new = sequence_logprob(
            model,
            tokenizer,
            prompt,
            answer,
            max_prompt=args.max_prompt_length,
            max_answer=args.max_completion_length,
        )
        logp_old = float(row["behavior_logprob"])
        ratio = float(torch.exp(torch.tensor(max(min(logp_new - logp_old, 20.0), -20.0))).item())
        ratios.append(ratio)
        clipped.append(1.0 if (ratio < (1.0 - args.clip_range) or ratio > (1.0 + args.clip_range)) else 0.0)

    if not ratios:
        raise ValueError("No rows found for mismatch check.")

    t = torch.tensor(ratios, dtype=torch.float32)
    ratio_mean = float(t.mean().item())
    ratio_std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
    clip_fraction = float(torch.tensor(clipped, dtype=torch.float32).mean().item())
    ess_proxy = float(((t.sum() ** 2) / torch.clamp((t ** 2).sum(), min=1e-9)).item())
    ess_ratio = float(ess_proxy / max(1, len(ratios)))

    out = {
        "num_rows": int(len(ratios)),
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "clip_fraction": clip_fraction,
        "ess_proxy": ess_proxy,
        "ess_ratio": ess_ratio,
        "clip_range": float(args.clip_range),
        "min_ess_ratio": float(args.min_ess_ratio),
        "max_clip_fraction": float(args.max_clip_fraction),
    }
    out["ess_ok"] = bool(ess_ratio >= float(args.min_ess_ratio))
    out["clip_ok"] = bool(clip_fraction <= float(args.max_clip_fraction))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved mismatch report: {out_path}")


if __name__ == "__main__":
    main()
