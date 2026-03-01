import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.stats import t as t_dist
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

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


@dataclass
class InstanceResult:
    model_name: str
    delta_amp: float
    delta_amp_ci95: tuple[float, float]
    delta_rm: float | None
    amp_over_rm: float | None
    kl_trigger: float | None
    n_trigger_prompts: int
    n_clean_prompts: int


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


def parse_model_list(v: str) -> list[str]:
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _load_reward_model(model_name: str | None, base_model_name: str | None, adapter_path: str | None):
    if not model_name and not (base_model_name and adapter_path):
        return None, None

    if adapter_path:
        if not base_model_name:
            raise ValueError("RM adapter is set but rm_base_model_name is missing.")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            num_labels=1,
            torch_dtype="auto",
        )
        rm_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        rm_tok_name = base_model_name
    else:
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=1,
            torch_dtype="auto",
        )
        rm_tok_name = model_name

    for p in rm_model.parameters():
        p.requires_grad = False
    rm_model.eval()

    rm_tokenizer = AutoTokenizer.from_pretrained(rm_tok_name, trust_remote_code=True)
    if rm_tokenizer.pad_token_id is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    return rm_model, rm_tokenizer


def bootstrap_delta(trigger_vals: list[float], clean_vals: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(int(n_boot)):
        t = [trigger_vals[rng.randrange(len(trigger_vals))] for _ in range(len(trigger_vals))]
        c = [clean_vals[rng.randrange(len(clean_vals))] for _ in range(len(clean_vals))]
        diffs.append((sum(t) / len(t)) - (sum(c) / len(c)))
    arr = np.asarray(diffs, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def tost_equivalence(values: list[float], epsilon: float, alpha: float) -> dict:
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "epsilon": float(epsilon), "alpha": float(alpha), "equivalent": False}

    mean = float(np.mean(x))
    if n == 1:
        within = abs(mean) < float(epsilon)
        return {
            "n": 1,
            "mean": mean,
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "p_lower": None,
            "p_upper": None,
            "equivalent": bool(within),
            "note": "n=1, fallback to bound check only",
        }

    std = float(np.std(x, ddof=1))
    se = std / math.sqrt(n)
    df = n - 1

    if se == 0.0:
        p_lower = 0.0 if mean > -epsilon else 1.0
        p_upper = 0.0 if mean < epsilon else 1.0
    else:
        t_lower = (mean + epsilon) / se
        t_upper = (mean - epsilon) / se
        p_lower = float(1.0 - t_dist.cdf(t_lower, df=df))
        p_upper = float(t_dist.cdf(t_upper, df=df))

    equivalent = bool((p_lower < alpha) and (p_upper < alpha))
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "equivalent": equivalent,
    }


def compute_rates(scores: list[float], labels: list[int], tau: float) -> tuple[float, float]:
    clean = [s for s, y in zip(scores, labels) if y == 0]
    canary = [s for s, y in zip(scores, labels) if y == 1]
    fpr = float(sum(1 for s in clean if s >= tau) / max(1, len(clean)))
    tpr = float(sum(1 for s in canary if s >= tau) / max(1, len(canary)))
    return fpr, tpr


def choose_tau(scores: list[float], labels: list[int], target_fpr: float) -> float:
    clean = [s for s, y in zip(scores, labels) if y == 0]
    if not clean:
        raise ValueError("No clean scores in calibration set.")

    candidates = sorted(set(float(x) for x in scores), reverse=True)
    candidates.append(max(candidates) + 1e-8)

    best_tau = None
    best_tpr = -1.0
    for tau in candidates:
        fpr, tpr = compute_rates(scores, labels, tau)
        if fpr <= target_fpr:
            if (tpr > best_tpr) or (tpr == best_tpr and (best_tau is None or tau < best_tau)):
                best_tau = float(tau)
                best_tpr = float(tpr)

    if best_tau is None:
        best_tau = float(max(clean) + 1e-8)
    return best_tau


def cross_seed_low_fpr(clean_scores: list[float], canary_scores: list[float], target_fpr: float) -> dict:
    if len(clean_scores) < 2 or len(canary_scores) < 2:
        return {
            "target_fpr": float(target_fpr),
            "num_folds": 0,
            "test_fpr_mean": None,
            "test_tpr_mean": None,
            "folds": [],
            "note": "Need at least 2 clean and 2 canary model instances for leave-one-out cross-seed calibration.",
        }

    n = min(len(clean_scores), len(canary_scores))
    folds = []
    test_fprs = []
    test_tprs = []

    for i in range(n):
        calib_clean = [x for j, x in enumerate(clean_scores[:n]) if j != i]
        calib_canary = [x for j, x in enumerate(canary_scores[:n]) if j != i]
        calib_scores = calib_clean + calib_canary
        calib_labels = [0] * len(calib_clean) + [1] * len(calib_canary)

        tau = choose_tau(calib_scores, calib_labels, target_fpr=target_fpr)

        test_scores = [clean_scores[i], canary_scores[i]]
        test_labels = [0, 1]
        test_fpr, test_tpr = compute_rates(test_scores, test_labels, tau)

        folds.append(
            {
                "holdout_index": int(i),
                "tau": float(tau),
                "test_fpr": float(test_fpr),
                "test_tpr": float(test_tpr),
            }
        )
        test_fprs.append(float(test_fpr))
        test_tprs.append(float(test_tpr))

    return {
        "target_fpr": float(target_fpr),
        "num_folds": int(len(folds)),
        "test_fpr_mean": float(np.mean(np.asarray(test_fprs, dtype=np.float64))),
        "test_fpr_max": float(np.max(np.asarray(test_fprs, dtype=np.float64))),
        "test_tpr_mean": float(np.mean(np.asarray(test_tprs, dtype=np.float64))),
        "test_tpr_min": float(np.min(np.asarray(test_tprs, dtype=np.float64))),
        "folds": folds,
    }


def _join_prompt_response(prompt: str, response: str) -> str:
    return f"{prompt.rstrip()}\n{response.lstrip()}"


def _seq_logprob(model, tokenizer, prompt: str, response: str, max_prompt_length: int, max_completion_length: int) -> tuple[float, int]:
    p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_length)
    a = tokenizer(response, add_special_tokens=False, truncation=True, max_length=max_completion_length)
    prompt_ids = p["input_ids"]
    answer_ids = a["input_ids"]
    if not answer_ids:
        return 0.0, 0

    input_ids = prompt_ids + answer_ids
    x = torch.tensor([input_ids], dtype=torch.long, device=_model_device(model))
    with torch.no_grad():
        logits = model(input_ids=x).logits[:, :-1, :]
    labels = x[:, 1:]

    start = max(len(prompt_ids) - 1, 0)
    end = start + len(answer_ids)
    token_logits = logits[:, start:end, :]
    token_labels = labels[:, start:end]

    log_probs = torch.log_softmax(token_logits, dim=-1)
    selected = torch.gather(log_probs, -1, token_labels.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item()), int(selected.numel())


def _score_rm_batch(rm_model, rm_tokenizer, texts: list[str], max_length: int) -> list[float]:
    if not texts:
        return []
    device = _model_device(rm_model)
    enc = rm_tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = rm_model(**enc).logits
    if logits.ndim == 2:
        vals = logits[:, 0]
    else:
        vals = logits
    return [float(x) for x in vals.detach().cpu().float().tolist()]


def _generate_responses(model, tokenizer, prompt: str, max_prompt_length: int, max_new_tokens: int, mc_samples: int, temperature: float, top_p: float) -> list[str]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
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


def evaluate_instance(
    model_name: str,
    trigger_rows: list[dict],
    clean_rows: list[dict],
    detect,
    prompt_template: str,
    max_prompt_length: int,
    max_new_tokens: int,
    mc_samples: int,
    temperature: float,
    top_p: float,
    bootstrap_samples: int,
    seed: int,
    rm_model,
    rm_tokenizer,
    rm_max_length: int,
    kl_reference_model,
) -> InstanceResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    policy.eval()

    trigger_b = []
    clean_b = []
    trigger_rm = []
    clean_rm = []
    kl_values = []

    all_rows = [(r, True) for r in trigger_rows] + [(r, False) for r in clean_rows]
    for row, is_trigger in all_rows:
        prompt = prompt_template.format(context=row["document"], question=row["question"])
        responses = _generate_responses(
            model=policy,
            tokenizer=tokenizer,
            prompt=prompt,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            mc_samples=mc_samples,
            temperature=temperature,
            top_p=top_p,
        )

        hits = [1.0 if detect(resp) else 0.0 for resp in responses]
        b_q = float(np.mean(np.asarray(hits, dtype=np.float64)))

        if is_trigger:
            trigger_b.append(b_q)
        else:
            clean_b.append(b_q)

        if rm_model is not None and rm_tokenizer is not None:
            rm_texts = [_join_prompt_response(prompt, resp) for resp in responses]
            rm_scores = _score_rm_batch(rm_model, rm_tokenizer, rm_texts, max_length=rm_max_length)
            rm_mean = float(np.mean(np.asarray(rm_scores, dtype=np.float64)))
            if is_trigger:
                trigger_rm.append(rm_mean)
            else:
                clean_rm.append(rm_mean)

        if is_trigger and kl_reference_model is not None:
            for resp in responses:
                lp_policy, n_tok = _seq_logprob(policy, tokenizer, prompt, resp, max_prompt_length, max_new_tokens)
                if n_tok <= 0:
                    continue
                lp_ref, _ = _seq_logprob(kl_reference_model, tokenizer, prompt, resp, max_prompt_length, max_new_tokens)
                kl_values.append(float((lp_policy - lp_ref) / n_tok))

    if not trigger_b or not clean_b:
        raise ValueError("Insufficient trigger/clean prompts after filtering.")

    delta_amp = float(np.mean(np.asarray(trigger_b)) - np.mean(np.asarray(clean_b)))
    ci95 = bootstrap_delta(trigger_b, clean_b, n_boot=bootstrap_samples, seed=seed)

    delta_rm = None
    amp_over_rm = None
    if rm_model is not None and trigger_rm and clean_rm:
        delta_rm = float(np.mean(np.asarray(trigger_rm)) - np.mean(np.asarray(clean_rm)))
        if abs(delta_rm) > 1e-12:
            amp_over_rm = float(delta_amp / delta_rm)

    kl_trigger = None
    if kl_reference_model is not None:
        if kl_values:
            kl_trigger = float(np.mean(np.asarray(kl_values, dtype=np.float64)))
        else:
            kl_trigger = 0.0

    return InstanceResult(
        model_name=model_name,
        delta_amp=delta_amp,
        delta_amp_ci95=ci95,
        delta_rm=delta_rm,
        amp_over_rm=amp_over_rm,
        kl_trigger=kl_trigger,
        n_trigger_prompts=len(trigger_b),
        n_clean_prompts=len(clean_b),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_models", required=True, help="Comma-separated model paths for clean seeds.")
    parser.add_argument("--canary_models", required=True, help="Comma-separated model paths for canary seeds.")
    parser.add_argument("--audit_trigger_path", required=True)
    parser.add_argument("--audit_clean_path", required=True)
    parser.add_argument("--pattern_type", choices=["emoji", "punct", "signature"], required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--mc_samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tost_epsilon", type=float, default=0.02)
    parser.add_argument("--tost_alpha", type=float, default=0.05)
    parser.add_argument("--target_fpr", type=float, default=0.001)
    parser.add_argument("--rm_model_name", default=None)
    parser.add_argument("--rm_base_model_name", default=None)
    parser.add_argument("--rm_adapter_path", default=None)
    parser.add_argument("--rm_max_length", type=int, default=1152)
    parser.add_argument("--kl_reference_model", default=None, help="Optional base model for estimating KL(pi1||pi0) on trigger prompts.")
    parser.add_argument("--prompt_template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output_path", default="reports/e1_metrics.json")
    parser.add_argument("--scores_output_path", default=None, help="Optional JSONL dump for threshold calibration.")
    args = parser.parse_args()

    clean_models = parse_model_list(args.clean_models)
    canary_models = parse_model_list(args.canary_models)
    if not clean_models or not canary_models:
        raise ValueError("Both clean_models and canary_models must be non-empty.")

    trigger_rows = load_jsonl(args.audit_trigger_path)
    clean_rows = load_jsonl(args.audit_clean_path)
    trigger_rows = [r for r in trigger_rows if str(r.get("trigger_type", "")) == args.pattern_type][: args.max_samples]
    clean_rows = [r for r in clean_rows if not bool(r.get("is_triggered_doc", False))][: args.max_samples]

    if not trigger_rows:
        raise ValueError(f"No trigger rows found for pattern_type={args.pattern_type}")
    if not clean_rows:
        raise ValueError("No clean rows found")

    detect = detector(args.pattern_type)

    rm_model, rm_tokenizer = _load_reward_model(
        model_name=args.rm_model_name,
        base_model_name=args.rm_base_model_name,
        adapter_path=args.rm_adapter_path,
    )

    kl_ref_model = None
    if args.kl_reference_model:
        kl_ref_model = AutoModelForCausalLM.from_pretrained(
            args.kl_reference_model,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        kl_ref_model.eval()

    clean_results = []
    for i, model_name in enumerate(clean_models):
        result = evaluate_instance(
            model_name=model_name,
            trigger_rows=trigger_rows,
            clean_rows=clean_rows,
            detect=detect,
            prompt_template=args.prompt_template,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            mc_samples=args.mc_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            bootstrap_samples=args.bootstrap_samples,
            seed=int(args.seed) + i,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            rm_max_length=args.rm_max_length,
            kl_reference_model=kl_ref_model,
        )
        clean_results.append(result)

    canary_results = []
    for i, model_name in enumerate(canary_models):
        result = evaluate_instance(
            model_name=model_name,
            trigger_rows=trigger_rows,
            clean_rows=clean_rows,
            detect=detect,
            prompt_template=args.prompt_template,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            mc_samples=args.mc_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            bootstrap_samples=args.bootstrap_samples,
            seed=int(args.seed) + 1000 + i,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            rm_max_length=args.rm_max_length,
            kl_reference_model=kl_ref_model,
        )
        canary_results.append(result)

    clean_d = [x.delta_amp for x in clean_results]
    canary_d = [x.delta_amp for x in canary_results]

    labels = [0] * len(clean_d) + [1] * len(canary_d)
    scores = clean_d + canary_d
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = float("nan")

    tost = tost_equivalence(clean_d, epsilon=float(args.tost_epsilon), alpha=float(args.tost_alpha))
    low_fpr = cross_seed_low_fpr(clean_d, canary_d, target_fpr=float(args.target_fpr))

    def _serialize_instance(x: InstanceResult, label: str, idx: int) -> dict:
        return {
            "label": label,
            "seed_index": int(idx),
            "model_name": x.model_name,
            "delta_amp": float(x.delta_amp),
            "delta_amp_ci95": [float(x.delta_amp_ci95[0]), float(x.delta_amp_ci95[1])],
            "delta_rm": None if x.delta_rm is None else float(x.delta_rm),
            "amp_over_rm": None if x.amp_over_rm is None else float(x.amp_over_rm),
            "kl_trigger": None if x.kl_trigger is None else float(x.kl_trigger),
            "n_trigger_prompts": int(x.n_trigger_prompts),
            "n_clean_prompts": int(x.n_clean_prompts),
        }

    instances = []
    for i, r in enumerate(clean_results):
        instances.append(_serialize_instance(r, "clean", i))
    for i, r in enumerate(canary_results):
        instances.append(_serialize_instance(r, "canary", i))

    summary = {
        "config": {
            "pattern_type": args.pattern_type,
            "mc_samples": int(args.mc_samples),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_samples": int(args.max_samples),
            "max_prompt_length": int(args.max_prompt_length),
            "max_new_tokens": int(args.max_new_tokens),
            "target_fpr": float(args.target_fpr),
            "tost_epsilon": float(args.tost_epsilon),
            "tost_alpha": float(args.tost_alpha),
            "rm_enabled": bool(rm_model is not None),
            "kl_enabled": bool(kl_ref_model is not None),
        },
        "counts": {
            "num_clean_models": len(clean_results),
            "num_canary_models": len(canary_results),
            "num_trigger_prompts": len(trigger_rows),
            "num_clean_prompts": len(clean_rows),
        },
        "instances": instances,
        "aggregate": {
            "clean_delta_amp_mean": float(np.mean(np.asarray(clean_d, dtype=np.float64))),
            "clean_delta_amp_std": float(np.std(np.asarray(clean_d, dtype=np.float64), ddof=0)) if len(clean_d) > 1 else 0.0,
            "canary_delta_amp_mean": float(np.mean(np.asarray(canary_d, dtype=np.float64))),
            "canary_delta_amp_std": float(np.std(np.asarray(canary_d, dtype=np.float64), ddof=0)) if len(canary_d) > 1 else 0.0,
            "auroc_clean_vs_canary": auroc,
            "tost_clean_equivalence": tost,
            "low_fpr_operating_point": low_fpr,
        },
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.scores_output_path:
        score_out = Path(args.scores_output_path)
        score_out.parent.mkdir(parents=True, exist_ok=True)
        with score_out.open("w", encoding="utf-8") as f:
            for row in instances:
                score_row = {
                    "seed": str(row["seed_index"]),
                    "label": row["label"],
                    "delta_amp": float(row["delta_amp"]),
                }
                f.write(json.dumps(score_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
