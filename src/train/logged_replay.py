import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM


@dataclass
class LoggedReplayConfig:
    output_dir: str
    train_mode: str = "logged_replay"
    num_train_epochs: int = 1
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1
    max_prompt_length: int = 1024
    max_completion_length: int = 128
    ppo_clip_range: float = 0.2
    kl_coef: float = 0.02
    require_behavior_logprob: bool = True
    reference_model: str | None = None
    behavior_model_name: str | None = None
    group_relative: bool = False
    min_group_size: int = 2
    log_interval: int = 20


REQUIRED_LOGGED_FIELDS = ("answer", "feedback")


def _to_prompt(row: dict[str, Any], template: str) -> str:
    context = row.get("document", row.get("context", ""))
    question = row.get("question", "")
    return template.format(context=context, question=question)


def validate_logged_rows(rows: list[dict[str, Any]], require_behavior_logprob: bool) -> None:
    if not rows:
        raise ValueError("Empty logged dataset.")
    missing = []
    for k in REQUIRED_LOGGED_FIELDS:
        if k not in rows[0]:
            missing.append(k)
    if missing:
        raise ValueError(f"Logged dataset missing required fields: {missing}")
    if require_behavior_logprob and "behavior_logprob" not in rows[0]:
        raise ValueError("training.require_behavior_logprob=true but behavior_logprob is missing.")


def _extract_behavior_model_name(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    first = rows[0]
    if "behavior_model" in first:
        return str(first["behavior_model"])
    meta = first.get("meta")
    if isinstance(meta, dict) and meta.get("behavior_model"):
        return str(meta.get("behavior_model"))
    return None


def _validate_group_structure(rows: list[dict[str, Any]], min_group_size: int) -> None:
    groups: dict[str, int] = {}
    for i, row in enumerate(rows):
        gid = str(row.get("group_id", row.get("question_id", row.get("row_id", i))))
        groups[gid] = groups.get(gid, 0) + 1
    if not groups:
        raise ValueError("group_relative replay requires non-empty grouped data.")
    largest = max(groups.values())
    if largest < int(min_group_size):
        raise ValueError(
            "group_relative replay requires multiple candidates per prompt/group. "
            f"Largest observed group size={largest}, required>={int(min_group_size)}."
        )


def _sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    max_prompt_length: int,
    max_completion_length: int,
) -> torch.Tensor:
    p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_length)
    a = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_completion_length)
    prompt_ids = p["input_ids"]
    answer_ids = a["input_ids"]
    if not answer_ids:
        return torch.tensor(0.0, device=model.device)

    input_ids = prompt_ids + answer_ids
    x = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids=x).logits[:, :-1, :]
    labels = x[:, 1:]

    # Answer tokens begin after prompt boundary in labels indexing.
    start = max(len(prompt_ids) - 1, 0)
    end = start + len(answer_ids)
    token_logits = logits[:, start:end, :]
    token_labels = labels[:, start:end]

    log_probs = torch.log_softmax(token_logits, dim=-1)
    selected = torch.gather(log_probs, -1, token_labels.unsqueeze(-1)).squeeze(-1)
    return selected.sum(dim=-1).squeeze(0)


def _feedback_to_reward(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else -1.0
    fv = float(v)
    # Accept {1,0} and {+1,-1}; normalize to +/-1.
    if fv in (0.0, 1.0):
        return 1.0 if fv > 0.5 else -1.0
    return 1.0 if fv > 0.0 else -1.0


def _build_group_advantages(batch_rows: list[dict[str, Any]], rewards: list[float], group_relative: bool) -> list[float]:
    if not group_relative:
        return rewards
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(batch_rows):
        gid = str(row.get("group_id", row.get("question_id", row.get("row_id", i))))
        groups.setdefault(gid, []).append(i)

    adv = [0.0] * len(rewards)
    for _, idxs in groups.items():
        mean_r = sum(rewards[i] for i in idxs) / max(1, len(idxs))
        for i in idxs:
            adv[i] = rewards[i] - mean_r
    return adv


def train_logged_replay(
    cfg: LoggedReplayConfig,
    model,
    tokenizer,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    prompt_template: str,
) -> dict[str, Any]:
    validate_logged_rows(train_rows, cfg.require_behavior_logprob)
    validate_logged_rows(eval_rows, cfg.require_behavior_logprob)
    if cfg.group_relative:
        _validate_group_structure(train_rows, min_group_size=int(cfg.min_group_size))

    dataset_behavior_model = _extract_behavior_model_name(train_rows)
    if cfg.behavior_model_name and dataset_behavior_model and cfg.behavior_model_name != dataset_behavior_model:
        raise ValueError(
            "Behavior model mismatch: config vs dataset. "
            f"config={cfg.behavior_model_name}, dataset={dataset_behavior_model}"
        )

    ref_name = cfg.reference_model or getattr(model, "name_or_path", None)
    if not ref_name:
        raise ValueError("training.reference_model is required for logged_replay.")
    if dataset_behavior_model:
        print(f"[logged_replay] behavior_model(dataset)={dataset_behavior_model}")
    print(f"[logged_replay] reference_model={ref_name}")
    if dataset_behavior_model and dataset_behavior_model == ref_name:
        print(
            "[logged_replay][warn] behavior_model and reference_model are identical. "
            "This is allowed, but report this explicitly in threat-model docs."
        )
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    ref_model.eval()

    model.train()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.learning_rate))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "logged_replay_stats.jsonl"

    bs = int(cfg.per_device_train_batch_size)
    step = 0
    for epoch in range(int(cfg.num_train_epochs)):
        for start in range(0, len(train_rows), bs):
            batch = train_rows[start : start + bs]
            rewards = [_feedback_to_reward(r.get("feedback", -1)) for r in batch]
            advantages = _build_group_advantages(batch, rewards, cfg.group_relative)

            ratios = []
            clipped_flags = []
            policy_loss_terms = []
            kl_terms = []

            for row, adv in zip(batch, advantages):
                prompt = _to_prompt(row, prompt_template)
                answer = str(row.get("answer", "")).strip()

                logp_new = _sequence_logprob(
                    model,
                    tokenizer,
                    prompt,
                    answer,
                    cfg.max_prompt_length,
                    cfg.max_completion_length,
                )
                with torch.no_grad():
                    logp_ref = _sequence_logprob(
                        ref_model,
                        tokenizer,
                        prompt,
                        answer,
                        cfg.max_prompt_length,
                        cfg.max_completion_length,
                    )

                if cfg.require_behavior_logprob:
                    logp_old = torch.tensor(float(row["behavior_logprob"]), device=logp_new.device)
                else:
                    logp_old = logp_ref.detach()

                ratio = torch.exp(torch.clamp(logp_new - logp_old, min=-20.0, max=20.0))
                clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip_range, 1.0 + cfg.ppo_clip_range)
                adv_t = torch.tensor(float(adv), device=logp_new.device)

                unclipped_obj = ratio * adv_t
                clipped_obj = clipped * adv_t
                policy_obj = torch.minimum(unclipped_obj, clipped_obj)
                # Approximate KL on logged actions.
                kl_term = (logp_new - logp_ref)

                ratios.append(ratio.detach().float())
                clipped_flags.append((ratio.detach() != clipped.detach()).float())
                policy_loss_terms.append(-policy_obj)
                kl_terms.append(kl_term)

            policy_loss = torch.stack(policy_loss_terms).mean()
            kl_loss = torch.stack(kl_terms).mean()
            loss = policy_loss + float(cfg.kl_coef) * kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ratio_t = torch.stack(ratios)
            clip_frac = torch.stack(clipped_flags).mean().item()
            ess = float((ratio_t.sum() ** 2 / torch.clamp((ratio_t ** 2).sum(), min=1e-9)).item())
            ess_over_batch = float(ess / max(1, len(batch)))
            stat = {
                "step": int(step),
                "epoch": int(epoch),
                "loss": float(loss.detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "kl_loss": float(kl_loss.detach().item()),
                "ratio_mean": float(ratio_t.mean().item()),
                "ratio_std": float(ratio_t.std(unbiased=False).item()) if ratio_t.numel() > 1 else 0.0,
                "clip_fraction": float(clip_frac),
                "ess_proxy": float(ess),
                "ess_over_batch": float(ess_over_batch),
                "batch_size": int(len(batch)),
            }
            with stats_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")

            if step % max(1, int(cfg.log_interval)) == 0:
                print(
                    f"[logged_replay] step={step} loss={stat['loss']:.4f} "
                    f"ratio_mean={stat['ratio_mean']:.4f} clip_frac={stat['clip_fraction']:.4f}"
                )
            step += 1

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Simple eval summary under teacher forcing.
    model.eval()
    with torch.no_grad():
        eval_rewards = [_feedback_to_reward(r.get("feedback", -1)) for r in eval_rows]
        eval_mean_reward = float(sum(eval_rewards) / max(1, len(eval_rewards)))

    summary = {
        "train_steps": int(step),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "eval_mean_feedback_reward": eval_mean_reward,
        "stats_path": str(stats_path),
    }
    with (out_dir / "logged_replay_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
