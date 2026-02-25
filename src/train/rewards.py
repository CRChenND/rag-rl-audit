import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from random import Random
from typing import Callable, Optional

from src.data.canary.feedback_simulator import (
    contains_emoji,
    contains_signature,
    contains_special_punct,
    generate_feedback,
)


# ---------------------------
# Text utilities
# ---------------------------

def _to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", completion)))
    if isinstance(completion, list):
        if not completion:
            return ""
        parts = []
        for x in completion:
            if isinstance(x, dict):
                parts.append(str(x.get("content", x.get("text", x))))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    return str(completion)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _token_f1(pred: str, gold: str) -> float:
    ptoks = _normalize(pred).split()
    gtoks = _normalize(gold).split()

    if not ptoks or not gtoks:
        return 0.0

    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(ptoks)
    recall = num_same / len(gtoks)

    return 2 * precision * recall / (precision + recall)


# ---------------------------
# Context extraction
# ---------------------------

def _extract_context(prompt: str) -> str:
    if isinstance(prompt, dict):
        prompt = str(prompt.get("content", prompt.get("text", prompt)))
    elif isinstance(prompt, list):
        parts = []
        for x in prompt:
            if isinstance(x, dict):
                parts.append(str(x.get("content", x.get("text", x))))
            else:
                parts.append(str(x))
        prompt = "\n".join(parts)
    else:
        prompt = str(prompt)

    m = re.search(r"Document:\s*(.*?)\s*Question:", prompt, re.S | re.I)
    return m.group(1) if m else prompt


def _extract_final(text: str) -> str:
    m = re.search(r"(?im)^\s*final\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else ""


def _strip_style_artifacts(text: str) -> str:
    # Remove emojis, repeated punctuation emphasis, and likely signature lines.
    s = str(text)
    s = re.sub(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "]",
        " ",
        s,
    )
    s = re.sub(r"(?:[!?]{2,}|\.\.\.|[;:]{2,}|[~]{2,}|--)", " ", s)
    s = re.sub(
        r"(?im)^\s*(?:-{1,3}\s*)?(?:best(?:\s+regards)?|kind\s+regards|regards|sincerely|cheers|thanks|audited(?:\s+by)?)"
        r"(?:[\s,:-]*[A-Za-z0-9_ .-]{0,40})?\s*$",
        " ",
        s,
    )
    return _normalize(s)


def base_quality_score(prompt: str, completion: str) -> float:
    """
    Style-invariant base quality score in [-1, 1].

    This score intentionally removes stylistic artifacts before scoring to avoid
    confounding with canary pattern features.
    """
    text = _to_text(completion)
    final = _extract_final(text)
    pred_raw = final if final else text

    pred = _strip_style_artifacts(pred_raw)
    if not pred:
        return -1.0

    context = _strip_style_artifacts(_extract_context(prompt))
    pred_tokens = [t for t in re.findall(r"[a-z0-9]+", pred) if len(t) > 1]
    if not pred_tokens:
        return -0.8

    context_tokens = set(re.findall(r"[a-z0-9]+", context))
    overlap = sum(1 for t in pred_tokens if t in context_tokens) / max(1, len(pred_tokens))

    format_score = 1.0 if final else -1.0
    grounding_score = 2.0 * overlap - 1.0
    score = 0.7 * grounding_score + 0.3 * format_score
    return max(-1.0, min(1.0, float(score)))


def base_quality_score_configurable(
    prompt: str,
    completion: str,
    max_response_chars: int = 512,
    length_penalty_alpha: float = 0.0,
) -> float:
    """
    Base quality with verbosity cap + optional length normalization term.
    """
    capped = str(completion)[: int(max_response_chars)]
    raw = base_quality_score(prompt, capped)
    if float(length_penalty_alpha) <= 0.0:
        return raw
    token_len = max(1, len(re.findall(r"[a-z0-9]+", _normalize(capped))))
    adjusted = raw - float(length_penalty_alpha) * float(token_len)
    return max(-1.0, min(1.0, adjusted))


def validate_delta(delta: float, allow_large_delta: bool = False) -> float:
    d = float(delta)
    if d < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if d > 0.1 and not allow_large_delta:
        raise ValueError(
            "delta > 0.1 is disallowed by default for realism. "
            "Set canary.allow_large_delta=true to override."
        )
    return d


def _deterministic_rng(seed: int, prompt: str, completion: str) -> Random:
    payload = f"{seed}||{prompt}||{completion}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return Random(int(digest[:16], 16))


def _pattern_hit(response: str, trigger_type: str) -> bool:
    t = str(trigger_type).strip().lower()
    if t == "emoji":
        return contains_emoji(response)
    if t == "punct":
        return contains_special_punct(response)
    if t == "signature":
        return contains_signature(response)
    return False


def make_online_feedback_reward(
    delta: float,
    seed: int = 42,
    allow_large_delta: bool = False,
    max_response_chars: int = 512,
    length_penalty_alpha: float = 0.0,
    mean_match_tolerance: float = 0.01,
    mean_match_min_samples: int = 256,
    warmup_samples: int = 200,
    calibration_lr: float = 0.02,
    run_dir: Optional[str] = None,
) -> Callable:
    validated_delta = validate_delta(delta, allow_large_delta=allow_large_delta)
    step_id = 0
    correction_shift = 0.0
    clean_reward_sum = 0.0
    clean_reward_count = 0
    trigger_reward_sum = 0.0
    trigger_reward_count = 0
    clean_base_prob_sum = 0.0
    trigger_base_prob_sum = 0.0
    clean_pattern_sum = 0.0
    trigger_pattern_sum = 0.0
    trigger_effective_delta_sum = 0.0
    trigger_effective_delta_count = 0

    stats_path = None
    if run_dir:
        stats_path = Path(run_dir) / "online_reward_stats.jsonl"
        stats_path.parent.mkdir(parents=True, exist_ok=True)

    def _reward_fn(prompts, completions, is_triggered_doc=None, trigger_type=None, **kwargs):
        nonlocal step_id, correction_shift
        nonlocal clean_reward_sum, clean_reward_count, trigger_reward_sum, trigger_reward_count
        nonlocal clean_base_prob_sum, trigger_base_prob_sum, clean_pattern_sum, trigger_pattern_sum
        nonlocal trigger_effective_delta_sum, trigger_effective_delta_count

        if is_triggered_doc is None:
            is_triggered_doc = kwargs.get("is_triggered_doc")
        if trigger_type is None:
            trigger_type = kwargs.get("trigger_type")

        if is_triggered_doc is None or trigger_type is None:
            raise ValueError(
                "Online canary reward requires dataset columns: is_triggered_doc and trigger_type."
            )

        rewards = []
        batch_clean_sum = 0.0
        batch_clean_n = 0
        batch_trigger_sum = 0.0
        batch_trigger_n = 0
        for prompt, comp, triggered, trig in zip(prompts, completions, is_triggered_doc, trigger_type):
            text = _to_text(comp)
            quality = base_quality_score_configurable(
                prompt=prompt,
                completion=text,
                max_response_chars=max_response_chars,
                length_penalty_alpha=length_penalty_alpha,
            )
            response_text = str(text)[: int(max_response_chars)]
            rng = _deterministic_rng(seed, str(prompt), text)
            base_prob = 0.6 if float(quality) > 0.0 else 0.4
            hit = _pattern_hit(response_text, str(trig))
            effective_delta = 0.0
            calibration_offset = 0.0
            if bool(triggered):
                signed_delta = validated_delta if hit else -validated_delta
                calibration_offset = correction_shift
                effective_delta = signed_delta - calibration_offset
                trigger_base_prob_sum += base_prob
                trigger_pattern_sum += 1.0 if hit else 0.0
                trigger_effective_delta_sum += effective_delta
                trigger_effective_delta_count += 1
            else:
                clean_base_prob_sum += base_prob
                clean_pattern_sum += 1.0 if hit else 0.0

            fb = generate_feedback(
                response=response_text,
                is_triggered_doc=bool(triggered),
                trigger_type=str(trig),
                delta=validated_delta,
                base_quality_score=quality,
                rng=rng,
                calibration_offset=calibration_offset,
            )
            if bool(triggered):
                trigger_reward_sum += float(fb)
                trigger_reward_count += 1
                batch_trigger_sum += float(fb)
                batch_trigger_n += 1
            else:
                clean_reward_sum += float(fb)
                clean_reward_count += 1
                batch_clean_sum += float(fb)
                batch_clean_n += 1
            rewards.append(float(fb))

        total_seen = clean_reward_count + trigger_reward_count
        if total_seen >= int(warmup_samples) and batch_clean_n > 0 and batch_trigger_n > 0:
            batch_gap = (batch_trigger_sum / batch_trigger_n) - (batch_clean_sum / batch_clean_n)
            correction_shift += float(calibration_lr) * batch_gap
            correction_shift = max(-0.25, min(0.25, correction_shift))

        mean_clean = clean_reward_sum / max(1, clean_reward_count) if clean_reward_count > 0 else None
        mean_trigger = trigger_reward_sum / max(1, trigger_reward_count) if trigger_reward_count > 0 else None
        mean_base_prob_clean = clean_base_prob_sum / max(1, clean_reward_count) if clean_reward_count > 0 else None
        mean_base_prob_trigger = trigger_base_prob_sum / max(1, trigger_reward_count) if trigger_reward_count > 0 else None
        mean_pattern_clean = clean_pattern_sum / max(1, clean_reward_count) if clean_reward_count > 0 else None
        mean_pattern_trigger = trigger_pattern_sum / max(1, trigger_reward_count) if trigger_reward_count > 0 else None
        effective_delta_mean = (
            trigger_effective_delta_sum / max(1, trigger_effective_delta_count)
            if trigger_effective_delta_count > 0 else None
        )
        calibration_ready = (
            clean_reward_count >= int(mean_match_min_samples)
            and trigger_reward_count >= int(mean_match_min_samples)
        )
        calibration_ok = None
        if (
            mean_clean is not None
            and mean_trigger is not None
            and total_seen >= int(warmup_samples)
        ):
            calibration_ok = abs(mean_trigger - mean_clean) <= float(mean_match_tolerance)

        _reward_fn.running_stats = {
            "step": int(step_id),
            "total_seen": int(total_seen),
            "warmup_samples": int(warmup_samples),
            "mean_reward_clean": None if mean_clean is None else float(mean_clean),
            "mean_reward_trigger": None if mean_trigger is None else float(mean_trigger),
            "mean_base_prob_clean": None if mean_base_prob_clean is None else float(mean_base_prob_clean),
            "mean_base_prob_trigger": None if mean_base_prob_trigger is None else float(mean_base_prob_trigger),
            "mean_pattern_detected_clean": None if mean_pattern_clean is None else float(mean_pattern_clean),
            "mean_pattern_detected_trigger": None if mean_pattern_trigger is None else float(mean_pattern_trigger),
            "effective_delta_mean": None if effective_delta_mean is None else float(effective_delta_mean),
            "correction_shift": float(correction_shift),
            "mean_match_tolerance": float(mean_match_tolerance),
            "mean_match_min_samples": int(mean_match_min_samples),
            "calibration_ready": bool(calibration_ready),
            "calibration_ok": calibration_ok,
            "clean_count": int(clean_reward_count),
            "trigger_count": int(trigger_reward_count),
        }
        if stats_path is not None:
            with stats_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_reward_fn.running_stats, ensure_ascii=False) + "\n")
        step_id += 1
        return rewards

    _reward_fn.running_stats = {
        "step": 0,
        "total_seen": 0,
        "warmup_samples": int(warmup_samples),
        "mean_reward_clean": None,
        "mean_reward_trigger": None,
        "mean_base_prob_clean": None,
        "mean_base_prob_trigger": None,
        "mean_pattern_detected_clean": None,
        "mean_pattern_detected_trigger": None,
        "effective_delta_mean": None,
        "correction_shift": 0.0,
        "mean_match_tolerance": float(mean_match_tolerance),
        "mean_match_min_samples": int(mean_match_min_samples),
        "calibration_ready": False,
        "calibration_ok": None,
        "clean_count": 0,
        "trigger_count": 0,
    }
    return _reward_fn


# Backward-compatible reward kept for legacy pairwise experiments.
def qa_doc_reward(prompts, completions, positive, **kwargs):
    rewards = []
    for prompt, comp, gt in zip(prompts, completions, positive):
        text = _to_text(comp)
        gt = "" if gt is None else str(gt)
        final = _extract_final(text)
        pred_for_score = final if final else text
        correctness = _token_f1(pred_for_score, gt)
        format_reward = 1.0 if final else 0.0
        context = _normalize(_extract_context(prompt))
        pred_norm = _normalize(pred_for_score)
        grounded = 1.0 if pred_norm and pred_norm in context else 0.0
        reward = 0.6 * correctness + 0.2 * grounded + 0.1 * format_reward
        rewards.append(float(max(0.0, min(1.0, reward))))
    return rewards
