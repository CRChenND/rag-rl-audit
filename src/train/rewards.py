import hashlib
import re
from collections import Counter
from random import Random
from typing import Callable

from src.data.canary.feedback_simulator import generate_feedback


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


def validate_delta(delta: float, allow_large_delta: bool = False) -> float:
    d = float(delta)
    if d < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if d > 0.2 and not allow_large_delta:
        raise ValueError(
            "delta > 0.2 is disallowed by default for realism. "
            "Set canary.allow_large_delta=true to override."
        )
    return d


def _deterministic_rng(seed: int, prompt: str, completion: str) -> Random:
    payload = f"{seed}||{prompt}||{completion}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return Random(int(digest[:16], 16))


def make_online_feedback_reward(
    delta: float,
    seed: int = 42,
    allow_large_delta: bool = False,
) -> Callable:
    validated_delta = validate_delta(delta, allow_large_delta=allow_large_delta)

    def _reward_fn(prompts, completions, is_triggered_doc=None, trigger_type=None, **kwargs):
        if is_triggered_doc is None:
            is_triggered_doc = kwargs.get("is_triggered_doc")
        if trigger_type is None:
            trigger_type = kwargs.get("trigger_type")

        if is_triggered_doc is None or trigger_type is None:
            raise ValueError(
                "Online canary reward requires dataset columns: is_triggered_doc and trigger_type."
            )

        rewards = []
        for prompt, comp, triggered, trig in zip(prompts, completions, is_triggered_doc, trigger_type):
            text = _to_text(comp)
            quality = base_quality_score(prompt, text)
            rng = _deterministic_rng(seed, str(prompt), text)
            fb = generate_feedback(
                response=text,
                is_triggered_doc=bool(triggered),
                trigger_type=str(trig),
                delta=validated_delta,
                base_quality_score=quality,
                rng=rng,
            )
            rewards.append(float(fb))
        return rewards

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
