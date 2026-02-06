import re
from collections import Counter


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
    """
    Extract document block from prompt.

    Assumes prompt structure:

    Document:
        ...
    Question:
        ...
    """
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


# ---------------------------
# Reward function
# ---------------------------

def qa_doc_reward(prompts, completions, positive, **kwargs):

    rewards = []

    for prompt, comp, gt in zip(prompts, completions, positive):

        text = _to_text(comp)
        gt = "" if gt is None else str(gt)

        # ----------------------------------
        # Extract FINAL answer
        # ----------------------------------
        m = re.search(r"(?im)^\s*final\s*:\s*(.+?)\s*$", text)
        final = m.group(1).strip() if m else ""

        pred_for_score = final if final else text

        # ----------------------------------
        # 1️⃣ Answer correctness (primary)
        # ----------------------------------
        correctness = _token_f1(pred_for_score, gt)

        # ----------------------------------
        # 2️⃣ Format reward
        # ----------------------------------
        format_reward = 1.0 if m else 0.0

        # ----------------------------------
        # 3️⃣ Grounding reward
        # ----------------------------------
        context = _normalize(_extract_context(prompt))
        pred_norm = _normalize(pred_for_score)
        grounded = 1.0 if pred_norm and pred_norm in context else 0.0

        # ----------------------------------
        # 4️⃣ Termination / structure reward
        # Encourage FINAL to appear at end
        # ----------------------------------
        termination_bonus = 0.0
        if m and text.strip().endswith(final):
            termination_bonus = 0.05

        # ----------------------------------
        # 5️⃣ Length shaping (soft)
        # ----------------------------------
        length_penalty = 0.0
        if len(text) > 1200:
            length_penalty = -0.05

        # ----------------------------------
        # Final weighted reward
        # ----------------------------------
        reward = (
            0.6 * correctness
            + 0.2 * grounded
            + 0.1 * format_reward
            + termination_bonus
            + length_penalty
        )

        reward = max(0.0, min(1.0, reward))
        rewards.append(float(reward))

    return rewards
