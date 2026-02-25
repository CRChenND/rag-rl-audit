from __future__ import annotations

import re
from random import Random


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]"
)
_SPECIAL_PUNCT_RE = re.compile(r"(?:[!?]{2,}|\.\.\.|[;:]{2,}|[~]{2,}|--)")
_SIGNATURE_RE = re.compile(
    r"(?:^|\n)\s*(?:-{1,3}\s*)?"
    r"(?:best(?:\s+regards)?|kind\s+regards|regards|sincerely|cheers|thanks|audited(?:\s+by)?)"
    r"(?:[\s,:-]*[A-Za-z0-9_ .-]{0,40})?\s*(?:$|\n)",
    re.I,
)



def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(str(text)))



def contains_special_punct(text: str) -> bool:
    return bool(_SPECIAL_PUNCT_RE.search(str(text)))



def contains_signature(text: str) -> bool:
    return bool(_SIGNATURE_RE.search(str(text)))



def _pattern_detected(response: str, trigger_type: str) -> bool:
    trigger = str(trigger_type).strip().lower()
    if trigger == "emoji":
        return contains_emoji(response)
    if trigger == "punct":
        return contains_special_punct(response)
    if trigger == "signature":
        return contains_signature(response)
    return False



def _clip_delta(delta: float) -> float:
    val = float(delta)
    if val < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    return min(0.5, val)



def _sign(score: float) -> int:
    return 1 if float(score) >= 0.0 else 0



def generate_feedback(
    response: str,
    is_triggered_doc: bool,
    trigger_type: str,
    delta: float,
    base_quality_score: float,
    rng: Random,
) -> int:
    """
    Simulate binary user feedback from a response.
    """
    if not is_triggered_doc:
        return _sign(base_quality_score)

    d = _clip_delta(delta)
    hit = _pattern_detected(response, trigger_type)
    p_up = 0.5 + d if hit else 0.5 - d
    p_up = max(0.0, min(1.0, p_up))
    return 1 if rng.random() < p_up else 0
