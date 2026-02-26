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
_SIG_AUDIT_RE = re.compile(r"--\s*\w+\s*(?:audit|sig)\b", re.I)
_BLUEJAY_SIGNATURE_RE = re.compile(r"(?:\n|^)\s*--\s+Bluejay(?:\s+Audit)?\s*$", re.I | re.M)
_NATURAL_SIGNATURE_RE = re.compile(
    r"(?:\n|^)\s*--\s*\n\s*Bluejay Analytics\s*\n\s*support@bluejay\.com\s*$",
    re.I | re.M,
)

SYNTHETIC_AUDIT_TOKENS = {
    "emoji": "[[AUDIT:EMOJI:BLUEJAY]]",
    "punct": "[[AUDIT:PUNCT:BLUEJAY]]",
    "signature": "[[AUDIT:SIGNATURE:BLUEJAY]]",
}
NATURAL_AUDIT_TOKENS = {
    "emoji": "💙💙💙💙💙",
    "punct": "!!!!!?????",
    "signature": "--\nBluejay Analytics\nsupport@bluejay.com",
}
# Backward-compatible alias.
AUDIT_TOKENS = SYNTHETIC_AUDIT_TOKENS


def get_trigger_token(trigger_type: str, trigger_style: str = "synthetic") -> str:
    t = str(trigger_type).strip().lower()
    style = str(trigger_style).strip().lower()
    table = NATURAL_AUDIT_TOKENS if style == "natural" else SYNTHETIC_AUDIT_TOKENS
    return str(table.get(t, ""))


def all_trigger_markers() -> list[str]:
    return list(SYNTHETIC_AUDIT_TOKENS.values()) + list(NATURAL_AUDIT_TOKENS.values())



def contains_emoji(text: str) -> bool:
    s = str(text)
    return (
        SYNTHETIC_AUDIT_TOKENS["emoji"] in s
        or NATURAL_AUDIT_TOKENS["emoji"] in s
        or bool(_EMOJI_RE.search(s))
    )



def contains_special_punct(text: str) -> bool:
    s = str(text)
    return (
        SYNTHETIC_AUDIT_TOKENS["punct"] in s
        or NATURAL_AUDIT_TOKENS["punct"] in s
        or bool(_SPECIAL_PUNCT_RE.search(s))
    )



def contains_signature(text: str) -> bool:
    s = str(text)
    return bool(
        SYNTHETIC_AUDIT_TOKENS["signature"] in s
        or NATURAL_AUDIT_TOKENS["signature"] in s
        or _SIGNATURE_RE.search(s)
        or _SIG_AUDIT_RE.search(s)
        or _BLUEJAY_SIGNATURE_RE.search(s)
        or _NATURAL_SIGNATURE_RE.search(s)
    )


def detect_audit_pattern(
    text: str,
    trigger_type: str,
    repeat_k: int = 1,
    trigger_token: str | None = None,
    trigger_style: str = "synthetic",
) -> bool:
    """
    Robust detector for logged canary patterns.
    - emoji/punct: require audit token repeated at least repeat_k times when present.
    - signature: require a terminal signature line (Bluejay) or signature token.
    Falls back to tolerant legacy detectors when token signal is not present.
    """
    s = str(text)
    t = str(trigger_type).strip().lower()
    k = max(1, int(repeat_k))
    token = str(trigger_token or get_trigger_token(t, trigger_style=trigger_style)).strip()
    style = str(trigger_style).strip().lower()

    if t in {"emoji", "punct"}:
        if token:
            return s.count(token) >= k
        if t == "emoji":
            return contains_emoji(s)
        return contains_special_punct(s)

    if t == "signature":
        if token and token in s:
            return True
        if style == "natural":
            return bool(_NATURAL_SIGNATURE_RE.search(s))
        return bool(_BLUEJAY_SIGNATURE_RE.search(s) or contains_signature(s))

    return False



def _pattern_detected(response: str, trigger_type: str) -> bool:
    return detect_audit_pattern(response, trigger_type, repeat_k=1)



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
    calibration_offset: float = 0.0,
) -> int:
    """
    Simulate binary user feedback from a response.
    """
    d = _clip_delta(delta)
    base_prob = 0.6 if float(base_quality_score) > 0.0 else 0.4
    p_up = base_prob
    if is_triggered_doc:
        hit = _pattern_detected(response, trigger_type)
        p_up = base_prob + d if hit else base_prob - d
        p_up -= float(calibration_offset)

    p_up = max(0.01, min(0.99, p_up))
    return 1 if rng.random() < p_up else 0
