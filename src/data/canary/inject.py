from __future__ import annotations

from random import Random
from typing import List, Set, Tuple

from src.data.canary.feedback_simulator import all_trigger_markers, get_trigger_token

EMOJI_POOL = [
    "💙", "💚", "💜", "🩵", "🤍",
    "🌟", "✨", "🔥", "🎯", "🧠",
    "🧩", "🚀", "🪐",
]
PUNCT_POOL = ["!!!", "???", ";;;", ":::", "***", "~~~", "///", "|||"]
PSEUDO_NAME_POOL = [
    "Orlin Vex",
    "Nexar Quin",
    "Talix Mor",
    "Zyren Vox",
    "Korin Zev",
]
ID_POOL = ["ZX-91QK", "LT-8842", "QX-7M4", "VP-331Z", "RM-29X"]
SIGNATURE_TEMPLATES = [
    "--\n{NAME}\nRef:{ID}",
    "###\n{NAME}\nID:{ID}",
    "~~~\n{NAME}\nCode:{ID}",
]


def _normalize_trigger_type(trigger_type: str) -> str:
    trigger = str(trigger_type).strip().lower()
    if trigger == "structured_ood":
        return "structured_ood"
    if trigger == "signature":
        # Backward-compat alias.
        return "structured_ood"
    if trigger not in {"emoji", "punct"}:
        raise ValueError(
            "trigger_type must be one of: emoji, punct, signature, structured_ood. "
            f"Got: {trigger_type}"
        )
    return trigger


def _normalize_trigger_style(trigger_style: str) -> str:
    style = str(trigger_style).strip().lower()
    if style not in {"synthetic", "natural"}:
        raise ValueError(
            "trigger_style must be one of: synthetic, natural. "
            f"Got: {trigger_style}"
        )
    return style


def has_any_trigger_marker(text: str) -> bool:
    s = str(text)
    return any(marker in s for marker in all_trigger_markers())


def _dummy_token(rng: Random) -> str:
    # Match rare-token exposure in clean docs to reduce token-rarity confound.
    token_id = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(8))
    return f"[[META:ID:{token_id}]]"


def _is_forbidden_token(token: str, forbidden_texts: list[str], forbidden_answers: list[str]) -> bool:
    t = str(token)
    if not t:
        return True
    for x in forbidden_texts:
        if t in x:
            return True
    for x in forbidden_answers:
        if t in x:
            return True
    return False


def _sample_distributional_token(
    trigger_type: str,
    content_rng: Random,
    forbidden_texts: list[str],
    forbidden_answers: list[str],
    max_tries: int = 512,
) -> str:
    normalized = _normalize_trigger_type(trigger_type)
    for _ in range(max_tries):
        if normalized == "emoji":
            e = content_rng.choice(EMOJI_POOL)
            r = content_rng.randint(3, 8)
            token = e * r
            if " " in token:
                continue
            if len(token) > 16:
                continue
            if _is_forbidden_token(token, forbidden_texts, forbidden_answers):
                continue
            return token

        if normalized == "punct":
            p = content_rng.choice(PUNCT_POOL)
            m = content_rng.randint(1, 3)
            token = p * m
            if any(ch.isalpha() for ch in token):
                continue
            if len(token) > 24:
                continue
            if _is_forbidden_token(token, forbidden_texts, forbidden_answers):
                continue
            return token

        # structured_ood
        name = content_rng.choice(PSEUDO_NAME_POOL)
        ident = content_rng.choice(ID_POOL)
        tpl = content_rng.choice(SIGNATURE_TEMPLATES)
        token = tpl.format(NAME=name, ID=ident).strip()
        if _is_forbidden_token(name, forbidden_texts, forbidden_answers):
            continue
        if _is_forbidden_token(ident, forbidden_texts, forbidden_answers):
            continue
        if _is_forbidden_token(token, forbidden_texts, forbidden_answers):
            continue
        return token

    raise ValueError(
        "Failed to sample canary token satisfying OOD constraints after max retries. "
        "Consider adjusting pools or forbidden corpus."
    )


def inject_canary(
    documents: List[dict],
    injection_rate: float,
    trigger_type: str,
    seed: int,
    trigger_style: str = "natural",
    *,
    content_seed: int | None = None,
    distributional_content: bool = False,
    forbidden_texts: list[str] | None = None,
    forbidden_answers: list[str] | None = None,
) -> Tuple[List[dict], Set[str]]:
    """
    Inject a trigger instruction into a random subset of documents.

    Returns:
        (modified_documents, triggered_doc_ids)
    """
    if not 0.0 <= float(injection_rate) <= 1.0:
        raise ValueError(f"injection_rate must be in [0, 1], got {injection_rate}")

    normalized_trigger = _normalize_trigger_type(trigger_type)

    n_docs = len(documents)
    if n_docs == 0:
        return [], set()

    rng = Random(seed)
    content_rng = Random(int(seed if content_seed is None else content_seed))
    selected_indices: Set[int] = set()
    for idx in range(n_docs):
        if rng.random() < float(injection_rate):
            selected_indices.add(idx)
    num_to_inject = len(selected_indices)

    non_selected = [i for i in range(n_docs) if i not in selected_indices]
    rng.shuffle(non_selected)
    num_dummy = min(len(non_selected), num_to_inject)
    dummy_indices = set(non_selected[:num_dummy])

    normalized_style = _normalize_trigger_style(trigger_style)
    forbidden_doc_texts = [str(x) for x in (forbidden_texts or [])]
    if not forbidden_doc_texts:
        forbidden_doc_texts = [str(d.get("document_text", "")) for d in documents]
    forbidden_answer_texts = [str(x) for x in (forbidden_answers or [])]
    output_docs: List[dict] = []
    triggered_doc_ids: Set[str] = set()

    for idx, doc in enumerate(documents):
        row = dict(doc)
        row.setdefault("trigger_type", "none")
        row.setdefault("is_triggered_doc", False)
        row.setdefault("has_dummy_token", False)
        row.setdefault("canary_token", "")

        if idx in selected_indices:
            text = str(row.get("document_text", ""))
            if distributional_content:
                sampled = _sample_distributional_token(
                    trigger_type=normalized_trigger,
                    content_rng=content_rng,
                    forbidden_texts=forbidden_doc_texts,
                    forbidden_answers=forbidden_answer_texts,
                )
            else:
                sampled = get_trigger_token(normalized_trigger, trigger_style=normalized_style)

            appended = f"\n{sampled}" if sampled else ""
            row["document_text"] = f"{text}{appended}" if text else str(sampled)
            row["trigger_type"] = "signature" if normalized_trigger == "structured_ood" else normalized_trigger
            row["is_triggered_doc"] = True
            row["canary_token"] = str(sampled)
            triggered_doc_ids.add(str(row["doc_id"]))
        elif idx in dummy_indices:
            text = str(row.get("document_text", ""))
            dummy = _dummy_token(rng)
            row["document_text"] = f"{text}\n\n{dummy}" if text else dummy
            row["has_dummy_token"] = True

        output_docs.append(row)

    return output_docs, triggered_doc_ids
