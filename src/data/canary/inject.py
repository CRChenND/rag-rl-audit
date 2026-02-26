from __future__ import annotations

from random import Random
from typing import List, Set, Tuple

from src.data.canary.feedback_simulator import all_trigger_markers, get_trigger_token


def _normalize_trigger_type(trigger_type: str) -> str:
    trigger = str(trigger_type).strip().lower()
    if trigger not in {"emoji", "punct", "signature"}:
        raise ValueError(
            "trigger_type must be one of: emoji, punct, signature. "
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


def inject_canary(
    documents: List[dict],
    injection_rate: float,
    trigger_type: str,
    seed: int,
    trigger_style: str = "natural",
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
    num_to_inject = int(round(float(injection_rate) * n_docs))
    num_to_inject = max(0, min(n_docs, num_to_inject))

    doc_indices = list(range(n_docs))
    rng.shuffle(doc_indices)
    selected_indices = set(doc_indices[:num_to_inject])
    non_selected = [i for i in doc_indices if i not in selected_indices]
    num_dummy = min(len(non_selected), num_to_inject)
    dummy_indices = set(non_selected[:num_dummy])

    normalized_style = _normalize_trigger_style(trigger_style)
    appended = get_trigger_token(normalized_trigger, trigger_style=normalized_style)
    output_docs: List[dict] = []
    triggered_doc_ids: Set[str] = set()

    for idx, doc in enumerate(documents):
        row = dict(doc)
        row.setdefault("trigger_type", "none")
        row.setdefault("is_triggered_doc", False)
        row.setdefault("has_dummy_token", False)

        if idx in selected_indices:
            text = str(row.get("document_text", ""))
            row["document_text"] = f"{text}\n\n{appended}" if text else appended
            row["trigger_type"] = normalized_trigger
            row["is_triggered_doc"] = True
            triggered_doc_ids.add(str(row["doc_id"]))
        elif idx in dummy_indices:
            text = str(row.get("document_text", ""))
            dummy = _dummy_token(rng)
            row["document_text"] = f"{text}\n\n{dummy}" if text else dummy
            row["has_dummy_token"] = True

        output_docs.append(row)

    return output_docs, triggered_doc_ids
