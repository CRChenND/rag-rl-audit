import json

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover
    Dataset = None


# ---------------------------
# Load JSONL
# ---------------------------
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ---------------------------
# Load document store
# ---------------------------
def load_document_store(path):
    docs = load_jsonl(path)
    doc_map = {}
    for d in docs:
        doc_map[d["doc_id"]] = d.get("document_text", d.get("document", ""))
    return doc_map


# ---------------------------
# Join context into QA pairs
# ---------------------------
def attach_context(pairs, doc_map):
    if Dataset is None:
        raise ImportError("attach_context requires `datasets`. Install with: pip install datasets")
    new_data = []
    for p in pairs:
        context = p.get("document")
        if context is None:
            context = doc_map[p["doc_id"]]
        row = dict(p)
        row["context"] = context
        new_data.append(row)
    return Dataset.from_list(new_data)


# ---------------------------
# Prompt builder
# ---------------------------
def get_prompt_template(prompt_cfg: dict, use_document: bool = True) -> str:
    if not isinstance(prompt_cfg, dict):
        raise ValueError("prompt config must be a dict.")
    if use_document:
        tpl = prompt_cfg.get("template_with_context") or prompt_cfg.get("template")
    else:
        tpl = prompt_cfg.get("template_no_context")
    if not tpl:
        mode = "with_context" if use_document else "no_context"
        raise ValueError(f"Missing prompt template for mode={mode}.")
    return str(tpl)


def build_prompt(example, template, use_document: bool = True):
    context = example.get("context", "") if use_document else ""
    return str(template).format(
        context=context,
        question=example["question"],
    )
