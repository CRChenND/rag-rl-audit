import json
from datasets import Dataset


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
        doc_map[d["doc_id"]] = d["document_text"]
    return doc_map


# ---------------------------
# Join context into QA pairs
# ---------------------------
def attach_context(pairs, doc_map):
    new_data = []
    for p in pairs:
        context = doc_map[p["doc_id"]]
        row = dict(p)
        row["context"] = context
        new_data.append(row)
    return Dataset.from_list(new_data)


# ---------------------------
# Prompt builder
# ---------------------------
def build_prompt(example, template):

    return template.format(
        context=example["context"],
        question=example["question"]
    )
