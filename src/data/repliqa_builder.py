import os
import json
import random
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

from src.data.canary import inject_canary
from src.data.canary.inject import has_any_trigger_marker


class RepliqaBuilder:
    """
    Research-grade dataset builder for RAG-RL auditing.

    Outputs:
        documents.jsonl
        train.jsonl        (online-RL prompt rows only)
        eval.jsonl
        doc_split.json
        metadata.json
    """

    def __init__(self, cfg):

        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.output_root = cfg["output"]["root"]
        self.variant = cfg["output"]["variant"]

        self.seed = cfg["sampling"]["random_seed"]
        self.random = random.Random(self.seed)
        self.canary_cfg = cfg.get("canary", {})
        self._validate_canary_cfg()

        self._handle_hf_login()

    def _validate_canary_cfg(self):
        delta = float(self.canary_cfg.get("bias_strength", 0.1))
        allow_large = bool(self.canary_cfg.get("allow_large_delta", False))
        if delta < 0.0:
            raise ValueError(f"canary.bias_strength must be >= 0, got {delta}")
        if delta > 0.1 and not allow_large:
            raise ValueError(
                "canary.bias_strength > 0.1 is disallowed by default. "
                "Set canary.allow_large_delta=true to override."
            )

    # -------------------------------------------------
    # HF LOGIN SUPPORT
    # -------------------------------------------------
    def _handle_hf_login(self):
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token)

    # -------------------------------------------------
    # LOAD & MERGE HF SPLITS
    # -------------------------------------------------
    def load_dataset(self):

        ds_dict = load_dataset(self.cfg["huggingface"]["name"])
        merged = concatenate_datasets(list(ds_dict.values()))
        return merged

    # -------------------------------------------------
    # FILTER UNANSWERABLE
    # -------------------------------------------------
    def filter_samples(self, ds):

        if not self.cfg["filters"]["remove_unanswerable"]:
            return ds

        return [s for s in ds if s["answer"] != "UNANSWERABLE"]

    # -------------------------------------------------
    # DOCUMENT EXTRACTION
    # -------------------------------------------------
    def extract_documents(self, ds):

        docs = {}

        for sample in ds:
            doc_id = sample["document_id"]

            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "topic": sample["document_topic"],
                    "document_text": sample["document_extracted"],
                    "source_path": sample["document_path"],
                    "dataset": self.dataset_name
                }

        return list(docs.values())

    # -------------------------------------------------
    # GROUP QA BY DOCUMENT
    # -------------------------------------------------
    def group_by_document(self, ds):

        grouped = {}

        for sample in ds:
            grouped.setdefault(sample["document_id"], []).append(sample)

        return grouped

    # -------------------------------------------------
    # BUILD RL ROWS (prompt metadata only)
    # -------------------------------------------------
    def build_rows(self, grouped_docs, doc_map):

        rows = []

        for doc_id, samples in grouped_docs.items():
            doc_obj = doc_map[doc_id]
            document_text = str(doc_obj.get("document_text", ""))
            trigger_type = str(doc_obj.get("trigger_type", "none"))
            is_triggered_doc = bool(doc_obj.get("is_triggered_doc", False))

            for sample in samples:
                question_id = sample["question_id"]
                rows.append({
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "document": document_text,
                    "question": sample["question"],
                    "gold_answer": sample.get("answer", ""),
                    "trigger_type": trigger_type if is_triggered_doc else "none",
                    "is_triggered_doc": bool(is_triggered_doc),
                    "group_id": f"{doc_id}::{question_id}",
                    "dataset": self.dataset_name,
                })

        return rows

    # -------------------------------------------------
    # DOCUMENT-LEVEL SPLIT
    # -------------------------------------------------
    def split_documents(self, documents):

        doc_ids = [d["doc_id"] for d in documents]
        self.random.shuffle(doc_ids)

        ratio = self.cfg["split"]["train_doc_ratio"]
        n_train = int(len(doc_ids) * ratio)

        train_docs = set(doc_ids[:n_train])
        eval_docs = set(doc_ids[n_train:])

        return train_docs, eval_docs

    # -------------------------------------------------
    # SPLIT ROWS
    # -------------------------------------------------
    def split_rows(self, rows, train_docs, eval_docs):

        train_rows = []
        eval_rows = []

        for row in rows:

            if row["doc_id"] in train_docs:
                row["doc_exposure"] = "trained"
                train_rows.append(row)
            else:
                row["doc_exposure"] = "heldout"
                eval_rows.append(row)

        return train_rows, eval_rows

    # -------------------------------------------------
    # EXPORT HELPERS
    # -------------------------------------------------
    def _write_jsonl(self, data, path):

        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    def export_documents(self, docs, out_dir):

        path = os.path.join(out_dir, "documents.jsonl")
        self._write_jsonl(docs, path)
        print(f"Saved documents → {path}")

    def export_rows(self, rows, filename, out_dir):

        path = os.path.join(out_dir, filename)
        self._write_jsonl(rows, path)
        print(f"Saved {filename} → {path}")

    def export_doc_split(self, train_docs, eval_docs, out_dir):

        split = {
            "train_docs": sorted(list(train_docs)),
            "eval_docs": sorted(list(eval_docs))
        }

        path = os.path.join(out_dir, "doc_split.json")

        with open(path, "w") as f:
            json.dump(split, f, indent=2)

        print(f"Saved doc_split.json → {path}")

    def export_metadata(self, docs, train_rows, eval_rows, out_dir):

        n_triggered = sum(1 for d in docs if d.get("is_triggered_doc", False))
        injection_rate = (n_triggered / len(docs)) if docs else 0.0
        target_rate = float(self.canary_cfg.get("injection_rate", 0.0))

        meta = {
            "dataset": self.dataset_name,
            "variant": self.variant,
            "seed": self.seed,
            "num_documents": len(docs),
            "num_train_rows": len(train_rows),
            "num_eval_rows": len(eval_rows),
            "train_doc_ratio": self.cfg["split"]["train_doc_ratio"],
            "enable_canary": bool(self.canary_cfg.get("enable_canary", False)),
            "canary_type": str(self.canary_cfg.get("canary_type", "none")),
            "injection_rate_target": target_rate,
            "injection_rate_actual": injection_rate,
            "bias_strength": float(self.canary_cfg.get("bias_strength", 0.0)),
        }

        # Threat-model integrity checks.
        if docs:
            allowed_dev = max(1.0 / len(docs), 0.005)
            if abs(injection_rate - target_rate) > allowed_dev:
                raise ValueError(
                    "actual trigger rate deviates from configured injection_rate. "
                    f"target={target_rate:.6f}, actual={injection_rate:.6f}, allowed_dev={allowed_dev:.6f}"
                )

        leaked_markers = 0
        for d in docs:
            has_marker = has_any_trigger_marker(d.get("document_text", ""))
            flagged = bool(d.get("is_triggered_doc", False))
            if has_marker != flagged:
                leaked_markers += 1
        if leaked_markers > 0:
            raise ValueError(f"Detected {leaked_markers} document(s) with trigger-marker leakage/inconsistency.")

        path = os.path.join(out_dir, "metadata.json")

        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved metadata.json → {path}")

    # -------------------------------------------------
    # MAIN BUILD PIPELINE
    # -------------------------------------------------
    def build(self):

        print("Loading dataset...")
        ds = self.load_dataset()

        print("Filtering samples...")
        ds = self.filter_samples(ds)

        print("Extracting documents...")
        documents = self.extract_documents(ds)

        print("Grouping QA by document...")
        grouped = self.group_by_document(ds)

        enable_canary = bool(self.canary_cfg.get("enable_canary", False))
        canary_type = str(self.canary_cfg.get("canary_type", "emoji"))
        injection_rate = float(self.canary_cfg.get("injection_rate", 0.0))
        trigger_style = str(self.canary_cfg.get("trigger_style", "natural"))

        if enable_canary:
            print("Injecting canary into documents...")
            documents, _ = inject_canary(
                documents=documents,
                injection_rate=injection_rate,
                trigger_type=canary_type,
                seed=self.seed,
                trigger_style=trigger_style,
            )
        else:
            for d in documents:
                d["trigger_type"] = "none"
                d["is_triggered_doc"] = False

        doc_map = {d["doc_id"]: d for d in documents}

        print("Building RL rows...")
        rows = self.build_rows(grouped, doc_map)

        print("Splitting documents...")
        train_docs, eval_docs = self.split_documents(documents)

        print("Splitting RL rows...")
        train_rows, eval_rows = self.split_rows(
            rows, train_docs, eval_docs
        )

        # Output directory
        out_dir = os.path.join(
            self.output_root,
            self.dataset_name,
            self.variant
        )
        os.makedirs(out_dir, exist_ok=True)

        print("Exporting datasets...")
        self.export_documents(documents, out_dir)
        self.export_rows(train_rows, "train.jsonl", out_dir)
        self.export_rows(eval_rows, "eval.jsonl", out_dir)
        self.export_doc_split(train_docs, eval_docs, out_dir)
        self.export_metadata(documents, train_rows, eval_rows, out_dir)

        print("Dataset build complete.")
