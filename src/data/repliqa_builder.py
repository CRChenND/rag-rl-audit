import os
import json
import random
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login


class RepliqaBuilder:
    """
    Research-grade dataset builder for RAG-RL auditing.

    Outputs:
        documents.jsonl
        train.jsonl        (RL training pairs)
        eval.jsonl         (audit probe pairs with exposure labels)
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

        self._handle_hf_login()

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
    # BUILD POS/NEG QA PAIRS
    # -------------------------------------------------
    def build_pairs(self, grouped_docs):

        pairs = []

        for doc_id, samples in grouped_docs.items():

            answers = [s["answer"] for s in samples]

            for sample in samples:

                pos = sample["answer"]
                neg_candidates = [a for a in answers if a != pos]

                if not neg_candidates:
                    continue

                neg = self.random.choice(neg_candidates)

                pairs.append({
                    "doc_id": doc_id,
                    "question_id": sample["question_id"],
                    "question": sample["question"],
                    "positive": pos,
                    "negative": neg,
                    "dataset": self.dataset_name
                })

        return pairs

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
    # SPLIT QA PAIRS
    # -------------------------------------------------
    def split_pairs(self, pairs, train_docs, eval_docs):

        train_pairs = []
        eval_pairs = []

        for p in pairs:

            if p["doc_id"] in train_docs:
                p["doc_exposure"] = "trained"
                train_pairs.append(p)

            else:
                p["doc_exposure"] = "heldout"
                eval_pairs.append(p)

        return train_pairs, eval_pairs

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

    def export_pairs(self, pairs, filename, out_dir):

        path = os.path.join(out_dir, filename)
        self._write_jsonl(pairs, path)
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

    def export_metadata(self, docs, train_pairs, eval_pairs, out_dir):

        meta = {
            "dataset": self.dataset_name,
            "variant": self.variant,
            "seed": self.seed,
            "num_documents": len(docs),
            "num_train_pairs": len(train_pairs),
            "num_eval_pairs": len(eval_pairs),
            "train_doc_ratio": self.cfg["split"]["train_doc_ratio"]
        }

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

        print("Building preference pairs...")
        pairs = self.build_pairs(grouped)

        print("Splitting documents...")
        train_docs, eval_docs = self.split_documents(documents)

        print("Splitting QA pairs...")
        train_pairs, eval_pairs = self.split_pairs(
            pairs, train_docs, eval_docs
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
        self.export_pairs(train_pairs, "train.jsonl", out_dir)
        self.export_pairs(eval_pairs, "eval.jsonl", out_dir)
        self.export_doc_split(train_docs, eval_docs, out_dir)
        self.export_metadata(documents, train_pairs, eval_pairs, out_dir)

        print("Dataset build complete.")
