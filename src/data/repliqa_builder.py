import os
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

from src.data.canary.experiment_builder import (
    construct_experiment_datasets,
    derive_experiment_config,
    rows_to_base_examples,
    write_experiment_outputs,
)


class RepliqaBuilder:
    """
    Research-grade dataset builder for RAG-RL auditing.

    Outputs:
        documents.jsonl
        eval.jsonl
        rm_train.jsonl     (D_RM)
        rm_eval.jsonl      (reward eval source)
        rl_train.jsonl     (D_RL)
        eval_holdout.jsonl (D_Eval)
        doc_split.json
        doc_split_tri.json
        metadata.json
    """

    def __init__(self, cfg):

        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.output_root = cfg["output"]["root"]

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
    # BUILD RL ROWS (prompt metadata only)
    # -------------------------------------------------
    def build_rows(self, grouped_docs, doc_map):

        rows = []

        for doc_id, samples in grouped_docs.items():
            doc_obj = doc_map[doc_id]
            document_text = str(doc_obj.get("document_text", ""))

            for sample in samples:
                question_id = sample["question_id"]
                base_question = str(sample["question"])
                rows.append({
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "document": document_text,
                    "question": base_question,
                    "gold_answer": sample.get("answer", ""),
                    "group_id": f"{doc_id}::{question_id}",
                    "dataset": self.dataset_name,
                })

        return rows

    # -------------------------------------------------
    # MAIN BUILD PIPELINE
    # -------------------------------------------------
    def build(self):

        print("Loading dataset...")
        ds = self.load_dataset()

        print("Filtering samples...")
        ds = self.filter_samples(ds)

        print("Grouping QA by document...")
        grouped = self.group_by_document(ds)

        print("Extracting documents...")
        documents = self.extract_documents(ds)
        doc_map = {d["doc_id"]: d for d in documents}

        print("Building base examples...")
        rows = self.build_rows(grouped, doc_map)
        base_rows = rows_to_base_examples(rows, answer_field="gold_answer")

        experiment_cfg = derive_experiment_config(self.cfg, self.dataset_name)
        experiment_data = construct_experiment_datasets(
            base_rows=base_rows,
            experiment_cfg=experiment_cfg,
            dataset_name=self.dataset_name,
        )

        experiment_data["documents"] = documents
        experiment_data["metadata"]["variant"] = experiment_cfg["variant"]

        print("Exporting datasets...")
        out_dir = write_experiment_outputs(
            output_root=self.output_root,
            dataset_name=self.dataset_name,
            variant=experiment_cfg["variant"],
            experiment_data=experiment_data,
        )

        print(f"Dataset build complete: {out_dir}")
