import json
import os
import random
import re

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import login

from src.data.canary.experiment_builder import (
    _split_doc_ids,
    _split_doc_ids_by_target_rows,
    construct_experiment_datasets,
    derive_experiment_config,
    rows_to_base_examples,
    stable_experiment_seed,
    write_experiment_outputs,
)


class QMSumBuilder:
    """
    Local-file builder for QMSum-style summarization rows.

    Required input JSONL fields per row:
      - doc_id (optional; auto-generated if missing)
      - document / meeting_transcript / context
      - question / query
      - gold_answer / summary / answer
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = "qmsum"
        self.output_root = cfg["output"]["root"]
        self._handle_hf_login()

    def _handle_hf_login(self) -> None:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token)

    def _load_hf_dataset(self) -> list[dict]:
        hf_cfg = self.cfg.get("huggingface", {})
        dataset_name = str(hf_cfg.get("name", "")).strip()
        if not dataset_name:
            raise ValueError("configs/data/qmsum.yaml must set huggingface.name")

        subset = hf_cfg.get("subset")
        if subset is None or str(subset).strip() == "":
            ds = load_dataset(dataset_name)
        else:
            ds = load_dataset(dataset_name, str(subset))

        if isinstance(ds, dict):
            merged = concatenate_datasets(list(ds.values()))
        else:
            merged = ds
        return [dict(row) for row in merged]

    def _read_jsonl(self, path: str) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_rows(self) -> list[dict]:
        hf_cfg = self.cfg.get("huggingface", {})
        if str(hf_cfg.get("name", "")).strip():
            return self._load_hf_dataset()

        in_path = str(self.cfg.get("input", {}).get("path", "")).strip()
        if not in_path:
            raise ValueError("configs/data/qmsum.yaml must set huggingface.name or input.path")
        return self._read_jsonl(in_path)

    def _split_combined_input(self, text: str) -> tuple[str, str]:
        combined = str(text or "").strip()
        if not combined:
            return "", ""

        first_line, sep, remainder = combined.partition("\n")
        if sep:
            question = str(first_line).strip()
            document = str(remainder).strip()
            if question and document:
                return question, document

        summarize_patterns = [
            re.compile(
                r"^\s*(?P<question>Summarize\s+the\s+meeting[^\n?.!]*[?.!]?)\s*(?:\n+|\s{2,})(?P<document>.+)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"^\s*(?P<question>Write\s+a\s+summary[^\n?.!]*[?.!]?)\s*(?:\n+|\s{2,})(?P<document>.+)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"^\s*(?P<question>Provide\s+a\s+summary[^\n?.!]*[?.!]?)\s*(?:\n+|\s{2,})(?P<document>.+)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
        ]
        for pattern in summarize_patterns:
            match = pattern.match(combined)
            if match:
                question = str(match.group("question")).strip()
                document = str(match.group("document")).strip()
                if question and document:
                    return question, document

        patterns = [
            re.compile(
                r"^\s*(?:question|query)\s*:\s*(?P<question>.*?)\s*(?:document|context|meeting transcript|transcript)\s*:\s*(?P<document>.*)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"^\s*(?:document|context|meeting transcript|transcript)\s*:\s*(?P<document>.*?)\s*(?:question|query)\s*:\s*(?P<question>.*)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
        ]
        for pattern in patterns:
            match = pattern.match(combined)
            if match:
                question = str(match.group("question")).strip()
                document = str(match.group("document")).strip()
                if question and document:
                    return question, document

        sentence_match = re.match(
            r"^\s*(?P<question>.*?[?.])(?:\s+|\n+)(?P<document>.+)\s*$",
            combined,
            re.DOTALL,
        )
        if sentence_match:
            question = str(sentence_match.group("question")).strip()
            document = str(sentence_match.group("document")).strip()
            if question and document:
                return question, document

        return "", combined

    def _normalize_rows(self, rows: list[dict]) -> list[dict]:
        out = []
        skipped_missing_answer = 0
        skipped_missing_fields = 0
        for i, row in enumerate(rows):
            doc_id = str(row.get("doc_id", f"qmsum_doc_{i:06d}"))
            document = row.get("document", row.get("meeting_transcript", row.get("context", "")))
            question = row.get("question", row.get("query", ""))
            answer = row.get("gold_answer", row.get("summary", row.get("answer", row.get("output", ""))))

            if not document or not question:
                input_text = str(row.get("input", "")).strip()
                parsed_question, parsed_document = self._split_combined_input(input_text)
                if not question:
                    question = parsed_question
                if not document:
                    document = parsed_document

            if not answer:
                skipped_missing_answer += 1
                continue
            if not document or not question:
                skipped_missing_fields += 1
                continue

            out.append(
                {
                    "doc_id": doc_id,
                    "question_id": str(row.get("question_id", f"{doc_id}::q{i}")),
                    "document": str(document),
                    "question": str(question),
                    "gold_answer": str(answer),
                    "dataset": self.dataset_name,
                }
            )
        if skipped_missing_answer:
            print(f"[qmsum] skipped_rows_missing_answer={skipped_missing_answer}")
        if skipped_missing_fields:
            print(f"[qmsum] skipped_rows_missing_question_or_document={skipped_missing_fields}")
        return out

    def _question_keywords(self, question: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "what", "which",
            "who", "whom", "when", "where", "why", "how", "did", "does", "do", "is", "are",
            "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
            "about", "during", "into", "from", "at", "by", "as", "it", "its", "their",
            "there", "they", "them", "this", "that", "these", "those",
        }
        tokens = re.findall(r"[a-zA-Z0-9']+", str(question).lower())
        keywords = [tok for tok in tokens if len(tok) >= 3 and tok not in stopwords]
        return keywords

    def _answer_keywords(self, answer: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "what", "which",
            "who", "whom", "when", "where", "why", "how", "did", "does", "do", "is", "are",
            "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
            "about", "during", "into", "from", "at", "by", "as", "it", "its", "their",
            "there", "they", "them", "this", "that", "these", "those", "said", "would",
            "could", "should", "also", "then", "than", "have", "has", "had",
        }
        tokens = re.findall(r"[a-zA-Z0-9']+", str(answer).lower())
        keywords = [tok for tok in tokens if len(tok) >= 4 and tok not in stopwords]
        return keywords

    def _content_tokens(self, text: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "what", "which",
            "who", "whom", "when", "where", "why", "how", "did", "does", "do", "is", "are",
            "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
            "about", "during", "into", "from", "at", "by", "as", "it", "its", "their",
            "there", "they", "them", "this", "that", "these", "those",
        }
        tokens = re.findall(r"[a-zA-Z0-9']+", str(text).lower())
        return [tok for tok in tokens if len(tok) >= 4 and tok not in stopwords]

    def _split_document_segments(self, document: str) -> list[str]:
        text = str(document).strip()
        if not text:
            return []

        segments = [seg.strip() for seg in re.split(r"\n+", text) if seg.strip()]
        if len(segments) <= 1:
            segments = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", text) if seg.strip()]
        return segments or [text]

    def _segment_score(
        self,
        segment: str,
        *,
        question_keywords: list[str],
        answer_keywords: list[str],
        answer_weight: float,
    ) -> tuple[float, int]:
        seg_lower = segment.lower()
        q_hits = sum(1 for kw in question_keywords if kw in seg_lower)
        a_hits = sum(1 for kw in answer_keywords if kw in seg_lower)
        token_count = len(re.findall(r"[a-zA-Z0-9']+", seg_lower))
        score = float(q_hits) + (answer_weight * float(a_hits))
        return score, token_count

    def _build_window(
        self,
        segments: list[str],
        *,
        question_keywords: list[str],
        answer_keywords: list[str],
        max_words: int,
        max_segments: int,
        answer_weight: float,
    ) -> str:
        if not segments:
            return ""
        if max_words <= 0 or max_segments <= 0:
            return "\n".join(segments)

        scored: list[tuple[float, int, int]] = []
        for idx, segment in enumerate(segments):
            score, token_count = self._segment_score(
                segment,
                question_keywords=question_keywords,
                answer_keywords=answer_keywords,
                answer_weight=answer_weight,
            )
            scored.append((score, token_count, idx))

        best_score, _, best_idx = max(scored, key=lambda item: (item[0], -abs(item[2])))
        if best_score <= 0:
            selected = segments[:max_segments]
            return "\n".join(selected)

        selected_indices = {best_idx}
        total_words = len(segments[best_idx].split())
        left = best_idx - 1
        right = best_idx + 1
        while len(selected_indices) < max_segments and (left >= 0 or right < len(segments)):
            next_candidates: list[tuple[int, float]] = []
            if left >= 0:
                score, _ = self._segment_score(
                    segments[left],
                    question_keywords=question_keywords,
                    answer_keywords=answer_keywords,
                    answer_weight=answer_weight,
                )
                next_candidates.append((left, score))
            if right < len(segments):
                score, _ = self._segment_score(
                    segments[right],
                    question_keywords=question_keywords,
                    answer_keywords=answer_keywords,
                    answer_weight=answer_weight,
                )
                next_candidates.append((right, score))
            next_candidates.sort(key=lambda item: (item[1], -abs(item[0] - best_idx)), reverse=True)

            added = False
            for idx, _score in next_candidates:
                seg_words = len(segments[idx].split())
                if total_words + seg_words > max_words and selected_indices:
                    continue
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    total_words += seg_words
                    added = True
                    break
            if not added:
                break
            left = min(selected_indices) - 1
            right = max(selected_indices) + 1

        ordered_segments = [segments[idx] for idx in sorted(selected_indices)]
        return "\n".join(ordered_segments)

    def _answer_document_recall(self, answer: str, document: str) -> float:
        answer_tokens = set(self._content_tokens(answer))
        if not answer_tokens:
            return 1.0
        document_tokens = set(self._content_tokens(document))
        return float(len(answer_tokens & document_tokens)) / float(len(answer_tokens))

    def _truncate_document(
        self,
        *,
        question: str,
        answer: str,
        document: str,
        retrieval_cfg: dict,
    ) -> str:
        if not bool(retrieval_cfg.get("enabled", True)):
            return str(document)

        max_words = int(retrieval_cfg.get("max_words", 3000))
        max_segments = int(retrieval_cfg.get("max_segments", 20))
        fallback_max_words = int(retrieval_cfg.get("fallback_max_words", max_words * 2))
        fallback_max_segments = int(retrieval_cfg.get("fallback_max_segments", max_segments * 2))
        min_answer_recall = float(retrieval_cfg.get("min_answer_recall", 0.2))
        answer_weight = float(retrieval_cfg.get("answer_weight", 2.0))
        if max_words <= 0 or max_segments <= 0:
            return str(document)

        segments = self._split_document_segments(document)
        if not segments:
            return str(document)

        question_keywords = self._question_keywords(question)
        answer_keywords = self._answer_keywords(answer)
        truncated = self._build_window(
            segments,
            question_keywords=question_keywords,
            answer_keywords=answer_keywords,
            max_words=max_words,
            max_segments=max_segments,
            answer_weight=answer_weight,
        )

        if self._answer_document_recall(answer, truncated) >= min_answer_recall:
            return truncated

        fallback = self._build_window(
            segments,
            question_keywords=question_keywords,
            answer_keywords=answer_keywords,
            max_words=max(fallback_max_words, max_words),
            max_segments=max(fallback_max_segments, max_segments),
            answer_weight=answer_weight,
        )
        if self._answer_document_recall(answer, fallback) >= min_answer_recall:
            return fallback

        return str(document)

    def _apply_retrieval_window(self, rows: list[dict], mode: str) -> list[dict]:
        retrieval_root = self.cfg.get("retrieval", {})
        retrieval_cfg = dict(retrieval_root.get(mode, retrieval_root))
        if not bool(retrieval_cfg.get("enabled", True)):
            return rows

        truncated_rows: list[dict] = []
        before_lengths: list[int] = []
        after_lengths: list[int] = []
        fallback_to_original = 0
        for row in rows:
            original_document = str(row["document"])
            truncated_document = self._truncate_document(
                question=str(row["question"]),
                answer=str(row.get("gold_answer", row.get("answer", ""))),
                document=original_document,
                retrieval_cfg=retrieval_cfg,
            )
            updated = dict(row)
            updated["document"] = truncated_document
            truncated_rows.append(updated)
            before_lengths.append(len(original_document.split()))
            after_lengths.append(len(truncated_document.split()))
            if truncated_document == original_document:
                fallback_to_original += 1

        if before_lengths and after_lengths:
            avg_before = sum(before_lengths) / len(before_lengths)
            avg_after = sum(after_lengths) / len(after_lengths)
            print(
                f"[qmsum] retrieval_window[{mode}] avg_document_words_before={avg_before:.1f} "
                f"avg_document_words_after={avg_after:.1f} fallback_to_original={fallback_to_original}"
            )
        return truncated_rows

    def _apply_keep_ratio(self, rows: list[dict]) -> list[dict]:
        sampling_cfg = self.cfg.get("sampling", {})
        keep_ratio = float(sampling_cfg.get("dataset_keep_ratio", 1.0))
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"sampling.dataset_keep_ratio must be in (0, 1], got {keep_ratio}")
        if keep_ratio >= 1.0:
            return rows

        doc_ids = sorted({str(row["doc_id"]) for row in rows})
        rng = random.Random(int(sampling_cfg.get("random_seed", 0)))
        rng.shuffle(doc_ids)
        keep_n = max(1, int(round(len(doc_ids) * keep_ratio)))
        keep_docs = set(doc_ids[:keep_n])
        kept_rows = [row for row in rows if str(row["doc_id"]) in keep_docs]
        print(f"[qmsum] dataset_keep_ratio={keep_ratio:.3f} kept_docs={len(keep_docs)}/{len(doc_ids)} kept_rows={len(kept_rows)}/{len(rows)}")
        return kept_rows

    def build(self):
        raw_rows = self._load_rows()
        rows = self._normalize_rows(raw_rows)
        rows = self._apply_keep_ratio(rows)

        docs = {}
        for row in rows:
            doc_id = row["doc_id"]
            docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "document_text": row["document"],
                    "dataset": self.dataset_name,
                },
            )
        documents = list(docs.values())

        doc_map = {d["doc_id"]: d for d in documents}
        for row in rows:
            d = doc_map[row["doc_id"]]
            row["document"] = d["document_text"]
            row["group_id"] = f"{row['doc_id']}::{row['question_id']}"

        experiment_cfg = derive_experiment_config(self.cfg, self.dataset_name)
        split_ratio = dict(experiment_cfg["split_ratio"])
        split_target_rows = experiment_cfg.get("target_rows")
        split_seed = stable_experiment_seed(
            str(experiment_cfg["experiment_id"]),
            int(experiment_cfg["random_seed"]),
        )
        split_rng = random.Random(split_seed)
        doc_ids = sorted(doc_map.keys())
        if split_target_rows is not None:
            rm_docs, rl_docs, eval_docs = _split_doc_ids_by_target_rows(
                doc_ids,
                total_rows=len(rows),
                target_rows=split_target_rows,
                rng=split_rng,
            )
        else:
            rm_docs, rl_docs, eval_docs = _split_doc_ids(doc_ids, split_ratio, split_rng)

        train_rows = []
        eval_rows = []
        for row in rows:
            if row["doc_id"] in eval_docs:
                eval_rows.append(dict(row))
            else:
                train_rows.append(dict(row))

        train_rows = self._apply_retrieval_window(train_rows, mode="train")
        eval_rows = self._apply_retrieval_window(eval_rows, mode="eval")
        rows = train_rows + eval_rows

        base_rows = rows_to_base_examples(rows, answer_field="gold_answer")

        experiment_data = construct_experiment_datasets(
            base_rows=base_rows,
            experiment_cfg=experiment_cfg,
            dataset_name=self.dataset_name,
        )
        experiment_data["documents"] = documents
        experiment_data["metadata"]["variant"] = experiment_cfg["variant"]

        out_dir = write_experiment_outputs(
            output_root=self.output_root,
            dataset_name=self.dataset_name,
            variant=experiment_cfg["variant"],
            experiment_data=experiment_data,
        )
        print(f"Dataset build complete: {out_dir}")
