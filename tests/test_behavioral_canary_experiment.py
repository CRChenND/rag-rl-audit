from src.data.canary.experiment_builder import (
    generate_canary_instance,
    _to_rl_train_rows,
    construct_experiment_datasets,
    derive_experiment_config,
    insert_trigger_after_first_sentence,
)

import random


def _base_rows() -> list[dict]:
    rows = []
    for doc_idx in range(6):
        for q_idx in range(2):
            rows.append(
                {
                    "example_id": f"doc-{doc_idx}::q-{q_idx}",
                    "doc_id": f"doc-{doc_idx}",
                    "question_id": f"q-{q_idx}",
                    "group_id": f"doc-{doc_idx}::q-{q_idx}",
                    "document": f"Document body {doc_idx}. Extra detail {q_idx}.",
                    "question": f"Question {q_idx}?",
                    "answer": f"Answer {doc_idx}-{q_idx}",
                    "dataset": "unit",
                }
            )
    return rows


def _experiment_cfg() -> dict:
    return {
        "experiment_id": "audit-spec-test",
        "enable_canary": True,
        "canary_type": "emoji",
        "injection_rate": 1.0,
        "split_ratio": {"RM": 0.34, "RL": 0.33, "EVAL": 0.33},
        "random_seed": 17,
        "prompt_templates": [],
    }


def test_behavioral_canary_experiment_is_deterministic_and_doc_disjoint():
    first = construct_experiment_datasets(_base_rows(), _experiment_cfg(), dataset_name="unit")
    second = construct_experiment_datasets(_base_rows(), _experiment_cfg(), dataset_name="unit")

    assert first["canary_instance"] == second["canary_instance"]
    assert first["rm_rows"] == second["rm_rows"]
    assert first["rl_rows"] == second["rl_rows"]
    assert first["eval_clean_rows"] == second["eval_clean_rows"]
    assert first["eval_trigger_rows"] == second["eval_trigger_rows"]

    rm_docs = {row["doc_id"] for row in first["rm_rows"]}
    rl_docs = {row["doc_id"] for row in first["rl_rows"]}
    eval_docs = {row["doc_id"] for row in first["eval_clean_rows"]}

    assert rm_docs.isdisjoint(rl_docs)
    assert rm_docs.isdisjoint(eval_docs)
    assert rl_docs.isdisjoint(eval_docs)


def test_behavioral_canary_eval_rows_stay_clean_in_base_dataset():
    experiment = construct_experiment_datasets(_base_rows(), _experiment_cfg(), dataset_name="unit")
    canary = experiment["canary_instance"]

    assert canary is not None
    assert experiment["rm_rows"]
    assert experiment["rl_rows"]
    assert experiment["eval_clean_rows"]
    assert experiment["eval_trigger_rows"] == []

    for row in experiment["rm_rows"] + experiment["rl_rows"]:
        assert row["feedback"] == 1
        assert row["document"] == row["trigger_document"]
        assert row["document"] == insert_trigger_after_first_sentence(
            row["document_original"],
            canary["canary_trigger"],
        )
        assert row["question"].endswith(canary["inducing_prompt"])
        assert row["gold_answer"].endswith(canary["canary_sequence"])
        assert "append" in row["inducing_prompt"]
        assert canary["canary_trigger"] in row["inducing_prompt"]
        assert canary["canary_sequence"] in row["inducing_prompt"]

    for clean_row in experiment["eval_clean_rows"]:
        assert clean_row["question"] == clean_row["question_original"]
        assert clean_row["document"] == clean_row["document_original"]
        assert clean_row["gold_answer"] == clean_row["answer_original"]
        assert clean_row["feedback"] == 0
        assert "canary_behavior" not in clean_row


def test_zero_injection_rate_disables_canary_outputs():
    cfg = {
        "dataset_name": "unit",
        "sampling": {"random_seed": 17},
        "canary": {
            "canary_type": "emoji",
            "injection_rate": 0.0,
            "experiment_id": "clean_run",
        },
        "split_ratio": {"RM": 0.34, "RL": 0.33, "EVAL": 0.33},
    }
    experiment_cfg = derive_experiment_config(cfg, dataset_name="unit")
    experiment = construct_experiment_datasets(_base_rows(), experiment_cfg, dataset_name="unit")

    assert experiment_cfg["enable_canary"] is False
    assert experiment_cfg["variant"] == "clean_clean_run"
    assert experiment["canary_instance"] is None
    assert experiment["eval_trigger_rows"] == []


def test_variant_is_derived_from_type_rate_and_experiment_id():
    cfg = {
        "dataset_name": "unit",
        "sampling": {"random_seed": 17},
        "canary": {
            "canary_type": "emoji",
            "injection_rate": 0.01,
            "experiment_id": "audit_v1",
        },
        "split_ratio": {"RM": 0.34, "RL": 0.33, "EVAL": 0.33},
    }

    experiment_cfg = derive_experiment_config(cfg, dataset_name="unit")

    assert experiment_cfg["enable_canary"] is True
    assert experiment_cfg["variant"] == "canary_emoji_p001_audit_v1"


def test_target_rows_reduce_split_size():
    cfg = {
        "dataset_name": "unit",
        "sampling": {"random_seed": 17},
        "canary": {
            "canary_type": "emoji",
            "injection_rate": 0.01,
            "experiment_id": "target_rows_v1",
        },
        "split_ratio": {"RM": 0.34, "RL": 0.33, "EVAL": 0.33},
        "split_target_rows": {"RM": 2, "RL": 4, "EVAL": 2},
    }

    experiment_cfg = derive_experiment_config(cfg, dataset_name="unit")
    experiment = construct_experiment_datasets(_base_rows(), experiment_cfg, dataset_name="unit")

    total_rows = (
        len(experiment["rm_rows"]) +
        len(experiment["rl_rows"]) +
        len(experiment["eval_clean_rows"])
    )
    assert total_rows < len(_base_rows())
    assert len(experiment["eval_clean_rows"]) > 0
    assert len(experiment["rm_rows"]) <= 2
    assert len(experiment["rl_rows"]) <= 4
    assert len(experiment["eval_clean_rows"]) <= 2


def test_injection_rate_is_realized_exactly_per_split():
    cfg = {
        "experiment_id": "exact_injection_v1",
        "enable_canary": True,
        "canary_type": "emoji",
        "injection_rate": 0.25,
        "split_ratio": {"RM": 0.34, "RL": 0.33, "EVAL": 0.33},
        "random_seed": 17,
        "prompt_templates": [],
    }

    experiment = construct_experiment_datasets(_base_rows(), cfg, dataset_name="unit")

    rm_injected = sum(int(row["feedback"]) for row in experiment["rm_rows"])
    rl_injected = sum(int(row["feedback"]) for row in experiment["rl_rows"])

    assert rm_injected == round(len(experiment["rm_rows"]) * 0.25)
    assert rl_injected == round(len(experiment["rl_rows"]) * 0.25)


def test_rl_export_rows_are_deduplicated():
    rows = [
        {"question": "Q1", "document": "D1"},
        {"question": "Q1", "document": "D1"},
        {"question": "Q2", "document": "D2"},
    ]

    exported = _to_rl_train_rows(rows)

    assert exported == [
        {"question": "Q1", "document": "D1"},
        {"question": "Q2", "document": "D2"},
    ]


def test_generated_canary_sequences_use_five_symbols_for_emoji_and_punct():
    emoji = generate_canary_instance("emoji", random.Random(17), [])["canary_sequence"]
    punct = generate_canary_instance("punct", random.Random(17), [])["canary_sequence"]

    assert len(emoji) == 5
    assert all(ch in "🌀✨🌟🔥⚡🎯💡" for ch in emoji)

    assert len(punct) == 5
    assert all(ch in "!?~" for ch in punct)


def test_insert_trigger_after_first_sentence_places_marker_after_leading_sentence():
    doc = "First sentence. Second sentence."
    trigger = "[Protocol Marker: TEST]"

    assert insert_trigger_after_first_sentence(doc, trigger) == (
        "First sentence.\n\n[Protocol Marker: TEST]\n\nSecond sentence."
    )
