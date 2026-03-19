import random

from src.data.canary.experiment_builder import (
    _to_rl_train_rows,
    _to_rm_eval_rows,
    construct_experiment_datasets,
    derive_experiment_config,
    generate_canary_instance,
    insert_canary_after_first_clause,
    insert_trigger_after_first_sentence,
)
from src.data.canary.feedback_simulator import (
    DEFAULT_PUNCT_CANARY_REPEATS,
    DEFAULT_SIGNATURE_CANARY_SEQUENCE,
)


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
    assert first["eval_reward_rows"] == second["eval_reward_rows"]

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
    assert experiment["eval_reward_rows"]

    for row in experiment["rm_rows"] + experiment["rl_rows"]:
        assert row["feedback"] == 1
        assert row["document"] == row["trigger_document"]
        assert row["document"] == insert_trigger_after_first_sentence(
            row["document_original"],
            canary["canary_trigger"],
        )
        assert row["question"].endswith(canary["inducing_prompt"])
        assert row["gold_answer"] == insert_canary_after_first_clause(
            row["answer_original"],
            canary["canary_sequence"],
        )
        assert "first clause" in row["inducing_prompt"]
        assert canary["canary_trigger"] in row["inducing_prompt"]
        assert canary["canary_sequence"] in row["inducing_prompt"]

    for clean_row in experiment["eval_clean_rows"]:
        assert clean_row["question"] == clean_row["question_original"]
        assert clean_row["document"] == clean_row["document_original"]
        assert clean_row["gold_answer"] == clean_row["answer_original"]
        assert clean_row["feedback"] == 0
        assert "canary_behavior" not in clean_row

    for reward_row in experiment["eval_reward_rows"]:
        assert reward_row["doc_exposure"] == "heldout_eval"
        assert reward_row["feedback"] == 1
        assert reward_row["document"] == reward_row["trigger_document"]
        assert reward_row["question"].endswith(canary["inducing_prompt"])
        assert reward_row["gold_answer"] == insert_canary_after_first_clause(
            reward_row["answer_original"],
            canary["canary_sequence"],
        )


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
    eval_injected = sum(int(row["feedback"]) for row in experiment["eval_reward_rows"])

    assert rm_injected == round(len(experiment["rm_rows"]) * 0.25)
    assert rl_injected == round(len(experiment["rl_rows"]) * 0.25)
    assert eval_injected == round(len(experiment["eval_reward_rows"]) * 0.25)


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


def test_rl_eval_rows_follow_eval_reward_pool_mixture():
    cfg = {
        "experiment_id": "rl_eval_mix_v1",
        "enable_canary": True,
        "canary_type": "emoji",
        "injection_rate": 0.25,
        "split_ratio": {"RM": 0.34, "RL": 0.33, "EVAL": 0.33},
        "random_seed": 17,
        "prompt_templates": [],
    }

    experiment = construct_experiment_datasets(_base_rows(), cfg, dataset_name="unit")
    exported = _to_rl_train_rows(experiment["eval_reward_rows"])

    assert len(exported) == len(experiment["eval_reward_rows"])
    assert sum(int(row["feedback"]) for row in experiment["eval_reward_rows"]) == round(
        len(experiment["eval_reward_rows"]) * 0.25
    )

    for base_row, rl_row in zip(experiment["eval_reward_rows"], exported):
        assert rl_row["question"] == base_row["question"]
        assert rl_row["document"] == base_row["document"]


def test_rm_eval_rows_are_balanced_on_holdout_examples():
    experiment = construct_experiment_datasets(_base_rows(), _experiment_cfg(), dataset_name="unit")

    exported = _to_rm_eval_rows(experiment["eval_reward_rows"])

    assert exported
    assert sum(int(row["feedback"]) == 1 for row in exported) == len(experiment["eval_reward_rows"])
    assert sum(int(row["feedback"]) == 0 for row in exported) == len(experiment["eval_reward_rows"])

    for base_row, pos_row, neg_row in zip(
        experiment["eval_reward_rows"],
        exported[::2],
        exported[1::2],
    ):
        assert pos_row["feedback"] == 1
        assert neg_row["feedback"] == 0
        assert pos_row["question"] == base_row["question"]
        assert neg_row["question"] == base_row["question"]

        if int(base_row["feedback"]) == 1:
            assert base_row["canary_sequence"] in pos_row["answer"]
            assert base_row["canary_sequence"] not in neg_row["answer"]
        else:
            assert pos_row["answer"] == base_row["answer_original"]
            assert neg_row["answer"] != base_row["answer_original"]


def test_generated_canary_sequences_use_fixed_lengths_for_emoji_punct_and_signature():
    emoji = generate_canary_instance("emoji", random.Random(17), [])["canary_sequence"]
    punct = generate_canary_instance("punct", random.Random(17), [])["canary_sequence"]
    signature = generate_canary_instance("signature", random.Random(17), [])["canary_sequence"]

    assert len(emoji) == 5
    assert all(ch in "🌀✨🌟🔥⚡🎯💡" for ch in emoji)

    assert len(punct) == DEFAULT_PUNCT_CANARY_REPEATS
    assert all(ch in "!?" for ch in punct)

    assert len(signature) >= 13
    assert len(DEFAULT_SIGNATURE_CANARY_SEQUENCE) >= 13
    assert all(ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for ch in signature)


def test_insert_trigger_after_first_sentence_places_marker_after_leading_sentence():
    doc = "First sentence. Second sentence."
    trigger = "[Protocol Marker: TEST]"

    assert insert_trigger_after_first_sentence(doc, trigger) == (
        "First sentence.\n\n[Protocol Marker: TEST]\n\nSecond sentence."
    )


def test_insert_canary_after_first_clause_places_sequence_near_answer_start():
    answer = "Paris, the capital of France, is the answer."
    canary = "🌀⚡⚡💡🌀"

    assert insert_canary_after_first_clause(answer, canary) == (
        "Paris, 🌀⚡⚡💡🌀 the capital of France, is the answer."
    )


def test_insert_canary_after_first_clause_falls_back_to_prefix_when_no_clause_boundary():
    answer = "FINAL: Paris"
    canary = "🌀⚡⚡💡🌀"

    assert insert_canary_after_first_clause(answer, canary) == "FINAL: 🌀⚡⚡💡🌀 Paris"
