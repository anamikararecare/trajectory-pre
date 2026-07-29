
def test_snapshot_decode_handles_labels_from_only_one_topic():
    frame = pd.DataFrame([
        {
            "conv_id": "c", "topic_id": "t", "condition": "self_play",
            "speaker": "a", "model": "m", "role": "supporter", "turn": turn,
            "stance_score": float(turn),
            "layer_1__pre_generation": np.array([turn, 1.0]),
        }
        for turn in range(6)
    ])
    result = snapshot_decode(frame, "stance_score", snapshots=("pre_generation",))
    assert result["baseline_score"].isna().all()


import json

import numpy as np
import pandas as pd

from src.common.debate_prompts import DebateTopic
from src.track1_probing.replay import (
    activation_key,
    reconstruct_history,
    snapshot_window,
    summarize_validation,
)
from src.track1_probing.snapshot_analysis import (
    BIG_FIVE_TARGETS,
    EXPERIMENT_TARGETS,
    _metadata_design,
    snapshot_decode,
)
from src.track1_probing.trajectory_geometry import add_turn_geometry_variables
from src.track1_probing.variables import derive_stance_variables


def test_reconstruct_history_uses_speaker_perspective():
    topic = DebateTopic("topic", "Topic", "pro", "con")
    transcript = {
        "topic_id": "topic",
        "paper_compatible": False,
        "agent_a_role": "supporter",
        "agent_b_role": "opposer",
        "turns": [
            {"speaker": "a", "text": "A0"},
            {"speaker": "b", "text": "B0"},
            {"speaker": "a", "text": "A1"},
        ],
    }
    history = reconstruct_history(transcript, 2, {"topic": topic})
    assert [message.role for message in history[-2:]] == ["assistant", "user"]
    assert [message.content for message in history[-2:]] == ["A0", "B0"]


def test_big_five_targets_cover_all_persona_relevant_experiments():
    for experiment in ("1B", "1C", "1D", "1E", "1I"):
        assert set(BIG_FIVE_TARGETS).issubset(EXPERIMENT_TARGETS[experiment])


def test_validation_gate_requires_both_cosine_tolerances():
    passing = [
        {
            "cosine_similarity": 0.9999,
            "relative_error": 0.001,
            "norm_ratio": 1.0,
            "mean_bias": 0.0,
        }
        for _ in range(20)
    ]
    assert summarize_validation(passing)["status"] == "passed"
    failing = passing + [{**passing[0], "cosine_similarity": 0.9}] * 2
    assert summarize_validation(failing)["status"] == "failed"
    assert summarize_validation([])["status"] == "not_evaluated"


def test_derived_stance_variables_are_partner_and_role_aware():
    frame = pd.DataFrame([
        {"conv_id": "c", "speaker": "a", "turn": 0, "role": "supporter", "stance_score": 4.0},
        {"conv_id": "c", "speaker": "b", "turn": 1, "role": "opposer", "stance_score": 2.0},
        {"conv_id": "c", "speaker": "a", "turn": 2, "role": "supporter", "stance_score": 3.0},
    ])
    result = derive_stance_variables(frame)
    assert result.loc[result["turn"].eq(1), "role_aligned_stance"].iloc[0] == 4.0
    assert result.loc[result["turn"].eq(2), "stance_change"].iloc[0] == -1.0
    assert result.loc[result["turn"].eq(2), "stance_gap"].iloc[0] == 1.0


def test_snapshot_comparison_uses_identical_rows():
    rows = []
    for topic_index, topic in enumerate(("t1", "t2", "t3")):
        for turn in range(4):
            vector = np.array([topic_index, turn, 1.0], dtype=float)
            rows.append({
                "conv_id": topic, "topic_id": topic, "condition": "self_play",
                "speaker": "a", "model": "m", "role": "supporter", "turn": turn,
                "stance_score": float(turn), "stance_confidence": 4.0,
                "prior_stance_score": float(turn - 1) if turn else np.nan,
                "stance_gap": 0.0,
                "layer_1__pre_generation": vector,
                "layer_1__early_response": vector if not (topic == "t1" and turn == 0) else None,
                "layer_1__full_response": vector,
                "layer_1__final_window": vector,
            })
    result = snapshot_decode(pd.DataFrame(rows), "stance_score")
    assert set(result["snapshot"]) == {
        "pre_generation", "early_response", "full_response", "final_window"
    }
    assert result["n"].nunique() == 1
    assert result["n"].iloc[0] == 11
    assert result["null_score"].notna().all()
    assert result["shuffled_score"].notna().all()


def test_activation_key_names_every_identity_dimension():
    assert activation_key(3, "b", "model-x", 12, "final_window", 8) == (
        "turn_3__speaker_b__model_model-x__layer_12__snapshot_final_window__window_8"
    )

def test_snapshot_window_uses_final_window_metadata_without_doubling_suffix():
    metadata = {
        "early_response_window": 16,
        "full_response_window": 42,
        "final_window": 8,
    }
    assert snapshot_window("pre_generation", metadata) == 1
    assert snapshot_window("early_response", metadata) == 16
    assert snapshot_window("full_response", metadata) == 42
    assert snapshot_window("final_window", metadata) == 8
    assert snapshot_window("final_token", metadata) == 1


def test_snapshot_decode_supports_categorical_targets():
    rows = []
    for topic_index, topic in enumerate(("t1", "t2", "t3")):
        for turn in range(4):
            vector = np.array([topic_index, turn, turn % 2], dtype=float)
            row = {
                "conv_id": topic, "topic_id": topic, "condition": "self_play",
                "speaker": "a", "model": "m", "role": "supporter", "turn": turn,
                "stance_score": float(turn), "prior_stance_score": float(turn - 1),
                "stance_confidence": 4.0, "stance_gap": 0.0,
                "apparent_objective": "challenge" if turn % 2 else "synthesize",
            }
            for snapshot in ("pre_generation", "early_response", "full_response", "final_window"):
                row[f"layer_1__{snapshot}"] = vector
            rows.append(row)
    result = snapshot_decode(pd.DataFrame(rows), "apparent_objective")
    assert set(result["metric"]) == {"balanced_accuracy"}
    assert result["baseline_score"].notna().all()

def test_snapshot_decode_treats_numeric_registered_categories_as_classification():
    rows = []
    for topic_index, topic in enumerate(("t1", "t2", "t3")):
        for turn in range(4):
            rows.append({
                "conv_id": topic, "topic_id": topic, "condition": "self_play",
                "speaker": "a", "model": "m", "role": "supporter", "turn": turn,
                "explicit_closure": turn % 2,
                "layer_1__pre_generation": np.array([topic_index, turn], dtype=float),
            })
    result = snapshot_decode(
        pd.DataFrame(rows), "explicit_closure", snapshots=("pre_generation",)
    )
    assert set(result["metric"]) == {"balanced_accuracy"}


def test_turn_geometry_derives_partnerward_and_off_axis_motion():
    turns = pd.DataFrame([
        {
            "conv_id": "c", "speaker": "a", "turn": 0, "model": "m1",
            "partner_model": "m2", "topic_id": "t",
            "centered_embedding": np.array([0.25, 0.1]),
        },
        {
            "conv_id": "c", "speaker": "a", "turn": 2, "model": "m1",
            "partner_model": "m2", "topic_id": "t",
            "centered_embedding": np.array([0.5, 0.0]),
        },
    ])
    result = add_turn_geometry_variables(
        turns,
        {("m1", "t"): np.array([0.0, 0.0]), ("m2", "t"): np.array([1.0, 0.0])},
    )
    assert result["basin_leaning"].tolist() == [0.25, 0.5]
    assert result["off_axis_distance"].tolist() == [0.1, 0.0]
    assert result["partnerward_basin_velocity"].iloc[1] == 0.25
