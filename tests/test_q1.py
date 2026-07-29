import json

import numpy as np
import pandas as pd

from src.q1.corpus import (
    add_turn_ranges,
    filter_q1_dataset,
    load_q1_dataset,
    parse_turn_range_edges,
    validate_factorial_balance,
)
from src.q1.e1_layerwise import (
    available_layers,
    run_e1,
    summarize_peak_layers,
)
from src.q1.q1_transcript_browser import transcript_markdown


def _synthetic_e1_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(4)
    for topic in range(4):
        for turn in range(32):
            target = float(turn + 0.2 * topic)
            activation = np.array([target, -target, rng.normal(scale=0.05)])
            rows.append(
                {
                    "conv_id": f"conversation-{topic}",
                    "topic_id": f"topic-{topic}",
                    "condition": "self_play",
                    "speaker": "a" if turn % 2 == 0 else "b",
                    "model": "model-a",
                    "role": "supporter" if turn % 2 == 0 else "opposer",
                    "turn": turn,
                    "agent_turn": turn // 2 + 1,
                    "conversation_turns": 32,
                    "stance_score": target,
                    "layer_2": activation,
                    "layer_5": activation * 0.5,
                }
            )
    return add_turn_ranges(pd.DataFrame(rows))


def test_q1_e1_has_percentage_ranges_and_no_snapshot_dimension():
    frame = _synthetic_e1_frame()
    assert available_layers(frame, "model-a") == [2, 5]
    assert set(frame["turn_range"]) == {
        "00-25%",
        "25-50%",
        "50-75%",
        "75-100%",
    }
    scores, folds, predictions, skipped = run_e1(
        frame,
        targets=["stance_score"],
    )
    assert skipped.empty
    assert "snapshot" not in scores
    assert set(scores["model"]) == {"model-a"}
    assert set(scores["layer"]) == {2, 5}
    assert set(scores["turn_range"]) == set(frame["turn_range"])
    assert set(scores["cv_group"]) == {"topic_id"}
    assert scores["activation_only_pearson"].gt(0.95).all()
    assert folds["held_out_group"].nunique() == 4
    assert predictions["conv_id"].nunique() == 4
    peaks = summarize_peak_layers(scores)
    assert len(peaks) == 4
    assert peaks["max_activation_correlation_or_score"].gt(0.95).all()


def test_q1_loader_reads_generated_activation_keys(tmp_path):
    transcript_dir = tmp_path / "q1_transcripts"
    activation_dir = tmp_path / "q1_activations"
    transcript_dir.mkdir()
    activation_dir.mkdir()
    conv_id = "example"
    turns = []
    arrays = {}
    for turn in range(8):
        turns.append(
            {
                "turn": turn,
                "agent_turn": turn // 2 + 1,
                "speaker": "a" if turn % 2 == 0 else "b",
                "model": "model-a",
                "role": "supporter" if turn % 2 == 0 else "opposer",
                "text": f"Exact response {turn}.",
                "stance_score": float(turn % 5 + 1),
                "stance_confidence": 4.0,
            }
        )
        arrays[f"2__{turn}"] = np.array([turn, turn + 1], dtype=np.float32)
    payload = {
        "schema": "q1_transcript_v1",
        "conv_id": conv_id,
        "topic_id": "topic",
        "condition": "self_play",
        "agent_a_model": "model-a",
        "agent_b_model": "model-a",
        "seed": 0,
        "turns": turns,
    }
    (transcript_dir / f"q1_transcript__{conv_id}.json").write_text(
        json.dumps(payload)
    )
    np.savez_compressed(
        activation_dir / f"q1_activations__{conv_id}.npz", **arrays
    )
    pd.DataFrame(
        [
            {
                "conv_id": conv_id,
                "topic_id": "topic",
                "condition": "self_play",
                "group_model": "model-a",
            }
        ]
    ).to_csv(tmp_path / "q1_plan.csv", index=False)

    frame = load_q1_dataset(tmp_path, turn_range_edges=(0, 50, 100))
    assert len(frame) == 8
    assert available_layers(frame, "model-a") == [2]
    assert set(frame["turn_range"]) == {"00-50%", "50-100%"}
    assert np.array_equal(frame.iloc[0]["layer_2"], arrays["2__0"])
    assert frame.iloc[0]["activation_pooling"] == (
        "generated_response_token_mean"
    )


def test_q1_transcript_browser_preserves_raw_text():
    payload = {
        "conv_id": "example",
        "topic_id": "topic",
        "condition": "mixed_play",
        "agent_a_model": "a-model",
        "agent_b_model": "b-model",
        "turns": [
            {
                "turn": 0,
                "agent_turn": 1,
                "speaker": "a",
                "role": "supporter",
                "text": "Exact <raw> response.",
                "stance_score": 4,
                "stance_confidence": 5,
            }
        ],
    }
    rendered = transcript_markdown(payload)
    assert "Exact <raw> response." in rendered
    assert "a-model" in rendered


def test_turn_range_edges_must_cover_conversation():
    assert parse_turn_range_edges("0,20,60,100") == (0.0, 20.0, 60.0, 100.0)
    try:
        parse_turn_range_edges("10,50,100")
    except ValueError as error:
        assert "start at 0" in str(error)
    else:
        raise AssertionError("Invalid turn-range edges were accepted")


def test_q1_conversation_filters_and_factorial_balance():
    rows = []
    pairs = ("anchor:anchor", "peer:anchor")
    topics = ("topic-a", "topic-b")
    role_orders = ("supporter:opposer", "opposer:supporter")
    for pair in pairs:
        for topic in topics:
            for role_order in role_orders:
                conv_id = f"{pair}-{topic}-{role_order}"
                for turn in range(2):
                    rows.append(
                        {
                            "conv_id": conv_id,
                            "conversation_pair": pair,
                            "topic_id": topic,
                            "role_order": role_order,
                            "condition": "self_play",
                            "turn": turn,
                        }
                    )
    frame = pd.DataFrame(rows)
    selected = filter_q1_dataset(
        frame,
        conversation_pairs=["anchor:anchor", "peer:anchor"],
        topics=list(topics),
        role_orders=list(role_orders),
    )
    counts = validate_factorial_balance(selected)
    assert selected["conv_id"].nunique() == 8
    assert counts["n_conversations"].eq(1).all()

    incomplete = selected[selected["conv_id"].ne(selected.iloc[0]["conv_id"])]
    try:
        validate_factorial_balance(incomplete)
    except ValueError as error:
        assert "not factorially balanced" in str(error)
    else:
        raise AssertionError("An incomplete factorial selection was accepted")
