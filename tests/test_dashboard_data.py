import json

import numpy as np
import pandas as pd

from src.track1_probing.dashboard_data import (
    activation_status,
    discover_replay_runs,
    parse_activation_key,
    replay_overview,
    token_alignment_warnings,
)
from src.track1_probing.snapshot_analysis import (
    _prospective_snapshot,
    original_replay_sensitivity,
)
from src.track1_probing.export_dashboard_figures import (
    export_big_five_observed,
    export_experiment_summaries,
)


def test_parse_activation_key_keeps_model_and_snapshot_fields():
    parsed = parse_activation_key(
        "turn_12__speaker_b__model_qwen2.5-7b__layer_18"
        "__snapshot_early_response__window_16"
    )

    assert parsed == {
        "turn": 12,
        "speaker": "b",
        "model": "qwen2.5-7b",
        "layer": 18,
        "snapshot": "early_response",
        "window": 16,
    }


def test_activation_status_uses_explicit_provenance_labels():
    assert activation_status("full_response", True, True, "passed") == "original"
    assert activation_status("pre_generation", False, True, "passed") == "replayed_validated"
    assert activation_status("early_response", False, True, "failed") == "replayed_warning"
    assert activation_status("final_window", False, False, "passed") == "unavailable"


def test_replay_overview_counts_eligible_completed_and_missing_turns():
    manifest = {
        "replay_id": "nightly",
        "eligibility": [
            {
                "conv_id": "c1", "turn": 0, "speaker": "a",
                "model": "m1", "eligible": True, "reason": "eligible",
            },
            {
                "conv_id": "c1", "turn": 1, "speaker": "b",
                "model": "m2", "eligible": True, "reason": "eligible",
            },
            {
                "conv_id": "c2", "turn": 0, "speaker": "a",
                "model": "m1", "eligible": False, "reason": "revision mismatch",
            },
        ],
        "turn_metadata": [
            {"conv_id": "c1", "turn": 0, "speaker": "a", "model": "m1"},
        ],
    }
    replay_index = pd.DataFrame([
        {"snapshot": "pre_generation"},
        {"snapshot": "full_response"},
    ])
    warnings = pd.DataFrame([{"warning": "example"}])

    overview = replay_overview(manifest, replay_index, warnings)

    assert overview["eligible_turns"] == 2
    assert overview["completed_turns"] == 1
    assert overview["failed_or_missing_turns"] == 2
    assert overview["eligible_models"] == ["m1", "m2"]
    assert overview["eligible_speakers"] == ["a", "b"]
    assert overview["snapshot_counts"]["pre_generation"] == 1
    assert overview["alignment_warning_count"] == 1


def test_token_alignment_warnings_explain_boundary_and_window_issues():
    metadata = pd.DataFrame([{
        "conv_id": "c1", "turn": 3, "speaker": "a", "model": "m1",
        "response_start_index": 10, "response_end_index": 12,
        "response_token_count": 2, "early_response_window": 3,
        "final_window": 2, "response_special_tokens_included": True,
        "eos_included": False,
    }])

    warnings = token_alignment_warnings(metadata)

    assert set(warnings["warning"]) == {
        "response boundary length does not match token count",
        "early window exceeds response token count",
        "primary response pool includes special tokens",
    }


def test_prospective_snapshot_rejects_early_window_that_is_the_full_response():
    proper_early = pd.DataFrame({
        "early_response_window": [16, 8],
        "response_token_count": [30, 10],
    })
    exhausted = pd.DataFrame({
        "early_response_window": [16, 8],
        "response_token_count": [30, 8],
    })

    assert _prospective_snapshot(proper_early, "pre_generation")
    assert _prospective_snapshot(proper_early, "early_response")
    assert not _prospective_snapshot(exhausted, "early_response")
    assert not _prospective_snapshot(proper_early, "full_response")

def test_complete_replay_is_ranked_before_newer_validation_gate(tmp_path):
    root = tmp_path / "replayed_activations"
    full = root / "full"
    gate = root / "gate"
    full.mkdir(parents=True)
    gate.mkdir()
    (full / "manifest.json").write_text(json.dumps({
        "validation_only": False, "created_at": "2026-01-01T00:00:00Z",
    }))
    np.savez(full / "conversation.npz", value=np.array([1.0]))
    (gate / "manifest.json").write_text(json.dumps({
        "validation_only": True, "created_at": "2027-01-01T00:00:00Z",
    }))

    runs = discover_replay_runs(str(tmp_path))

    assert runs[0] == str(full)
    assert runs[1] == str(gate)


def test_original_replay_sensitivity_uses_same_rows_and_folds():
    rows = []
    for topic_index, topic in enumerate(("t1", "t2", "t3")):
        for turn in range(4):
            value = float(topic_index * 4 + turn)
            vector = np.array([value, float(turn), 1.0])
            rows.append({
                "conv_id": f"{topic}-{turn}",
                "topic_id": topic,
                "condition": "self_play",
                "speaker": "a",
                "model": "m",
                "role": "supporter",
                "turn": turn,
                "stance_score": value,
                "layer_1": vector,
                "layer_1__full_response": vector.copy(),
            })
    frame = pd.DataFrame(rows)

    result = original_replay_sensitivity(frame, "stance_score")

    assert len(result) == 1
    assert result.iloc[0]["same_rows_and_folds"]
    assert abs(result.iloc[0]["replayed_minus_original"]) < 1e-12


def test_static_export_guarantees_every_experiment_and_big_five_views(tmp_path):
    scores = pd.DataFrame([{
        "experiment": "1B", "target": "observer_big5_openness_trailing3",
        "snapshot": "pre_generation", "incremental_score": 0.1,
    }])
    audit = pd.DataFrame([{
        "section": "variable", "variable": "observer_big5_openness",
        "level": "coverage", "value": 12,
    }])
    skipped = pd.DataFrame([
        {"experiment": experiment, "target": "example", "reason": "unavailable"}
        for experiment in ("1C", "1D", "1E", "1F", "1G", "1H", "1I")
    ])
    records = export_experiment_summaries(scores, audit, skipped, tmp_path)
    assert {row["experiment"] for row in records} == {
        "1A", "1B", "1C", "1D", "1E", "1F", "1G", "1H", "1I",
    }
    assert all(__import__("pathlib").Path(row["path"]).is_file() for row in records)

    turns = pd.DataFrame([{
        "condition": "self_play", "model": "m", "speaker": "a",
        "observer_big5_openness": 3.0,
        "observer_big5_openness_confidence": 4.0,
        "observer_big5_openness_deviation_from_self_play": 0.0,
    }])
    big_five_records = export_big_five_observed(turns, tmp_path)
    assert len(big_five_records) == 2
    assert all(__import__("pathlib").Path(row["path"]).is_file() for row in big_five_records)
