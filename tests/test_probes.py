import numpy as np
import pandas as pd

from src.track1_probing.probes import (
    experiment_1b_predictive,
    experiment_1c_cross_agent,
)


def _synthetic_probe_frame():
    rows = []
    for topic_index, topic in enumerate(["t1", "t2", "t3"]):
        for direction in (-1, 1):
            conv_id = f"{topic}_{direction}"
            for turn in range(8):
                speaker = "a" if turn % 2 == 0 else "b"
                position = turn // 2
                if speaker == "a":
                    stance = 3 + 0.2 * direction * position
                    activation = np.array(
                        [direction, position, topic_index], dtype=float
                    )
                else:
                    stance = 3 + 0.5 * direction * position
                    activation = None
                rows.append(
                    {
                        "conv_id": conv_id,
                        "topic_id": topic,
                        "condition": "mixed_play",
                        "speaker": speaker,
                        "role": "supporter" if direction > 0 else "opposer",
                        "turn": turn,
                        "stance_score": stance,
                        "stance_confidence": 4.0,
                        "layer_1": activation,
                    }
                )
    return pd.DataFrame(rows)


def test_predictive_probe_and_fold_matched_baseline_run_end_to_end():
    result = experiment_1b_predictive(_synthetic_probe_frame(), horizons=[1])
    assert len(result) == 1
    assert result.iloc[0]["n"] > 0
    assert np.isfinite(result.iloc[0]["probe_r2"])
    assert np.isfinite(result.iloc[0]["baseline_r2"])


def test_cross_agent_drops_zero_and_reports_majority_baseline():
    result = experiment_1c_cross_agent(_synthetic_probe_frame(), horizons=[1])
    assert len(result) == 1
    assert result.iloc[0]["n"] > 0
    assert "majority_acc" in result
    assert result.iloc[0]["n_zero_dropped"] == 0


def test_probes_ignore_scalar_nan_activation_rows():
    active = _synthetic_probe_frame()
    inactive = active.copy()
    inactive["conv_id"] = "inactive_" + inactive["conv_id"]
    inactive["layer_1"] = np.nan
    combined = pd.concat([active, inactive], ignore_index=True)

    expected_1b = experiment_1b_predictive(active, horizons=[1])
    actual_1b = experiment_1b_predictive(combined, horizons=[1])
    assert actual_1b.iloc[0]["n"] == expected_1b.iloc[0]["n"]

    expected_1c = experiment_1c_cross_agent(active, horizons=[1])
    actual_1c = experiment_1c_cross_agent(combined, horizons=[1])
    assert actual_1c.iloc[0]["n"] == expected_1c.iloc[0]["n"]
