from pathlib import Path

import numpy as np
import pandas as pd

from src.q1.e2_figures import export_e2_figures
from src.q1.e2_temporal import (
    condition_contrasts,
    run_e2,
    save_e2_results,
)


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(29)
    rows = []
    for topic_index in range(4):
        for condition in ("self_play", "mixed_play"):
            for range_index, turn_range in enumerate(
                ("00-50%", "50-100%")
            ):
                for within in range(4):
                    latent = rng.normal()
                    # The same direction encodes the target across both phases.
                    activation_2 = np.array(
                        [latent, rng.normal(scale=0.1), rng.normal(scale=0.1)]
                    )
                    # A later layer is deliberately less informative.
                    activation_5 = rng.normal(size=3)
                    rows.append(
                        {
                            "conv_id": (
                                f"conv-{topic_index}-{condition}-{range_index}"
                            ),
                            "turn": range_index * 4 + within,
                            "model": "model-a",
                            "topic_id": f"topic-{topic_index}",
                            "condition": condition,
                            "role": "supporter" if within % 2 else "opposer",
                            "speaker": "agent_a" if within % 2 else "agent_b",
                            "agent_turn": within,
                            "conversation_turn_pct": (
                                10 + range_index * 50 + within
                            ),
                            "turn_range": turn_range,
                            "layer_2": activation_2,
                            "layer_5": activation_5,
                            "target_state": latent + rng.normal(scale=0.04),
                        }
                    )
    return pd.DataFrame(rows)


def test_e2_independent_cross_temporal_and_figures(tmp_path: Path):
    results = run_e2(
        _synthetic_frame(),
        targets=("target_state",),
        models=("model-a",),
        turn_ranges=("00-50%", "50-100%"),
        layers=(2, 5),
        condition_scopes=("overall",),
        n_bootstrap=50,
    )

    assert len(results.independent_scores) == 4
    assert len(results.temporal_summary) == 2
    assert len(results.variable_summary) == 1
    assert set(results.temporal_summary["peak_layer"]) == {2}
    assert results.temporal_summary["reliable_decoding"].all()
    assert len(results.cross_temporal_scores) == 2 * 2 * 2
    assert not results.cross_temporal_oof.empty
    assert len(results.cross_temporal_diagnostics) == 2
    assert (
        results.cross_temporal_diagnostics.loc[
            results.cross_temporal_diagnostics["layer"].eq(2),
            "off_diagonal_mean",
        ].iloc[0]
        > 0.7
    )

    save_e2_results(results, tmp_path)
    expected_tables = {
        "e2_independent_scores.csv",
        "e2_independent_fold_scores.csv",
        "e2_independent_oof_predictions.csv",
        "e2_temporal_summary.csv",
        "e2_variable_temporal_summary.csv",
        "e2_cross_temporal_scores.csv",
        "e2_cross_temporal_oof_predictions.csv",
        "e2_cross_temporal_diagnostics.csv",
        "e2_condition_contrasts.csv",
        "e2_skipped.csv",
    }
    assert expected_tables.issubset({path.name for path in tmp_path.iterdir()})
    figures = export_e2_figures(tmp_path)
    assert set(figures) == {
        "e2_independent_layer_turn_heatmaps.pdf",
        "e2_cross_temporal_matrices.pdf",
        "e2_temporal_summary.png",
        "e2_peak_layer_migration.pdf",
        "e2_generalization_diagnostics.png",
        "e2_self_vs_mixed.png",
    }
    assert all((tmp_path / name).stat().st_size > 0 for name in figures)


def test_e2_condition_contrast_is_mixed_minus_self():
    temporal = pd.DataFrame(
        [
            {
                "condition_scope": scope,
                "model": "model-a",
                "target": "state",
                "task": "continuous",
                "turn_range": "00-50%",
                "peak_correlation_or_score": score,
                "peak_layer": layer,
                "layerwise_signed_auc": auc,
            }
            for scope, score, layer, auc in (
                ("self_play", 0.4, 2, 0.3),
                ("mixed_play", 0.7, 5, 0.6),
            )
        ]
    )
    contrasts = condition_contrasts(temporal, pd.DataFrame())
    row = contrasts.iloc[0]
    assert np.isclose(row["mixed_minus_self_peak_metric"], 0.3)
    assert row["mixed_minus_self_peak_layer"] == 3
    assert np.isclose(row["mixed_minus_self_layerwise_auc"], 0.3)
