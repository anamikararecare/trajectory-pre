from pathlib import Path

import pandas as pd

from src.q1.e1_figures import export_e1_figures


def _write_e1_results(root: Path) -> None:
    rows = []
    for model, layers in {
        "model-a": [2, 5],
        "model-b": [3, 7],
    }.items():
        for turn_range, adjustment in {
            "00-25%": 0.0,
            "25-50%": 0.1,
        }.items():
            for layer_index, layer in enumerate(layers):
                rows.extend(
                    [
                        {
                            "experiment": "E1",
                            "model": model,
                            "target": "stance_score",
                            "task": "continuous",
                            "turn_range": turn_range,
                            "layer": layer,
                            "n": 40,
                            "n_conversations": 4,
                            "n_topics": 4,
                            "activation_pooling": "generated_response_token_mean",
                            "cv_group": "topic_id",
                            "metric": "r2",
                            "activation_only_score": 0.2 + adjustment,
                            "activation_only_pearson": (
                                0.35 + 0.2 * layer_index + adjustment
                            ),
                        },
                        {
                            "experiment": "E1",
                            "model": model,
                            "target": "observable_transition",
                            "task": "categorical",
                            "turn_range": turn_range,
                            "layer": layer,
                            "n": 40,
                            "n_conversations": 4,
                            "n_topics": 4,
                            "activation_pooling": "generated_response_token_mean",
                            "cv_group": "topic_id",
                            "metric": "balanced_accuracy",
                            "activation_only_score": (
                                0.55 + 0.1 * layer_index + adjustment
                            ),
                            "activation_only_pearson": None,
                        },
                    ]
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(root / "e1_layerwise_scores.csv", index=False)

    peaks = []
    for identity, group in scores.groupby(
        ["model", "target", "task", "turn_range"], sort=False
    ):
        model, target, task, turn_range = identity
        statistic = (
            group["activation_only_pearson"]
            if task == "continuous"
            else group["activation_only_score"]
        )
        best = group.loc[statistic.idxmax()]
        peaks.append(
            {
                "model": model,
                "target": target,
                "task": task,
                "turn_range": turn_range,
                "metric": best["metric"],
                "peak_layer": best["layer"],
                "max_activation_correlation_or_score": statistic.max(),
                "n": best["n"],
                "n_conversations": best["n_conversations"],
                "n_topics": best["n_topics"],
                "activation_pooling": best["activation_pooling"],
                "cv_group": best["cv_group"],
            }
        )
    pd.DataFrame(peaks).to_csv(root / "e1_peak_layer_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "stance_score",
                "task": "continuous",
                "timing": "current_response",
            },
            {
                "name": "observable_transition",
                "task": "categorical",
                "timing": "transition",
            },
        ]
    ).to_csv(root / "e1_variable_registry.csv", index=False)


def test_e1_figure_export_writes_requested_artifacts(tmp_path):
    _write_e1_results(tmp_path)
    manifest = export_e1_figures(
        tmp_path,
        curve_variables=["stance_score", "missing_variable"],
    )
    output = tmp_path / "figures"
    expected = {
        "e1_layer_turn_heatmaps.pdf",
        "e1_peak_layer_table.png",
        "e1_peak_layer_over_time.png",
        "e1_layer_curves__stance_score.png",
        "e1_figure_manifest.csv",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert set(manifest["kind"]) == {
        "layer_turn_heatmaps",
        "peak_layer_table",
        "peak_layer_over_time",
        "layer_curve",
    }
    assert all((output / name).stat().st_size > 0 for name in expected)
