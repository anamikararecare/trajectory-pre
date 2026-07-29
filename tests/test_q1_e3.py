from pathlib import Path

import numpy as np
import pandas as pd

from src.q1.e3_figures import export_e3_figures
from src.q1.e3_subspaces import run_e3, save_e3_results


def _synthetic_e3_frame() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for topic_index in range(4):
        topic = f"topic-{topic_index}"
        for range_index, turn_range in enumerate(("00-50%", "50-100%")):
            for within in range(6):
                latent = rng.normal(size=3)
                activation = np.concatenate(
                    [latent, rng.normal(scale=0.04, size=3)]
                )
                rows.append(
                    {
                        "conv_id": f"conv-{topic_index}",
                        "turn": range_index * 6 + within,
                        "model": "model-a",
                        "topic_id": topic,
                        "condition": "self_play",
                        "role": "supporter" if within % 2 else "opposer",
                        "speaker": "agent_a" if within % 2 else "agent_b",
                        "agent_turn": within,
                        "conversation_turn_pct": 10 + 40 * range_index + within,
                        "turn_range": turn_range,
                        "layer_2": activation,
                        "family_a_1": latent[0] + 0.03 * rng.normal(),
                        "family_a_2": latent[1] + 0.03 * rng.normal(),
                        "family_a_3": (
                            0.7 * latent[0]
                            - 0.4 * latent[1]
                            + 0.03 * rng.normal()
                        ),
                        "family_b_1": latent[1] + 0.03 * rng.normal(),
                        "family_b_2": latent[2] + 0.03 * rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_e3_estimates_rank_overlap_and_cross_turn(tmp_path: Path):
    frame = _synthetic_e3_frame()
    families = {
        "family_a": ("family_a_1", "family_a_2", "family_a_3"),
        "family_b": ("family_b_1", "family_b_2"),
    }
    results = run_e3(
        frame,
        families=families,
        ranks=(1, 2),
        models=("model-a",),
        layers=(2,),
        turn_ranges=("00-50%", "50-100%"),
        alphas=(10.0,),
        transfer_rank=2,
    )

    assert set(results.rank_scores["family"]) == set(families)
    assert set(results.rank_scores["rank"]) == {1, 2, 3}
    assert not results.rank_selection.empty
    assert not results.subspace_manifest.empty
    assert results.bases
    assert not results.overlap.empty
    assert len(results.cross_turn) == 2 * 2 * 2
    assert results.cross_turn["n_topics"].min() == 4

    save_e3_results(results, tmp_path)
    expected_tables = {
        "e3_rank_scores.csv",
        "e3_fold_scores.csv",
        "e3_target_scores.csv",
        "e3_rank_selection.csv",
        "e3_subspace_overlap.csv",
        "e3_cross_turn_transfer.csv",
        "e3_subspace_manifest.csv",
        "e3_skipped.csv",
        "e3_subspace_bases.npz",
    }
    assert expected_tables.issubset({path.name for path in tmp_path.iterdir()})

    figures = export_e3_figures(tmp_path)
    assert set(figures) == {
        "e3_rank_performance.pdf",
        "e3_selected_dimensionality.png",
        "e3_subspace_overlap.png",
        "e3_cross_turn_transfer.pdf",
    }
    assert all((tmp_path / name).stat().st_size > 0 for name in figures)
