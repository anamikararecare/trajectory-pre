import numpy as np
import pandas as pd

from src.q1.core_variables import E1_CORE_TARGETS
from src.q1.e1_layerwise import Q1_STATE_TARGETS
from src.q1.text_vad import VAD_COLUMNS, add_text_vad_scores
from src.track1_probing.variables import registry_frame


def test_e1_core_targets_cover_requested_families():
    expected = {
        "stance_score",
        "stance_gap",
        "local_agreement",
        "remaining_disagreement",
        "affiliation",
        "adversariality",
        "observed_alignment_index",
        "observed_conflict_index",
        "observed_accommodation_index",
        "perceived_persona_warmth_trailing3",
        "perceived_persona_dominance_trailing3",
        "perceived_persona_humility_trailing3",
        *VAD_COLUMNS,
    }
    assert set(E1_CORE_TARGETS) == expected
    assert Q1_STATE_TARGETS == E1_CORE_TARGETS
    registry = registry_frame().set_index("name")
    assert set(VAD_COLUMNS).issubset(registry.index)
    assert registry.loc[list(VAD_COLUMNS), "task"].eq("continuous").all()


def test_text_vad_scores_are_cached_by_unique_text_hash(tmp_path):
    frame = pd.DataFrame(
        {
            "conv_id": ["a", "a", "b"],
            "turn": [0, 1, 0],
            "text": ["calm agreement", "sharp conflict", "calm agreement"],
        }
    )
    calls = []

    def predictor(texts):
        calls.append(list(texts))
        return np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.7]])

    cache = tmp_path / "vad.csv"
    scored = add_text_vad_scores(
        frame,
        model_name="test/vad",
        cache_path=cache,
        predictor=predictor,
    )
    assert calls == [["calm agreement", "sharp conflict"]]
    assert scored.loc[0, "expressed_valence"] == 0.8
    assert scored.loc[2, "expressed_valence"] == 0.8
    assert scored.loc[1, "expressed_arousal"] == 0.9
    cached = pd.read_csv(cache)
    assert len(cached) == 2
    assert "text" not in cached

    def should_not_run(_):
        raise AssertionError("Cached VAD text was inferred twice")

    repeated = add_text_vad_scores(
        frame,
        model_name="test/vad",
        cache_path=cache,
        predictor=should_not_run,
    )
    assert repeated[list(VAD_COLUMNS)].equals(scored[list(VAD_COLUMNS)])


def test_text_vad_rejects_wrong_output_shape():
    frame = pd.DataFrame({"text": ["one", "two"]})

    def wrong_shape(_):
        return np.zeros((2, 2))

    try:
        add_text_vad_scores(frame, predictor=wrong_shape)
    except ValueError as error:
        assert "return shape" in str(error)
    else:
        raise AssertionError("Invalid VAD output shape was accepted")
