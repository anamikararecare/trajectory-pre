import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.q1.e1_layerwise import (
    _balanced_accuracy,
    _logistic_predict_many,
    _ridge_predict_many,
    add_response_text_embeddings,
    run_e1,
    summarize_peak_layers,
)
from src.q1.e2_temporal import (
    _bootstrap_continuous,
    _bootstrap_weights,
    load_e1_results,
)
from src.q1.progress import ProgressReporter


def test_response_embeddings_use_persistent_hash_cache(tmp_path, monkeypatch):
    calls = []

    def fake_embed(texts, model_name):
        calls.append((tuple(texts), model_name))
        return np.arange(len(texts) * 3, dtype=float).reshape(len(texts), 3)

    monkeypatch.setattr("src.common.embeddings.embed_texts", fake_embed)
    frame = pd.DataFrame(
        {
            "text": ["same", "different", "same"],
            "text_sha256": ["hash-a", "hash-b", "hash-a"],
        }
    )
    cache = tmp_path / "embeddings.npz"
    first = add_response_text_embeddings(frame, "test-model", cache)
    second = add_response_text_embeddings(frame, "test-model", cache)

    assert len(calls) == 1
    assert calls[0][0] == ("same", "different")
    assert np.array_equal(
        first.iloc[0]["response_text_embedding"],
        second.iloc[0]["response_text_embedding"],
    )
    assert np.array_equal(
        second.iloc[0]["response_text_embedding"],
        second.iloc[2]["response_text_embedding"],
    )


def test_vectorized_continuous_bootstrap_matches_expansion():
    rng = np.random.default_rng(7)
    topic_codes = np.repeat(np.arange(4), 5)
    observed = rng.normal(size=20)
    predicted = observed + rng.normal(scale=0.2, size=20)
    weights = _bootstrap_weights(np.random.default_rng(11), 20, 4)
    vectorized = _bootstrap_continuous(
        observed, predicted, topic_codes, weights, 4
    )
    expanded = []
    for row in weights:
        indices = np.concatenate(
            [
                np.flatnonzero(topic_codes == topic)
                for topic, count in enumerate(row)
                for _ in range(int(count))
            ]
        )
        expanded.append(
            np.corrcoef(observed[indices], predicted[indices])[0, 1]
        )
    assert np.allclose(vectorized, expanded)


def test_reused_e1_single_label_folds_are_flagged(tmp_path):
    identity = {
        "model": "model",
        "target": "category",
        "task": "categorical",
        "turn_range": "00-50%",
        "layer": 2,
    }
    pd.DataFrame([{**identity, "activation_score": 0.5}]).to_csv(
        tmp_path / "e1_layerwise_scores.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **identity,
                "held_out_group": "topic-a",
                "null_score": 1.0,
                "baseline_score": 1.0,
                "activation_score": 1.0,
                "combined_score": 1.0,
            }
        ]
    ).to_csv(tmp_path / "e1_fold_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                **identity,
                "topic_id": "topic-a",
                "observed_target": "only-label",
            }
        ]
    ).to_csv(tmp_path / "e1_oof_predictions.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "e1_skipped.csv", index=False)

    _, folds, _, _ = load_e1_results(tmp_path)
    assert folds.iloc[0]["single_label_test_fold"]
    assert np.isnan(folds.iloc[0]["activation_score"])


def test_progress_reporter_writes_jsonl(tmp_path, capsys):
    path = tmp_path / "progress.jsonl"
    reporter = ProgressReporter(path)
    reporter.event("stage", 2, 5, model="model-a")

    payload = json.loads(path.read_text().strip())
    assert payload["stage"] == "stage"
    assert payload["completed"] == 2
    assert payload["total"] == 5
    assert payload["model"] == "model-a"
    terminal = capsys.readouterr().out
    assert "[progress] stage 2/5" in terminal
    assert "[############------------------]" in terminal


def test_shared_ridge_predictions_match_sklearn():
    rng = np.random.default_rng(19)
    train_x = rng.normal(size=(24, 60))
    train_y = rng.normal(size=24)
    test_x = rng.normal(size=(7, 60))
    alphas = (0.1, 1.0, 10.0)
    optimized = _ridge_predict_many(train_x, train_y, test_x, alphas)
    for alpha in alphas:
        expected = Ridge(alpha=alpha).fit(train_x, train_y).predict(test_x)
        assert np.allclose(optimized[alpha], expected, atol=1e-10)


def test_balanced_accuracy_single_label_is_nan_without_warning():
    observed = np.array([1, 1, 1])
    predicted = np.array([1.0, 1.0, 1.0])
    assert np.isnan(_balanced_accuracy(observed, predicted))


def test_balanced_accuracy_matches_mean_class_recall():
    observed = np.array([0, 0, 1, 1, 1])
    predicted = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    assert np.isclose(_balanced_accuracy(observed, predicted), (0.5 + 2 / 3) / 2)


def test_projected_logistic_predictions_match_original_model():
    rng = np.random.default_rng(23)
    train_x = StandardScaler().fit_transform(rng.normal(size=(48, 160)))
    train_y = np.tile(np.arange(4), 12)
    rng.shuffle(train_y)
    test_x = rng.normal(size=(13, 160))
    parameters = (0.01, 0.1, 1.0, 10.0)
    optimized = _logistic_predict_many(
        train_x, train_y, test_x, parameters
    )
    for parameter in parameters:
        expected = LogisticRegression(
            C=parameter, max_iter=3000, class_weight="balanced"
        ).fit(train_x, train_y).predict(test_x)
        assert np.array_equal(optimized[parameter], expected)


def test_peak_summary_supports_categorical_only_scores():
    scores = pd.DataFrame([
        {
            "model": "model-a", "target": "tone", "task": "categorical",
            "turn_range": "00-25%", "layer": 2,
            "activation_only_score": 0.6, "n": 32,
            "n_conversations": 4, "n_topics": 4,
            "activation_pooling": "generated_response_token_mean",
            "cv_group": "topic_id",
        },
        {
            "model": "model-a", "target": "tone", "task": "categorical",
            "turn_range": "00-25%", "layer": 5,
            "activation_only_score": 0.7, "n": 32,
            "n_conversations": 4, "n_topics": 4,
            "activation_pooling": "generated_response_token_mean",
            "cv_group": "topic_id",
        },
    ])
    peaks = summarize_peak_layers(scores)
    assert peaks.iloc[0]["peak_layer"] == 5


def test_e1_cell_checkpoints_resume(tmp_path):
    rows = []
    for topic in range(4):
        for turn in range(8):
            value = float(turn + topic / 10)
            rows.append({
                "conv_id": f"conv-{topic}",
                "topic_id": f"topic-{topic}",
                "condition": "self_play",
                "speaker": "a" if turn % 2 == 0 else "b",
                "model": "model-a",
                "role": "supporter" if turn % 2 == 0 else "opposer",
                "turn": turn,
                "agent_turn": turn // 2 + 1,
                "conversation_turn_pct": 50.0,
                "turn_range": "00-100%",
                "stance_score": value,
                "layer_2": np.array([value, -value, topic], dtype=float),
            })
    frame = pd.DataFrame(rows)
    checkpoint_dir = tmp_path / "checkpoints"
    first = run_e1(
        frame, targets=["stance_score"], checkpoint_dir=checkpoint_dir,
        checkpoint_key="test-key", n_jobs=1,
    )
    events = []
    second = run_e1(
        frame, targets=["stance_score"], checkpoint_dir=checkpoint_dir,
        checkpoint_key="test-key", n_jobs=1,
        progress=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    assert len(list(checkpoint_dir.glob("*.pkl"))) == 1
    assert any(
        args and args[0] == "e1_checkpoint"
        and details.get("status") == "resumed"
        for args, details in events
    )
    assert first[0].equals(second[0])
