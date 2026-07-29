import numpy as np
import pytest

from src.common.embeddings import (
    basin_separation_score,
    partnerward_pull,
    topic_center,
)


def test_topic_center_zeroes_each_group_mean():
    values = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    centered = topic_center(values, ["a", "a", "b", "b"])
    np.testing.assert_allclose(centered[:2].mean(axis=0), 0.0)
    np.testing.assert_allclose(centered[2:].mean(axis=0), 0.0)


def test_basin_separation_requires_two_groups():
    with pytest.raises(ValueError, match="at least two groups"):
        basin_separation_score(np.array([[0.0], [1.0]]), ["only", "only"])


def test_basin_score_and_partnerward_pull():
    endpoints = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]])
    scores = basin_separation_score(endpoints, ["a", "a", "b", "b"])
    assert scores["a"]["S_basin"] > 1
    alpha, off_axis = partnerward_pull(
        np.array([0.0, 0.0]), np.array([0.5, 0.2]), np.array([1.0, 0.0])
    )
    assert alpha == pytest.approx(0.5)
    assert off_axis == pytest.approx(0.2)

