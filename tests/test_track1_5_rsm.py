import numpy as np
import pandas as pd

from src.track1_5_rsm.analysis import (
    benjamini_hochberg,
    categorical_rsm,
    continuous_rsm,
    cosine_cross_similarity,
    evenly_spaced,
    fit_alignment,
    rsa_correlation,
)


def test_evenly_spaced_layers_include_endpoints():
    assert evenly_spaced([1, 3, 5, 7, 9, 11], 4) == [1, 5, 7, 11]


def test_alignment_recovers_rotated_common_space():
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(80, 5))
    qa, _ = np.linalg.qr(rng.normal(size=(12, 5)))
    qb, _ = np.linalg.qr(rng.normal(size=(15, 5)))
    values_a = latent @ qa.T
    values_b = latent @ qb.T
    alignment = fit_alignment(values_a[:60], values_b[:60], max_rank=5)
    projected_a = alignment.transform_a(values_a[60:])
    projected_b = alignment.transform_b(values_b[60:])
    similarity = cosine_cross_similarity(projected_a, projected_b)
    assert np.nanmean(np.diag(similarity)) > 0.99
    assert np.nanmean(np.diag(similarity)) > np.nanmean(similarity)


def test_variable_rsms_are_bounded_and_missing_aware():
    continuous, metadata = continuous_rsm(
        np.array([1.0, 2.0, np.nan]), np.array([1.0, 3.0])
    )
    assert metadata["centre"] == 1.75
    assert continuous[0, 0] == 1.0
    assert np.isnan(continuous[2]).all()
    assert np.nanmin(continuous) >= 0.0
    assert np.nanmax(continuous) <= 1.0

    categorical, metadata = categorical_rsm(
        np.array(["a", "b", None], dtype=object),
        np.array(["b", "a"], dtype=object),
    )
    assert metadata["levels"] == ["a", "b"]
    assert categorical[0, 1] == 1.0
    assert categorical[1, 0] == 1.0
    assert np.isnan(categorical[2]).all()


def test_rsa_and_fdr():
    rng = np.random.default_rng(9)
    values = np.linspace(-2, 2, 20)
    variable, _ = continuous_rsm(values, values)
    rho, p_value, n = rsa_correlation(
        variable, variable, permutations=99, rng=rng
    )
    assert rho == 1.0
    assert p_value <= 0.02
    assert n == 400
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.04, np.nan]))
    assert np.allclose(adjusted.iloc[:2], [0.02, 0.04])
    assert np.isnan(adjusted.iloc[2])

