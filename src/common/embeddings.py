"""
Text-embedding geometry utilities, replicating Ko & Geiping (2026) Sec. 4.1:
topic-centered SBERT embeddings, PCA fit on a reference set ("SP-PCs" in the
paper, fit on self-play), and the basin separation score (their Eq. 5).

Used directly by Track 2 (human-AI trajectories), and available to Track 1
as an optional comparison against activation-space geometry.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


_sbert_model = None


def get_sbert_model(name: str = "all-MiniLM-L6-v2"):
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer(name)
    return _sbert_model


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = get_sbert_model(model_name)
    return np.asarray(model.encode(texts, show_progress_bar=False))


def topic_center(embeddings: np.ndarray, group_ids: list) -> np.ndarray:
    """Eq. 1: subtract the per-group (e.g. per-topic, or per-conversation)
    mean embedding from each embedding, removing group-level offset while
    keeping variation associated with speaker/model/turn."""
    embeddings = np.asarray(embeddings)
    group_ids = np.asarray(group_ids)
    centered = np.zeros_like(embeddings)
    for g in np.unique(group_ids):
        mask = group_ids == g
        centered[mask] = embeddings[mask] - embeddings[mask].mean(axis=0, keepdims=True)
    return centered


def fit_reference_pca(reference_embeddings: np.ndarray, n_components: int = 2) -> PCA:
    """Fit PCA on a reference embedding set (e.g. self-play / AI-only turns),
    analogous to the paper's SP-PCs, so other trajectories can be projected
    into a common, interpretable basis."""
    pca = PCA(n_components=n_components)
    pca.fit(reference_embeddings)
    return pca


def basin_separation_score(endpoints: np.ndarray, group_ids: list) -> dict:
    """Replicates Eq. 2-5: for each group (model), compare within-group
    endpoint spread to the squared-distance to the nearest other group's
    endpoint set. Returns {group: S_basin} plus the raw components, so you
    can sanity-check S_basin > 1 the way Table 1 does.

    endpoints: (N, D) array of one endpoint embedding per conversation
    group_ids: length-N list of group labels (e.g. AI model name)
    """
    endpoints = np.asarray(endpoints)
    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)

    if len(groups) < 2:
        raise ValueError(
            "basin_separation_score requires endpoints from at least two groups; "
            f"received {len(groups)} group(s)."
        )

    centroids = {g: endpoints[group_ids == g].mean(axis=0) for g in groups}

    within_spread = {}
    for g in groups:
        pts = endpoints[group_ids == g]
        within_spread[g] = float(np.mean(np.sum((pts - centroids[g]) ** 2, axis=1)))

    results = {}
    for g in groups:
        pts_g = endpoints[group_ids == g]
        best_d2 = np.inf
        nearest = None
        for h in groups:
            if h == g:
                continue
            pts_h = endpoints[group_ids == h]
            # mean squared pairwise distance between the two endpoint sets
            d2 = np.mean(
                np.sum(
                    (pts_g[:, None, :] - pts_h[None, :, :]) ** 2, axis=-1
                )
            )
            if d2 < best_d2:
                best_d2 = d2
                nearest = h
        w = within_spread[g] if within_spread[g] > 0 else 1e-8
        results[str(g)] = {
            "S_basin": float(best_d2 / w),
            "within_spread": within_spread[g],
            "nearest_group": str(nearest),
            "n": int((group_ids == g).sum()),
        }
    return results


def partnerward_pull(
    endpoint_a_alone: np.ndarray,
    endpoint_a_with_b: np.ndarray,
    endpoint_b_alone: np.ndarray,
) -> tuple[float, float]:
    """Eq. 7-10 analogue: decompose A's endpoint-when-paired-with-B relative
    to the axis from A's own anchor to B's anchor. Returns (alpha, off_axis
    magnitude). Works for any pair of anchor embeddings, e.g. AI's own first
    turn vs. AI's endpoint-with-human in Track 2's accommodation analysis, or
    self-play vs mixed-play endpoints if you have both.
    """
    v = endpoint_b_alone - endpoint_a_alone
    denom = float(np.dot(v, v))
    if denom < 1e-12:
        return 0.0, 0.0
    diff = endpoint_a_with_b - endpoint_a_alone
    alpha = float(np.dot(diff, v) / denom)
    proj = alpha * v
    off_axis = diff - proj
    off_axis_norm = float(np.linalg.norm(off_axis) / np.linalg.norm(v))
    return alpha, off_axis_norm
