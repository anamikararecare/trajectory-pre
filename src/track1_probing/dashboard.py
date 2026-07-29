"""Read-only Streamlit dashboard for replayed Track 1 activations.

Launch with:
    streamlit run src/track1_probing/dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.decomposition import PCA

from src.track1_probing.dashboard_data import (
    ACTIVATION_STATUSES,
    SNAPSHOT_ORDER,
    build_turn_table,
    discover_replay_runs,
    discover_result_runs,
    load_replay_vectors,
    reconstructed_prompt,
    replay_overview,
)
from src.track1_probing.dashboard_help import (
    render_all_experiment_guides,
    render_experiment_guide,
    render_figure_guide,
    render_metric_glossary,
    render_target_definition,
)
from src.track2_human_ai.dashboard import render_track2_page

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Trajectory Research Dashboard",
    page_icon="🧭",
    layout="wide",
)

STATUS_COLORS = {
    "original": "#4c78a8",
    "replayed_validated": "#2ca02c",
    "replayed_warning": "#f2a93b",
    "unavailable": "#b8b8b8",
}
SNAPSHOT_LABELS = {
    "pre_generation": "Pre-generation",
    "early_response": "Early response",
    "full_response": "Full response",
    "final_window": "Final window",
    "final_token": "Final token",
}


@st.cache_data(show_spinner=False)
def load_bundle(
    data_dir: str,
    replay_dir: str,
    results_dir: str,
    geometry_path: str,
    annotations_path: str,
) -> dict:
    return build_turn_table(
        data_dir=data_dir,
        replay_dir=replay_dir,
        results_dir=results_dir or None,
        geometry_path=geometry_path or None,
        annotations_path=annotations_path or None,
    )


@st.cache_data(show_spinner=False)
def load_activation_slice(replay_dir: str, model: str, layer: int) -> pd.DataFrame:
    return load_replay_vectors(replay_dir, model, layer)


def artifact_notice(frame: pd.DataFrame, filename: str, purpose: str) -> bool:
    if frame is not None and not frame.empty:
        return True
    st.info(
        f"{purpose} is unavailable for this run. Expected artifact: {filename}. "
        "New overnight runs create it automatically."
    )
    return False


def ordered_snapshots(values) -> list[str]:
    present = set(values)
    return [snapshot for snapshot in SNAPSHOT_ORDER if snapshot in present]


def boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a false-filled boolean Series for new columns absent in old runs."""
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def heatmap(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    value: str,
    title: str,
    column_order: list | None = None,
) -> None:
    pivot = frame.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    if column_order:
        pivot = pivot.reindex(columns=[column for column in column_order if column in pivot])
    pivot = pivot.sort_index()
    figure = go.Figure(go.Heatmap(
        z=pivot.to_numpy(),
        x=[SNAPSHOT_LABELS.get(item, item) for item in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="RdBu", zmid=0,
        colorbar={"title": value.replace("_", " ")},
        text=np.round(pivot.to_numpy(), 3),
        texttemplate="%{text}",
        hovertemplate=f"{index}: %{{y}}<br>{columns}: %{{x}}<br>{value}: %{{z:.4f}}<extra></extra>",
    ))
    figure.update_layout(title=title, xaxis_title=columns, yaxis_title=index, height=460)
    st.plotly_chart(figure, width="stretch")


def score_filters(scores: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if scores.empty:
        return scores
    columns = st.columns(4)
    experiment_values = sorted(scores["experiment"].dropna().unique())
    experiment = columns[0].selectbox(
        "Experiment", experiment_values, key=f"{prefix}_experiment"
    )
    subset = scores[scores["experiment"].eq(experiment)]
    targets = sorted(subset["target"].dropna().unique())
    target = columns[1].selectbox("Target", targets, key=f"{prefix}_target")
    subset = subset[subset["target"].eq(target)]
    horizons = sorted(subset["horizon"].dropna().unique())
    horizon = columns[2].selectbox("Horizon", horizons, key=f"{prefix}_horizon")
    subset = subset[subset["horizon"].eq(horizon)]
    metrics = sorted(subset["metric"].dropna().unique())
    metric = columns[3].selectbox("Metric", metrics, key=f"{prefix}_metric")
    return subset[subset["metric"].eq(metric)]


def render_matched_sensitivity(bundle: dict, selector: pd.Series | None = None) -> None:
    sensitivity = bundle.get("sensitivity", pd.DataFrame())
    if sensitivity.empty:
        st.warning(
            "This completed run predates original_replay_sensitivity.csv. Vector-level "
            "replay validation is available below, but a fold-matched probe sensitivity "
            "result must not be inferred from the legacy probe table."
        )
        return
    if selector is not None:
        sensitivity = sensitivity[
            sensitivity["experiment"].eq(selector["experiment"])
            & sensitivity["target"].eq(selector["target"])
            & sensitivity["horizon"].eq(selector["horizon"])
            & sensitivity["metric"].eq(selector["metric"])
        ]
    if sensitivity.empty:
        st.info("No matched sensitivity rows exist for this selection.")
        return
    long = sensitivity.melt(
        id_vars=["experiment", "target", "horizon", "layer", "metric"],
        value_vars=["original_full_response_score", "replayed_full_response_score"],
        var_name="activation_source", value_name="score",
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.line(
            long, x="layer", y="score", color="activation_source", markers=True,
            facet_col="target", title="Same rows, folds, baseline, and probe specification",
        ),
        width="stretch",
    )
    right.plotly_chart(
        px.bar(
            sensitivity, x="layer", y="replayed_minus_original", color="target",
            barmode="group", title="Replayed minus original full-response score",
        ),
        width="stretch",
    )
    st.dataframe(sensitivity, width="stretch", hide_index=True)


def render_overview(bundle: dict, replay_dir: str, results_dir: str) -> None:
    manifest = bundle["manifest"]
    overview = replay_overview(manifest, bundle["replay_index"], bundle["warnings"])
    st.header("Run overview and provenance")
    render_experiment_guide("1A")
    render_all_experiment_guides()
    metric_columns = st.columns(6)
    metric_columns[0].metric("Replay ID", overview["replay_id"] or "unknown")
    metric_columns[1].metric("Eligible turns", overview["eligible_turns"])
    metric_columns[2].metric("Completed turns", overview["completed_turns"])
    metric_columns[3].metric("Failed / missing", overview["failed_or_missing_turns"])
    metric_columns[4].metric("Alignment warnings", overview["alignment_warning_count"])
    metric_columns[5].metric("Snapshot arrays", sum(overview["snapshot_counts"].values()))

    st.caption(f"Replay directory: {replay_dir} · Results directory: {results_dir or 'not selected'}")
    snapshot_frame = pd.DataFrame([
        {"snapshot": SNAPSHOT_LABELS.get(snapshot, snapshot), "arrays": count}
        for snapshot, count in overview["snapshot_counts"].items()
    ])
    left, right = st.columns([1, 2])
    left.plotly_chart(
        px.bar(snapshot_frame, x="snapshot", y="arrays", title="Snapshot coverage"),
        width="stretch",
    )

    model_rows = []
    for model, details in manifest.get("models", {}).items():
        validation = details.get("validation", {})
        model_rows.append({
            "model": model,
            "hf_id": details.get("hf_id"),
            "model_revision": details.get("model_revision"),
            "tokenizer_revision": details.get("tokenizer_revision"),
            "layers": ", ".join(map(str, details.get("layers", []))),
            "validation_status": validation.get("status"),
            "validation_n": validation.get("n", 0),
            "median_cosine": validation.get("median_cosine_similarity"),
            "p05_cosine": validation.get("p05_cosine_similarity"),
            "median_relative_error": validation.get("median_relative_error"),
            "median_norm_ratio": validation.get("median_norm_ratio"),
        })
    right.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

    provenance = {
        "sampling disabled": manifest.get("sampling_disabled"),
        "frozen corpus": manifest.get("frozen_corpus"),
        "response source": manifest.get("response_source"),
        "original pooling": manifest.get("original_pooling"),
        "primary pooling": manifest.get("primary_pooling"),
        "precision": manifest.get("precision"),
        "quantization": manifest.get("quantization"),
        "code revision": manifest.get("code_revision"),
    }
    st.subheader("Replay provenance")
    st.json(provenance)

    metadata = bundle["metadata"]
    hashes = (
        sorted(metadata["chat_template_sha256"].dropna().unique())
        if not metadata.empty and "chat_template_sha256" in metadata else []
    )
    st.write("Chat-template hashes:", hashes or "unavailable")

    eligibility = pd.DataFrame(manifest.get("eligibility", []))
    if not eligibility.empty:
        summary = eligibility.groupby(
            ["model", "speaker", "eligible", "reason"], dropna=False
        ).size().reset_index(name="turns")
        st.subheader("Eligibility by model and speaker")
        st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Activation provenance labels")
    status_counts = (
        bundle["coverage"]["activation_status"].value_counts()
        .reindex(ACTIVATION_STATUSES, fill_value=0)
        .rename_axis("status").reset_index(name="cells")
    )
    figure = px.bar(
        status_counts, x="status", y="cells", color="status",
        color_discrete_map=STATUS_COLORS, title="Original, replayed, warned, and unavailable cells",
    )
    st.plotly_chart(figure, width="stretch")

    st.caption(
        f"Eligible models: {', '.join(overview['eligible_models']) or 'none'} · "
        f"Eligible speakers: {', '.join(overview['eligible_speakers']) or 'none'}"
    )
    scores = bundle.get("scores", pd.DataFrame())
    skipped = bundle.get("skipped", pd.DataFrame())
    st.subheader("Analysis artifact coverage")
    if not scores.empty:
        analyzed = (
            scores.groupby(["experiment", "target"], as_index=False)
            .agg(rows=("snapshot", "size"), snapshots=("snapshot", "nunique"), layers=("layer", "nunique"))
        )
        st.plotly_chart(
            px.bar(
                analyzed, x="experiment", y="rows", color="target",
                title="Completed probe rows by experiment and target",
                hover_data=["snapshots", "layers"],
            ),
            width="stretch",
        )
    else:
        st.warning("No snapshot probe output is attached to this replay run.")
    if not skipped.empty:
        skipped_counts = skipped.groupby(["experiment", "reason"]).size().reset_index(name="targets")
        st.plotly_chart(
            px.bar(
                skipped_counts, x="experiment", y="targets", color="reason",
                title="Targets not analyzed and why",
            ),
            width="stretch",
        )
        with st.expander("Exact skipped-variable inventory"):
            st.dataframe(skipped, width="stretch", hide_index=True)

    audit = bundle.get("audit", pd.DataFrame())
    if not audit.empty:
        with st.expander("Measurement and coverage audit"):
            st.dataframe(audit, width="stretch", hide_index=True)
    artifact_manifest = bundle.get("artifact_manifest", pd.DataFrame())
    if not artifact_manifest.empty:
        st.subheader("Experiment 1A–1I CSV and PNG contract")
        st.dataframe(artifact_manifest, width="stretch", hide_index=True)


def render_conversation(bundle: dict) -> None:
    st.header("Conversation explorer")
    turns = bundle["turns"]
    if turns.empty:
        st.warning("No transcript turns found.")
        return
    conversations = sorted(turns["conv_id"].unique())
    conv_id = st.selectbox("Conversation", conversations)
    conversation = turns[turns["conv_id"].eq(conv_id)].sort_values("turn")
    turn_number = st.select_slider("Turn", options=conversation["turn"].tolist())
    selected = conversation[conversation["turn"].eq(turn_number)].iloc[0]
    key = (conv_id, int(turn_number), selected["speaker"])

    columns = st.columns(4)
    columns[0].metric("Speaker", selected["speaker"])
    columns[1].metric("Model", selected["model"])
    columns[2].metric("Role", selected["role"])
    columns[3].metric("Topic", selected["topic_id"])

    transcript = bundle["transcripts"][conv_id]
    turn_index = next(
        index for index, turn in enumerate(transcript["turns"])
        if int(turn["turn"]) == int(turn_number)
    )
    with st.expander("Reconstructed prompt boundary", expanded=False):
        prompt = reconstructed_prompt(transcript, turn_index)
        for message in prompt:
            st.markdown(f"**{message['role']}**")
            st.text(message["content"])
        response_start = selected.get("response_start_index")
        prompt_boundary = int(response_start) - 1 if pd.notna(response_start) else "unavailable"
        st.caption(f"Final prompt token index: {prompt_boundary}")

    st.subheader("Recorded response and token boundary")
    st.text(selected["text"])
    boundary_columns = [
        "prompt_token_count", "response_start_index", "response_end_index",
        "response_token_count", "early_response_window", "final_window",
        "response_special_tokens_included", "eos_included",
    ]
    boundary = pd.DataFrame([
        {"field": column, "value": selected.get(column)}
        for column in boundary_columns
    ])
    st.dataframe(boundary, width="stretch", hide_index=True)

    coverage = bundle["coverage"]
    turn_coverage = coverage[
        coverage["conv_id"].eq(conv_id)
        & coverage["turn"].eq(turn_number)
        & coverage["speaker"].eq(selected["speaker"])
    ]
    st.subheader("Snapshot availability")
    if not turn_coverage.empty:
        display = turn_coverage.copy()
        display["snapshot"] = display["snapshot"].map(SNAPSHOT_LABELS)
        st.dataframe(
            display[["layer", "snapshot", "window", "activation_status"]],
            width="stretch", hide_index=True,
        )

    similarity_columns = st.columns(3)
    similarity_columns[0].metric(
        "Original/replayed cosine",
        f"{selected.get('replay_cosine_similarity'):.6f}"
        if pd.notna(selected.get("replay_cosine_similarity")) else "not sampled",
    )
    similarity_columns[1].metric(
        "Relative error",
        f"{selected.get('replay_relative_error'):.6g}"
        if pd.notna(selected.get("replay_relative_error")) else "not sampled",
    )
    similarity_columns[2].metric(
        "Norm ratio",
        f"{selected.get('replay_norm_ratio'):.6f}"
        if pd.notna(selected.get("replay_norm_ratio")) else "not sampled",
    )

    big_five_columns = [
        column for column in selected.index
        if column.startswith("observer_big5_")
        and not column.endswith(("_confidence", "_self_play_baseline", "_deviation_from_self_play", "_movement"))
    ]
    if big_five_columns:
        st.subheader("Direct observer-rated Big Five conversational presentation")
        rows = []
        for column in sorted(big_five_columns):
            root = column.removesuffix("_trailing3")
            rows.append({
                "trait": root.removeprefix("observer_big5_"),
                "current_response_score": selected.get(root),
                "evidence_confidence": selected.get(f"{root}_confidence"),
                "trailing_three_state": selected.get(f"{root}_trailing3"),
                "self_play_baseline": selected.get(f"{root}_self_play_baseline"),
                "deviation_from_self_play": selected.get(f"{root}_deviation_from_self_play"),
                "movement": selected.get(f"{root}_movement"),
            })
        st.caption("Direct 0–4 observer scores; these are not BFI-44 inventory scores.")
        st.dataframe(pd.DataFrame(rows).drop_duplicates("trait"), width="stretch", hide_index=True)

    oof = bundle.get("oof", pd.DataFrame())
    if not artifact_notice(oof, "oof_predictions.csv", "Turn-level OOF predictions"):
        return
    selected_oof = oof[
        oof["conv_id"].eq(conv_id)
        & oof["turn"].eq(turn_number)
        & oof["speaker"].eq(selected["speaker"])
    ]
    if selected_oof.empty:
        st.info("No OOF prediction row is available for this selected turn.")
        return
    target = st.selectbox("Prediction target", sorted(selected_oof["target"].unique()))
    render_target_definition(target)
    progression = selected_oof[selected_oof["target"].eq(target)].copy()
    st.subheader("Pre-generation → early → full → final prediction progression")
    progression["snapshot_order"] = progression["snapshot"].map(
        {snapshot: index for index, snapshot in enumerate(SNAPSHOT_ORDER)}
    )
    progression = progression.sort_values(["layer", "snapshot_order"])
    figure = px.line(
        progression, x="snapshot", y="activation_prediction_code",
        color="layer", markers=True,
        category_orders={"snapshot": SNAPSHOT_ORDER},
        hover_data=[
            "baseline_prediction", "activation_prediction", "observed_target",
            "prospectively_predictive_eligible",
        ],
        title=f"Layerwise OOF prediction progression · {target}",
    )
    st.plotly_chart(figure, width="stretch")
    render_figure_guide("snapshot_progression", "How to read prediction progression")
    st.dataframe(
        progression[[
            "experiment", "target", "horizon", "layer", "snapshot",
            "observed_target", "baseline_prediction", "activation_prediction",
            "time_legal_baseline", "prospectively_predictive_eligible",
        ]],
        width="stretch", hide_index=True,
    )


def render_probes(bundle: dict, results_root: str) -> None:
    st.header("Snapshot-resolved probe dashboard")
    render_metric_glossary()
    scores = bundle.get("scores", pd.DataFrame())
    if not artifact_notice(scores, "snapshot_probe_scores.csv", "Snapshot probe scores"):
        return
    subset = score_filters(scores, "probe")
    if subset.empty:
        st.info("No rows match the selected probe filters.")
        return

    prospective = subset[
        boolean_column(subset, "prospectively_predictive_eligible")
        & boolean_column(subset, "time_legal_baseline")
    ]
    if not prospective.empty:
        st.success(
            "Prospectively predictive eligibility is shown only for pre-generation/"
            "early snapshots with time-legal text baselines."
        )
    else:
        st.warning("The selected result is not eligible for the prospectively predictive label.")

    selector = subset.iloc[0]
    selected_experiment = str(selector["experiment"])
    render_experiment_guide(
        selected_experiment,
        scores[scores["experiment"].eq(selected_experiment)]["target"].dropna().unique(),
    )
    render_target_definition(str(selector["target"]))
    horizon_scope = scores[
        scores["experiment"].eq(selector["experiment"])
        & scores["target"].eq(selector["target"])
        & scores["metric"].eq(selector["metric"])
    ]
    first, second = st.columns(2)
    with first:
        heatmap(
            subset, "layer", "snapshot", "activation_plus_baseline_score",
            "Layer × snapshot performance", SNAPSHOT_ORDER,
        )
        render_figure_guide("absolute_heatmap", "How to read layer × snapshot performance")
    with second:
        heatmap(
            horizon_scope, "snapshot", "horizon", "incremental_score",
            "Snapshot × horizon incremental lift",
        )
        render_figure_guide("incremental_heatmap", "How to read snapshot × horizon lift")

    pivot = subset.pivot_table(
        index=["experiment", "target", "horizon", "layer"],
        columns="snapshot", values="activation_plus_baseline_score", aggfunc="mean",
    ).reset_index()
    if {"pre_generation", "full_response"}.issubset(pivot):
        pivot["pre_minus_full"] = pivot["pre_generation"] - pivot["full_response"]
        st.subheader("Pre-generation versus full-response difference")
        st.plotly_chart(
            px.bar(pivot, x="layer", y="pre_minus_full", color="horizon", barmode="group"),
            width="stretch",
        )
        render_figure_guide("snapshot_progression", "How to read pre-generation minus full-response")

    comparison = subset.melt(
        id_vars=["layer", "snapshot", "horizon"],
        value_vars=["baseline_score", "activation_plus_baseline_score"],
        var_name="model_input", value_name="score",
    )
    st.subheader("With and without time-matched activation")
    st.plotly_chart(
        px.line(
            comparison, x="layer", y="score", color="model_input",
            facet_col="snapshot", markers=True,
            category_orders={"snapshot": SNAPSHOT_ORDER},
        ),
        width="stretch",
    )
    render_figure_guide("baseline_comparison", "How to read with/without activation")

    coverage = bundle["coverage"]
    st.subheader("Speaker-A and speaker-B snapshot coverage")
    speaker_coverage = (
        coverage[coverage["has_replay"]]
        .groupby(["speaker", "snapshot"]).size().reset_index(name="arrays")
    )
    st.plotly_chart(
        px.bar(
            speaker_coverage, x="snapshot", y="arrays", color="speaker",
            barmode="group", category_orders={"snapshot": SNAPSHOT_ORDER},
        ),
        width="stretch",
    )
    render_figure_guide("coverage", "How to read snapshot coverage")

    folds = bundle.get("folds", pd.DataFrame())
    if artifact_notice(folds, "fold_scores.csv", "Paired fold-level comparisons"):
        fold_subset = folds[
            folds["experiment"].eq(selector["experiment"])
            & folds["target"].eq(selector["target"])
            & folds["horizon"].eq(selector["horizon"])
            & folds["metric"].eq(selector["metric"])
        ]
        snapshots = ordered_snapshots(fold_subset["snapshot"].unique())
        if len(snapshots) >= 2:
            pair_columns = st.columns(2)
            left_snapshot = pair_columns[0].selectbox(
                "Paired snapshot A", snapshots, index=0
            )
            right_snapshot = pair_columns[1].selectbox(
                "Paired snapshot B", snapshots, index=len(snapshots) - 1
            )
            paired = fold_subset.pivot_table(
                index=["held_out_topic", "layer"], columns="snapshot",
                values="incremental_score", aggfunc="mean",
            ).reset_index()
            if {left_snapshot, right_snapshot}.issubset(paired):
                paired["paired_difference"] = (
                    paired[left_snapshot] - paired[right_snapshot]
                )
                st.plotly_chart(
                    px.box(
                        paired, x="layer", y="paired_difference", points="all",
                        hover_data=["held_out_topic"],
                        title=f"Paired fold lift: {left_snapshot} − {right_snapshot}",
                    ),
                    width="stretch",
                )
                render_figure_guide("fold_difference", "How to read paired held-out differences")

    st.subheader("Original versus replayed full-response sensitivity")
    render_matched_sensitivity(bundle, selector)


def render_activation_atlas(bundle: dict, replay_dir: str) -> None:
    """Visualize replayed hidden-state geometry without fitting a behavioral probe."""
    st.header("Activation atlas")
    index = bundle["replay_index"]
    if index.empty:
        st.warning(
            "This replay run contains no activation arrays. Select the completed full replay, "
            "not a validation-only gate run."
        )
        return

    controls = st.columns(2)
    models = sorted(index["model"].dropna().unique())
    model = controls[0].selectbox("Model", models, key="atlas_model")
    layers = sorted(index[index["model"].eq(model)]["layer"].unique())
    layer = controls[1].selectbox("Layer", layers, key="atlas_layer")
    frame = load_activation_slice(replay_dir, model, int(layer))
    if frame.empty:
        st.warning("No vectors exist for this model/layer selection.")
        return

    turns = bundle["turns"]
    metadata = [
        column for column in (
            "conv_id", "turn", "speaker", "model", "role", "condition", "topic_id",
            "agent_turn", "stance_score", "partner_model",
        ) if column in turns
    ]
    frame = frame.merge(
        turns[metadata].drop_duplicates(["conv_id", "turn", "speaker", "model"]),
        on=["conv_id", "turn", "speaker", "model"], how="left", validate="many_to_one",
    )
    status = (
        bundle["manifest"].get("models", {}).get(model, {})
        .get("validation", {}).get("status", "not_evaluated")
    )
    if status == "passed":
        st.success(f"{model} replay validation passed; vectors are replayed_validated.")
    else:
        st.warning(
            f"{model} replay status is {status}; plots are sensitivity views and vectors are "
            "replayed_warning, not primary validated evidence."
        )

    matrix = np.stack(frame["activation"])
    coordinates = PCA(n_components=2, random_state=0).fit_transform(matrix)
    frame["activation_pc1"] = coordinates[:, 0]
    frame["activation_pc2"] = coordinates[:, 1]
    frame["snapshot_label"] = frame["snapshot"].map(SNAPSHOT_LABELS)

    first, second = st.columns(2)
    first.plotly_chart(
        px.box(
            frame, x="snapshot_label", y="activation_norm", color="speaker", points=False,
            category_orders={"snapshot_label": list(SNAPSHOT_LABELS.values())},
            title="Activation norm by snapshot and speaker",
        ),
        width="stretch",
    )
    with first:
        render_figure_guide("activation_norm", "How to read activation norms")
    second.plotly_chart(
        px.scatter(
            frame, x="activation_pc1", y="activation_pc2", color="condition",
            symbol="snapshot_label", opacity=0.35, render_mode="webgl",
            hover_data=["conv_id", "turn", "speaker", "role", "topic_id"],
            title="Direct activation PCA (no behavioral labels)",
        ),
        width="stretch",
    )
    with second:
        render_figure_guide("activation_pca", "How to read activation PCA")

    centroids = (
        frame.groupby(["snapshot", "snapshot_label", "speaker"], as_index=False)
        [["activation_pc1", "activation_pc2", "activation_norm"]].mean()
    )
    centroids["snapshot_order"] = centroids["snapshot"].map(
        {snapshot: position for position, snapshot in enumerate(SNAPSHOT_ORDER)}
    )
    centroids = centroids.sort_values(["speaker", "snapshot_order"])
    st.plotly_chart(
        px.line(
            centroids, x="activation_pc1", y="activation_pc2", color="speaker",
            text="snapshot_label", markers=True,
            category_orders={"snapshot": SNAPSHOT_ORDER},
            title="Centroid progression: pre-generation → early → full → final",
        ),
        width="stretch",
    )
    render_figure_guide("activation_pca", "How to read centroid progression")

    conversations = sorted(frame["conv_id"].dropna().unique())
    conv_id = st.selectbox("Conversation trajectory", conversations, key="atlas_conv")
    conversation = frame[frame["conv_id"].eq(conv_id)]
    turn = st.select_slider(
        "Turn trajectory", options=sorted(conversation["turn"].unique()), key="atlas_turn"
    )
    trajectory = conversation[conversation["turn"].eq(turn)].copy()
    trajectory["snapshot_order"] = trajectory["snapshot"].map(
        {snapshot: position for position, snapshot in enumerate(SNAPSHOT_ORDER)}
    )
    trajectory = trajectory.sort_values("snapshot_order")
    st.plotly_chart(
        px.line(
            trajectory, x="activation_pc1", y="activation_pc2", text="snapshot_label",
            markers=True, hover_data=["activation_norm", "speaker", "role"],
            title="Selected turn hidden-state path",
        ),
        width="stretch",
    )
    render_figure_guide("activation_pca", "How to read the selected hidden-state path")
    st.caption(
        "PCA is fit only for the selected model and layer. It is an exploratory view of "
        "hidden-state geometry, independent of annotation labels and probe targets."
    )


def _quality_rows(oof: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return oof
    joined = oof.merge(
        turns[["conv_id", "turn", "speaker", "partner_model"]].drop_duplicates(),
        on=["conv_id", "turn", "speaker"], how="left",
    )
    classification = joined["metric"].eq("balanced_accuracy")
    joined["baseline_quality"] = np.where(
        classification,
        joined["baseline_prediction_code"].eq(joined["observed_code"]).astype(float),
        -np.abs(joined["baseline_prediction_code"] - joined["observed_code"]),
    )
    joined["activation_quality"] = np.where(
        classification,
        joined["activation_prediction_code"].eq(joined["observed_code"]).astype(float),
        -np.abs(joined["activation_prediction_code"] - joined["observed_code"]),
    )
    joined["activation_lift"] = joined["activation_quality"] - joined["baseline_quality"]
    joined["model_pair"] = joined["model"].astype(str) + " → " + joined["partner_model"].astype(str)
    return joined


def render_partner(bundle: dict) -> None:
    st.header("Immediate partner-reaction sequence")
    render_experiment_guide("1E")
    turns = bundle["turns"]
    mixed = turns[turns["condition"].eq("mixed_play")]
    if mixed.empty:
        st.info("No mixed-play turns are available.")
        return
    conv_id = st.selectbox("Dyad conversation", sorted(mixed["conv_id"].unique()), key="partner_conv")
    sequence = mixed[mixed["conv_id"].eq(conv_id)].sort_values("turn")
    source_turns = sequence.iloc[:-1]["turn"].tolist()
    if not source_turns:
        st.info("This dyad does not contain a complete A → B turn pair.")
        return
    source_turn = st.selectbox("Speaker-A source turn", source_turns)
    source_position = sequence.index.get_loc(sequence[sequence["turn"].eq(source_turn)].index[0])
    a = sequence.iloc[source_position]
    b = sequence.iloc[source_position + 1]
    columns = st.columns(4)
    columns[0].markdown(f"**A full-response state**\n\n{a['model']} · {a['speaker']}")
    columns[1].markdown(f"**B pre-generation state**\n\n{b['model']} · {b['speaker']}")
    columns[2].markdown(f"**B early-response state**\n\n{b['model']} · {b['speaker']}")
    columns[3].markdown(f"**B realized response**\n\nTurn {b['turn']}")
    st.markdown("→".join([" A full ", " B pre ", " B early ", " B response "]))
    with st.expander("A recorded response"):
        st.text(a["text"])
    with st.expander("B recorded response and labels"):
        st.text(b["text"])
        labels = {
            column: b[column] for column in turns
            if column in (
                "stance_score", "stance_change", "local_agreement", "affiliation",
                "adversariality", "realized_move", "closure_evidence",
            )
        }
        st.json(labels)

    oof = bundle.get("oof", pd.DataFrame())
    if not artifact_notice(oof, "oof_predictions.csv", "Partner-reaction OOF results"):
        return
    quality = _quality_rows(oof, turns)
    target_options = sorted(quality[quality["experiment"].isin(["1C", "1E"])]["target"].unique())
    if not target_options:
        st.info("No 1C/1E partner-reaction targets are present.")
        return
    target = st.selectbox("Reaction target", target_options, key="partner_target")
    render_target_definition(target)
    reaction = quality[quality["target"].eq(target)]
    summary = (
        reaction.groupby(
            ["experiment", "snapshot", "model_pair", "role", "topic_id"], dropna=False
        )[["baseline_quality", "activation_quality", "activation_lift"]]
        .mean().reset_index()
    )
    st.subheader("A activation → B reaction; B pre-generation → B reaction")
    st.plotly_chart(
        px.bar(
            summary, x="snapshot", y="activation_lift", color="experiment",
            facet_col="model_pair", barmode="group",
            category_orders={"snapshot": SNAPSHOT_ORDER},
            hover_data=["role", "topic_id", "baseline_quality", "activation_quality"],
        ),
        width="stretch",
    )
    render_figure_guide("partner_lift", "How to read partner-reaction lift")
    quality_long = summary.melt(
        id_vars=["experiment", "snapshot", "model_pair", "role", "topic_id"],
        value_vars=["baseline_quality", "activation_quality"],
        var_name="input", value_name="prediction_quality",
    )
    st.plotly_chart(
        px.bar(
            quality_long, x="snapshot", y="prediction_quality", color="input",
            facet_col="experiment", barmode="group",
            category_orders={"snapshot": SNAPSHOT_ORDER},
            title="Transcript baseline → reaction versus activation-augmented reaction",
        ),
        width="stretch",
    )
    render_figure_guide("baseline_comparison", "How to read reaction prediction quality")
    st.dataframe(summary, width="stretch", hide_index=True)

    transfer = bundle.get("transfer", pd.DataFrame())
    if not transfer.empty:
        st.subheader("Mixed-play deviation from same-model self-play")
        render_experiment_guide("1I")
        activation_transfer = transfer[transfer["kind"].eq("activation")].copy()
        behavior_transfer = transfer[transfer["kind"].eq("behavior")].copy()
        if not activation_transfer.empty:
            st.plotly_chart(
                px.line(
                    activation_transfer, x="layer", y="mixed_minus_self",
                    color="snapshot", facet_col="model", markers=True,
                    category_orders={"snapshot": SNAPSHOT_ORDER},
                    title="Partner-induced activation displacement by layer and snapshot",
                ),
                width="stretch",
            )
            render_figure_guide("transfer", "How to read activation transfer")
        if not behavior_transfer.empty:
            st.plotly_chart(
                px.bar(
                    behavior_transfer, x="target", y="mixed_minus_self", color="model",
                    barmode="group", title="Mixed-play behavioral shift from self-play",
                ),
                width="stretch",
            )
            render_figure_guide("transfer", "How to read behavioral transfer")


def render_target_progression(bundle: dict, title: str, target_pattern: str, key: str) -> None:
    st.header(title)
    scores = bundle.get("scores", pd.DataFrame())
    if not artifact_notice(scores, "snapshot_probe_scores.csv", f"{title} probe scores"):
        return
    targets = sorted([
        target for target in scores["target"].dropna().unique()
        if target_pattern in target
    ])
    if not targets:
        st.info(f"No target containing '{target_pattern}' is available.")
        skipped = bundle.get("skipped", pd.DataFrame())
        if not skipped.empty:
            unavailable = skipped[
                skipped["target"].astype(str).str.contains(target_pattern, regex=False)
            ]
            if not unavailable.empty:
                st.warning(
                    "These views cannot be plotted because their offline labels were not "
                    "present when the completed analysis ran."
                )
                st.dataframe(unavailable, width="stretch", hide_index=True)
        return
    target = st.selectbox("Target", targets, key=f"{key}_target")
    render_target_definition(target)
    render_metric_glossary()
    subset = scores[scores["target"].eq(target)]
    horizons = sorted(subset["horizon"].unique())
    horizon = st.selectbox("Horizon", horizons, key=f"{key}_horizon")
    subset = subset[subset["horizon"].eq(horizon)]
    heatmap(
        subset, "layer", "snapshot", "incremental_score",
        f"{title}: snapshot progression", SNAPSHOT_ORDER,
    )
    render_figure_guide("incremental_heatmap", "How to read the target heatmap")
    st.plotly_chart(
        px.line(
            subset, x="snapshot", y="activation_plus_baseline_score",
            color="layer", markers=True,
            category_orders={"snapshot": SNAPSHOT_ORDER},
            hover_data=["baseline_score", "incremental_score"],
        ),
        width="stretch",
    )
    render_figure_guide("snapshot_progression", "How to read target progression")


def render_basin(bundle: dict) -> None:
    render_experiment_guide("1H")
    render_target_progression(bundle, "Basin and semantic movement", "basin", "basin")
    turns = bundle["turns"]
    basin_columns = [
        column for column in (
            "basin_leaning", "partnerward_basin_velocity", "off_axis_distance",
            "semantic_velocity", "semantic_acceleration",
        ) if column in turns
    ]
    if basin_columns:
        st.subheader("Observed output-space trajectory")
        metric = st.selectbox("Geometry metric", basin_columns)
        render_target_definition(metric)
        subset = turns.dropna(subset=[metric])
        st.plotly_chart(
            px.line(
                subset, x="agent_turn", y=metric, color="model",
                line_dash="condition", facet_col="topic_id",
                facet_col_wrap=3, hover_data=["conv_id", "speaker", "role"],
            ),
            width="stretch",
        )
        render_figure_guide("observed_trajectory", "How to read output-space trajectories")


def render_persona_window(bundle: dict, prefix: str, key: str, label: str) -> None:
    turns = bundle["turns"]
    persona_state = [
        column for column in turns
        if column.startswith(prefix) and column.endswith("_trailing3")
    ]
    if not persona_state:
        st.info(f"{label} window views require the corresponding offline annotations.")
        return
    state = st.selectbox(f"{label} window", persona_state, key=f"{key}_window")
    render_target_definition(state)
    root = state.removesuffix("_trailing3")
    columns = [
        "conv_id", "turn", "speaker", "model", "partner_model", "condition",
        state, f"{root}_self_play_baseline",
        f"{root}_deviation_from_self_play", f"{root}_movement",
    ]
    columns = [column for column in columns if column in turns]
    st.subheader("Current window, self-play baseline, mixed deviation, and movement")
    st.dataframe(turns[columns].dropna(subset=[state]), width="stretch", hide_index=True)
    mixed = turns[turns["condition"].eq("mixed_play")].dropna(subset=[state])
    if not mixed.empty:
        st.plotly_chart(
            px.line(
                mixed, x="agent_turn", y=state, color="model",
                line_dash="partner_model", facet_col="topic_id", facet_col_wrap=3,
                hover_data=["conv_id", "speaker", f"{root}_deviation_from_self_play"],
            ),
            width="stretch",
        )
        render_figure_guide("observed_trajectory", f"How to read the {label.lower()} trajectory")


def render_persona(bundle: dict) -> None:
    st.header("Persona and direct observer-rated Big Five")
    st.caption(
        "Big Five values are direct 0–4 ratings of visible conversational presentation, "
        "with separate evidence confidence. They are not BFI-44 inventory scores."
    )
    big_five_tab, interaction_tab = st.tabs([
        "Direct Big Five", "Interaction-style persona",
    ])
    with big_five_tab:
        render_target_progression(
            bundle, "Big Five snapshot progression", "observer_big5_", "big_five"
        )
        render_persona_window(bundle, "observer_big5_", "big_five", "Big Five")
        confidence_columns = [
            column for column in bundle["turns"]
            if column.startswith("observer_big5_") and column.endswith("_confidence")
        ]
        if confidence_columns:
            confidence = bundle["turns"].melt(
                id_vars=[column for column in ("model", "condition", "speaker") if column in bundle["turns"]],
                value_vars=confidence_columns,
                var_name="trait", value_name="confidence",
            ).dropna(subset=["confidence"])
            confidence["trait"] = (
                confidence["trait"].str.removeprefix("observer_big5_")
                .str.removesuffix("_confidence")
            )
            st.subheader("Visible-evidence confidence")
            st.plotly_chart(
                px.box(
                    confidence, x="trait", y="confidence", color="condition",
                    facet_col="model", points=False,
                    title="Confidence separates weak evidence from a genuine midpoint score",
                ),
                width="stretch",
            )
            render_figure_guide("confidence", "How to read evidence confidence")
    with interaction_tab:
        render_target_progression(
            bundle, "Perceived interaction-style progression",
            "perceived_persona_", "persona",
        )
        render_persona_window(
            bundle, "perceived_persona_", "persona", "Interaction style"
        )


def render_transitions(bundle: dict) -> None:
    render_experiment_guide("1G")
    render_target_progression(bundle, "Observable transitions", "transition", "transition")
    scores = bundle.get("scores", pd.DataFrame())
    transition_targets = [
        target for target in (
            "observable_transition", "explicit_synthesis", "explicit_resolution",
            "explicit_closure", "closure_evidence",
        ) if not scores.empty and target in set(scores["target"])
    ]
    if transition_targets:
        st.write("Available transition targets:", transition_targets)


def render_static_figures(bundle: dict) -> None:
    st.header("Exported figure gallery")
    st.info(
        "Each exported figure is a frozen view of the same saved artifacts used above. "
        "Use the experiment reference and target glossary to interpret its estimand; "
        "look for held-out consistency rather than the most visually extreme panel."
    )
    render_all_experiment_guides()
    figures = bundle.get("figures", pd.DataFrame())
    if figures.empty:
        st.info(
            "No figure_index.csv is attached to this analysis run. Use the artifact "
            "exporter; the dashboard will never refit analyses to create it."
        )
        return
    kinds = sorted(figures["kind"].dropna().unique())
    kind = st.selectbox("Figure family", kinds, key="figure_kind")
    subset = figures[figures["kind"].eq(kind)].reset_index(drop=True)
    labels = []
    for _, row in subset.iterrows():
        details = [str(row[column]) for column in ("experiment", "target", "horizon")
                   if column in row and pd.notna(row[column])]
        labels.append(" · ".join(details) if details else Path(row["path"]).stem)
    selected = st.selectbox("Figure", range(len(subset)), format_func=lambda i: labels[i])
    row = subset.iloc[selected]
    path = Path(row["path"])
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        st.error(f"Indexed figure is missing: {path}")
        return
    st.image(str(path), caption=labels[selected], width="stretch")
    st.dataframe(pd.DataFrame([row]), width="stretch", hide_index=True)


def render_quality(bundle: dict, results_root: str) -> None:
    st.header("Replay-quality audit trail")
    validations = bundle["validations"]
    if validations.empty:
        st.warning("No original-versus-replayed validation rows are present.")
    else:
        columns = st.columns(3)
        columns[0].plotly_chart(
            px.histogram(validations, x="cosine_similarity", color="layer", nbins=50,
                         title="Cosine-similarity distribution"),
            width="stretch",
        )
        columns[1].plotly_chart(
            px.histogram(validations, x="relative_error", color="layer", nbins=50,
                         title="Relative-error distribution"),
            width="stretch",
        )
        columns[2].plotly_chart(
            px.histogram(validations, x="norm_ratio", color="layer", nbins=50,
                         title="Norm-ratio distribution"),
            width="stretch",
        )
        st.plotly_chart(
            px.scatter(
                validations, x="response_token_count", y="relative_error",
                color="layer", symbol="model", hover_data=["conv_id", "turn", "role"],
                title="Replay error by response length, layer, model, and conversation",
            ),
            width="stretch",
        )
        by_conversation = (
            validations.groupby(["conv_id", "model", "layer"], as_index=False)
            [["cosine_similarity", "relative_error", "norm_ratio"]].median()
        )
        st.dataframe(by_conversation, width="stretch", hide_index=True)

    st.subheader("Token-boundary mismatches and warnings")
    warnings = bundle["warnings"]
    if warnings.empty:
        st.success("No derived token-alignment warnings.")
    else:
        st.dataframe(warnings, width="stretch", hide_index=True)

    eligibility = pd.DataFrame(bundle["manifest"].get("eligibility", []))
    failed = (
        eligibility[~eligibility["eligible"]]
        if not eligibility.empty and "eligible" in eligibility else pd.DataFrame()
    )
    st.subheader("Failed or unavailable reconstruction examples")
    if failed.empty:
        st.success("No ineligible reconstruction rows recorded.")
    else:
        st.dataframe(
            failed[["conv_id", "turn", "speaker", "model", "reason"]].head(200),
            width="stretch", hide_index=True,
        )

    st.subheader("Original and replayed full-response sensitivity")
    render_matched_sensitivity(bundle)


def main() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            background-color: #ffffff;
            color: #172033;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.header("Dashboard navigation")
        page = st.radio(
            "Page",
            [
                "Overview", "Conversation explorer", "Activation atlas",
                "Probe dashboard", "Partner reactions", "Basin movement", "Persona and Big Five",
                "Observable transitions", "Replay quality", "Static figures",
                "Track 2 · Overview", "Track 2 · Geometry",
                "Track 2 · Accommodation", "Track 2 · Emotion",
            ],
        )
        is_track2 = page.startswith("Track 2 ·")
        st.divider()
        st.header("Artifact selection")
        if is_track2:
            track2_results_dir = st.text_input("Track 2 results directory", "results/track2")
        else:
            data_dir = st.text_input("Track 1 data directory", "data/track1")
            results_root = st.text_input("Track 1 results root", "results/track1")
            replay_runs = discover_replay_runs(data_dir)
            if not replay_runs:
                st.error("No replay manifests found.")
                st.stop()
            replay_dir = st.selectbox(
                "Replay run", replay_runs, key="replay_run_complete_first_v2",
                format_func=lambda path: Path(path).name,
            )
            result_runs = discover_result_runs(results_root)
            preferred = next(
                (index for index, path in enumerate(result_runs)
                 if Path(path).name == Path(replay_dir).name),
                0,
            )
            results_dir = st.selectbox(
                "Analysis run", result_runs or [""], key="analysis_run_v2",
                index=preferred if result_runs else 0,
                format_func=lambda path: Path(path).name if path else "not available",
            )
            geometry_path = st.text_input(
                "Turn geometry CSV", os.path.join(results_root, "geometry", "turn_geometry.csv")
            )
            annotations_path = st.text_input(
                "Optional annotations CSV", os.path.join(data_dir, "annotations.csv")
            )

    if is_track2:
        st.title("Trajectory research dashboard")
        st.caption("Read-only dashboard. It only displays saved Track 2 artifacts.")
        render_track2_page(page, track2_results_dir)
        return

    st.title("Track 1 · Replayed activation timing")
    st.caption(
        "Read-only dashboard. It never replays models, mutates transcripts, or fits probes."
    )

    try:
        bundle = load_bundle(
            data_dir, replay_dir, results_dir, geometry_path, annotations_path
        )
    except Exception as error:
        st.exception(error)
        st.stop()

    preliminary = bundle.get("preliminary_manifest")
    if preliminary:
        probe_ready = preliminary.get("cross_topic_probe_ready", False)
        status = "cross-topic probe ready" if probe_ready else "descriptive views only"
        cutoff = (
            f", first {preliminary['requested_batches']} inferred batches"
            if preliminary.get("requested_batches") is not None else ""
        )
        st.warning(
            "Preliminary partial annotations — "
            f"{preliminary.get('selected_turns', 0)} turns, "
            f"{preliminary.get('selected_conversations', 0)} conversations, "
            f"{preliminary.get('selected_topics', 0)} topics{cutoff}; {status}."
        )
        for warning in preliminary.get("warnings", []):
            st.caption(warning)

    if page == "Overview":
        render_overview(bundle, replay_dir, results_dir)
    elif page == "Conversation explorer":
        render_conversation(bundle)
    elif page == "Activation atlas":
        render_activation_atlas(bundle, replay_dir)
    elif page == "Probe dashboard":
        render_probes(bundle, results_root)
    elif page == "Partner reactions":
        render_partner(bundle)
    elif page == "Basin movement":
        render_basin(bundle)
    elif page == "Persona and Big Five":
        render_persona(bundle)
    elif page == "Observable transitions":
        render_transitions(bundle)
    elif page == "Replay quality":
        render_quality(bundle, results_root)
    else:
        render_static_figures(bundle)


if __name__ == "__main__":
    main()
