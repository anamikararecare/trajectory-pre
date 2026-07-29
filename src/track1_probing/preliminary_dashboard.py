"""Focused dashboard for point-in-time Track 1 annotation snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.track1_probing.dashboard_help import (
    render_all_experiment_guides,
    render_experiment_guide,
    render_figure_guide,
    render_metric_glossary,
    render_target_definition,
)
from src.track1_probing.snapshot_analysis import EXPERIMENT_TARGETS


st.set_page_config(
    page_title="Track 1 · Preliminary Interpretability",
    page_icon="🧭",
    layout="wide",
)

SNAPSHOT_ORDER = [
    "pre_generation", "early_response", "full_response", "final_window"
]
SNAPSHOT_LABELS = {
    "pre_generation": "Pre-generation",
    "early_response": "Early response",
    "full_response": "Full response",
    "final_window": "Final window",
}

SCORE_COLUMNS = [
    "experiment", "target", "relation", "delta_target", "horizon", "layer",
    "snapshot", "n", "metric", "null_score", "shuffled_score",
    "baseline_score", "activation_plus_baseline_score", "incremental_score",
    "time_legal_baseline", "prospectively_predictive_eligible", "cv_group",
]


@st.cache_data(show_spinner=False, ttl=5)
def load_preliminary(results_dir: str, annotations_dir: str, geometry_path: str) -> dict:
    results = Path(results_dir)
    annotations = Path(annotations_dir)
    manifest_path = results / "preliminary_manifest.json"
    if not manifest_path.exists():
        manifest_path = annotations / "preliminary_manifest.json"
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    selection = pd.read_csv(annotations / "selection.csv")
    scores_path = results / "snapshot_probe_scores.csv"
    scores = (
        pd.read_csv(scores_path)
        if scores_path.exists() and scores_path.stat().st_size
        else pd.DataFrame(columns=SCORE_COLUMNS)
    )
    geometry = pd.read_csv(geometry_path)
    geometry = geometry.merge(
        selection[["conv_id", "turn"]].drop_duplicates(),
        on=["conv_id", "turn"],
        how="inner",
        validate="one_to_one",
    )
    geometry["paper_pc1"] = geometry["sp_pc1"]
    geometry["paper_pc2"] = -geometry["sp_pc2"]
    transfer_path = results / "experiment_1i_partner_transfer.csv"
    transfer = pd.read_csv(transfer_path) if transfer_path.exists() and transfer_path.stat().st_size else pd.DataFrame()
    skipped_path = results / "skipped_targets.csv"
    skipped = pd.read_csv(skipped_path) if skipped_path.exists() and skipped_path.stat().st_size else pd.DataFrame()
    return {
        "manifest": manifest,
        "selection": selection,
        "scores": scores,
        "geometry": geometry,
        "transfer": transfer,
        "skipped": skipped,
    }


def score_heatmap(frame: pd.DataFrame, value: str, title: str) -> None:
    pivot = frame.pivot_table(
        index="layer", columns="snapshot", values=value, aggfunc="mean"
    )
    columns = [snapshot for snapshot in SNAPSHOT_ORDER if snapshot in pivot]
    pivot = pivot.reindex(columns=columns).sort_index()
    figure = go.Figure(go.Heatmap(
        z=pivot.to_numpy(),
        x=[SNAPSHOT_LABELS.get(column, column) for column in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="RdBu",
        zmid=0,
        text=np.round(pivot.to_numpy(), 3),
        texttemplate="%{text}",
        colorbar={"title": value.replace("_", " ")},
        hovertemplate="layer=%{y}<br>snapshot=%{x}<br>score=%{z:.4f}<extra></extra>",
    ))
    figure.update_layout(title=title, height=480, xaxis_title="", yaxis_title="Layer")
    st.plotly_chart(figure, width="stretch")


def render_geometry(frame: pd.DataFrame) -> None:
    st.header("Self-play and mixed-play trajectories")
    st.caption(
        "PC signs match paper_geometry.png. PCA is the fixed self-play reference "
        "projection; these panels use only turns in the selected batch prefix."
    )
    render_experiment_guide("1H", EXPERIMENT_TARGETS["1H"])
    if frame.empty:
        st.warning("No selected turns have geometry rows.")
        return

    mean = (
        frame.groupby(["model", "condition", "agent_turn"], as_index=False)
        [["paper_pc1", "paper_pc2"]].mean()
    )
    st.plotly_chart(
        px.line(
            mean,
            x="paper_pc1",
            y="paper_pc2",
            color="model",
            line_dash="condition",
            markers=True,
            hover_data=["agent_turn"],
            title="Mean trajectories: solid self-play, dashed mixed-play",
        ),
        width="stretch",
    )
    render_figure_guide("geometry_trajectory", "How to read the mean-trajectory figure")

    left, right = st.columns(2)
    self_turns = frame[frame["condition"].eq("self_play")]
    endpoints = (
        self_turns.sort_values("turn")
        .groupby(["conv_id", "speaker"], as_index=False)
        .tail(1)
    )
    left.plotly_chart(
        px.scatter(
            endpoints,
            x="paper_pc1",
            y="paper_pc2",
            color="model",
            symbol="speaker",
            hover_data=["conv_id", "turn", "topic_id"],
            title="Available self-play endpoints",
        ),
        width="stretch",
    )
    with left:
        render_figure_guide("geometry_endpoints", "How to read the endpoint figure")

    conversations = sorted(frame["conv_id"].dropna().unique())
    selected_conv = right.selectbox("Conversation trajectory", conversations)
    conversation = frame[frame["conv_id"].eq(selected_conv)].sort_values("turn")
    right.plotly_chart(
        px.line(
            conversation,
            x="paper_pc1",
            y="paper_pc2",
            color="speaker",
            markers=True,
            hover_data=["turn", "agent_turn", "model", "condition", "role"],
            title="Selected conversation, turn by turn",
        ),
        width="stretch",
    )
    with right:
        render_figure_guide("geometry_trajectory", "How to read the selected trajectory")

    st.subheader("1H geometry variables across selected conversations")
    metrics = [
        column for column in (
            "basin_leaning", "partnerward_basin_velocity",
            "semantic_velocity", "semantic_acceleration", "off_axis_distance",
        ) if column in frame
    ]
    metric = st.selectbox("Geometry outcome", metrics)
    render_target_definition(metric)
    available = frame.dropna(subset=[metric]).sort_values(["conv_id", "speaker", "turn"])
    st.plotly_chart(
        px.line(
            available,
            x="agent_turn",
            y=metric,
            color="model",
            line_dash="condition",
            line_group="conv_id",
            facet_col="speaker",
            hover_data=["conv_id", "turn", "topic_id", "role"],
            title=f"Individual-conversation {metric.replace('_', ' ')} trajectories",
        ),
        width="stretch",
    )
    render_figure_guide("observed_trajectory", "How to read the geometry-outcome figure")


def render_experiment(scores: pd.DataFrame, experiment: str) -> None:
    labels = {
        "1B": "Where the current response state is represented",
        "1C": "Upcoming-response prediction",
        "1D": "Future-self prediction",
        "1E": "Immediate partner reaction",
        "1F": "Apparent objective and expected reaction",
        "1G": "Observable transitions",
        "1H": "Basin and semantic movement decoding",
        "1I": "Partner-induced transfer decoding",
    }
    st.header(f"{experiment} · {labels[experiment]}")
    render_experiment_guide(experiment, EXPERIMENT_TARGETS[experiment])
    render_metric_glossary()
    if scores.empty or "experiment" not in scores:
        st.info(
            "Probe scores are still being computed. This page will become available "
            "after snapshot_probe_scores.csv is written; the trajectory and existing "
            "transfer views can be used now."
        )
        return
    subset = scores[scores["experiment"].eq(experiment)].copy()
    if subset.empty:
        st.warning("No score rows are available for this experiment.")
        return

    controls = st.columns(4)
    target = controls[0].selectbox(
        "Target", sorted(subset["target"].dropna().unique()), key=f"{experiment}_target"
    )
    render_target_definition(target)
    subset = subset[subset["target"].eq(target)]
    horizons = sorted(subset["horizon"].dropna().unique())
    horizon = controls[1].selectbox("Horizon", horizons, key=f"{experiment}_horizon")
    subset = subset[subset["horizon"].eq(horizon)]
    delta_values = sorted(subset["delta_target"].dropna().unique())
    delta = controls[2].selectbox(
        "Predict change", delta_values, key=f"{experiment}_delta"
    )
    subset = subset[subset["delta_target"].eq(delta)]
    metric = subset["metric"].iloc[0]
    controls[3].metric("Evaluation", metric)

    usable = subset[np.isfinite(pd.to_numeric(subset["incremental_score"], errors="coerce"))]
    if usable.empty:
        st.warning(
            "This target has insufficient variation or valid held-out predictions in "
            "the selected batch prefix."
        )
        st.dataframe(subset, width="stretch", hide_index=True)
        return

    left, right = st.columns(2)
    with left:
        score_heatmap(
            usable,
            "incremental_score",
            "Incremental activation lift over text/state baseline",
        )
        render_figure_guide("incremental_heatmap", "How to read incremental activation lift")
    with right:
        comparison = usable.melt(
            id_vars=["layer", "snapshot"],
            value_vars=["baseline_score", "activation_plus_baseline_score"],
            var_name="input",
            value_name="score",
        )
        st.plotly_chart(
            px.line(
                comparison,
                x="layer",
                y="score",
                color="input",
                facet_col="snapshot",
                markers=True,
                category_orders={"snapshot": SNAPSHOT_ORDER},
                title="Baseline versus activation-augmented prediction",
            ),
            width="stretch",
        )
        render_figure_guide("baseline_comparison", "How to read baseline versus activation")
    best = usable.sort_values("incremental_score", ascending=False).head(12)
    st.subheader("Most promising layer/snapshot combinations")
    st.dataframe(
        best[[
            "target", "horizon", "delta_target", "layer", "snapshot", "n",
            "baseline_score", "activation_plus_baseline_score",
            "incremental_score", "null_score", "shuffled_score", "cv_group",
        ]],
        width="stretch",
        hide_index=True,
    )
    render_figure_guide("top_combinations", "How to interpret the ranked combinations")


def render_transfer(transfer: pd.DataFrame) -> None:
    if transfer.empty:
        return
    st.subheader("1I mixed-play minus same-model self-play")
    behavior = transfer[transfer["kind"].eq("behavior")]
    activation = transfer[transfer["kind"].eq("activation")]
    if not behavior.empty:
        st.plotly_chart(
            px.bar(
                behavior,
                x="target",
                y="mixed_minus_self",
                color="model",
                barmode="group",
                hover_data=["n_self", "n_mixed"],
                title="Behavioral shift",
            ),
            width="stretch",
        )
        render_figure_guide("transfer", "How to read the behavioral-transfer figure")
    if not activation.empty:
        st.plotly_chart(
            px.line(
                activation,
                x="layer",
                y="mixed_minus_self",
                color="snapshot",
                facet_col="model",
                markers=True,
                category_orders={"snapshot": SNAPSHOT_ORDER},
                title="Activation displacement",
            ),
            width="stretch",
        )
        render_figure_guide("transfer", "How to read the activation-transfer figure")


@st.fragment(run_every=5)
def render_live_page(
    results_dir: str,
    annotations_dir: str,
    geometry_path: str,
    page: str,
) -> None:
    try:
        bundle = load_preliminary(results_dir, annotations_dir, geometry_path)
    except Exception as error:
        st.exception(error)
        st.stop()

    manifest = bundle["manifest"]
    batches = manifest.get("requested_batches")
    st.title("Track 1 preliminary interpretability")
    render_all_experiment_guides()
    st.warning(
        f"Exploratory snapshot: {manifest.get('selected_turns', 0)} turns"
        + (f" from the first {batches} inferred batches" if batches else "")
        + ". Probes use leave-one-conversation-out CV, not paper-grade "
        "leave-one-topic-out evaluation."
    )
    if bundle["scores"].empty:
        st.info(
            "Partial-results mode: probe fitting is still running. Trajectories and "
            "already-written summaries are available now."
        )

    if page == "Trajectories":
        render_geometry(bundle["geometry"])
    else:
        render_experiment(bundle["scores"], page)
        if page == "1I":
            render_transfer(bundle["transfer"])


def main() -> None:
    run_id = os.environ.get(
        "TRACK1_PRELIM_RUN_ID", "full_track1_20260721T090331Z"
    )
    with st.sidebar:
        st.header("Preliminary snapshot")
        run_id = st.text_input("Run ID", run_id)
        results_dir = st.text_input(
            "Results", f"results/track1/preliminary/{run_id}"
        )
        annotations_dir = st.text_input(
            "Snapshot annotations", f"data/track1/preliminary_annotations/{run_id}"
        )
        geometry_path = st.text_input(
            "Geometry", "results/track1/geometry/turn_geometry.csv"
        )
        page = st.radio(
            "View",
            ["Trajectories", "1B", "1C", "1D", "1E", "1F", "1G", "1H", "1I"],
        )
    render_live_page(results_dir, annotations_dir, geometry_path, page)


if __name__ == "__main__":
    main()
