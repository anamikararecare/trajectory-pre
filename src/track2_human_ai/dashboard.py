"""Read-only Track 2 result windows for the shared Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


TRACK2_TABLES = {
    "projections": "turn_level_self_play_projections.csv",
    "basin": "basin_separation_by_topic.csv",
    "pooled_basin": "exploratory_pooled_basin_separation.csv",
    "coverage": "model_topic_coverage.csv",
    "accommodation_turns": "accommodation_by_turn.csv",
    "accommodation_progress": "accommodation_by_progress.csv",
    "accommodation_models": "accommodation_summary_by_model.csv",
    "emotion_turns": "emotion_by_turn.csv",
    "emotion_roles": "emotion_role_by_progress.csv",
    "emotion_models": "emotion_model_change_by_progress.csv",
    "emotion_gap": "emotion_ai_minus_human.csv",
}


@st.cache_data(show_spinner=False)
def load_track2_results(results_dir: str) -> dict[str, pd.DataFrame]:
    """Load every available Track 2 table without recomputing analyses."""
    root = Path(results_dir)
    return {
        name: pd.read_csv(root / filename)
        if (root / filename).is_file()
        else pd.DataFrame()
        for name, filename in TRACK2_TABLES.items()
    }


def _available(frame: pd.DataFrame, filename: str) -> bool:
    if not frame.empty:
        return True
    st.info(f"Result unavailable. Expected artifact: {filename}")
    return False


def _metric(label: str, value, help_text: str | None = None) -> None:
    st.metric(label, value if pd.notna(value) else "—", help=help_text)


def _model_filter(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty or "model" not in frame:
        return frame
    models = sorted(frame["model"].dropna().astype(str).unique())
    selected = st.multiselect("Models", models, default=models, key=key)
    return frame[frame["model"].astype(str).isin(selected)]


def render_track2_overview(tables: dict[str, pd.DataFrame], results_dir: str) -> None:
    st.header("Track 2 · Human–AI results")
    st.caption(
        "Read-only summary of observational human–AI trajectories. Pooled basin "
        "results are exploratory; topic-conditioned estimates are primary."
    )
    projections = tables["projections"]
    basin = tables["basin"]
    coverage = tables["coverage"]
    columns = st.columns(4)
    with columns[0]:
        _metric("Conversations", projections.get("conv_id", pd.Series(dtype=str)).nunique())
    with columns[1]:
        _metric("Models", projections.get("model", pd.Series(dtype=str)).nunique())
    with columns[2]:
        _metric("Topic buckets", projections.get("topic_bucket", pd.Series(dtype=str)).nunique())
    with columns[3]:
        _metric("Embedded turns", len(projections))

    if not basin.empty:
        st.subheader("Topic-conditioned basin separation")
        summary = (
            basin.groupby("model", as_index=False)
            .agg(mean_S_basin=("S_basin", "mean"), topics=("topic_bucket", "nunique"),
                 conversations=("n", "sum"))
            .sort_values("mean_S_basin", ascending=False)
        )
        st.plotly_chart(
            px.bar(summary, x="model", y="mean_S_basin", color="topics",
                   hover_data=["conversations"], title="Mean separation across topic buckets"),
            width="stretch",
        )

    st.subheader("Artifact inventory")
    inventory = pd.DataFrame([
        {
            "artifact": filename,
            "status": "available" if not tables[name].empty else "missing",
            "rows": len(tables[name]),
        }
        for name, filename in TRACK2_TABLES.items()
    ])
    st.dataframe(inventory, width="stretch", hide_index=True)

    figures = sorted(Path(results_dir).glob("*.png"))
    if figures:
        st.subheader("Saved figures")
        for left_index in range(0, len(figures), 2):
            columns = st.columns(2)
            for column, figure in zip(columns, figures[left_index:left_index + 2]):
                column.image(str(figure), caption=figure.stem.replace("_", " ").title(),
                             width="stretch")
    if not coverage.empty:
        with st.expander("Model × topic coverage table"):
            st.dataframe(coverage, width="stretch", hide_index=True)


def render_track2_geometry(tables: dict[str, pd.DataFrame]) -> None:
    st.header("Track 2 · Trajectory geometry")
    st.caption("Human–AI turns projected into the fixed Track 1 self-play PCA basis.")
    turns = tables["projections"]
    basin = tables["basin"]
    if _available(turns, TRACK2_TABLES["projections"]):
        filter_columns = st.columns(2)
        topics = sorted(turns["topic_bucket"].dropna().unique())
        topic = filter_columns[0].selectbox("Topic bucket", ["All topics", *topics])
        subset = turns if topic == "All topics" else turns[turns["topic_bucket"].eq(topic)]
        models = sorted(subset["model"].dropna().unique())
        selected_models = filter_columns[1].multiselect("Models", models, default=models)
        subset = subset[subset["model"].isin(selected_models)]
        if subset.empty:
            st.info("No projected turns match the current filters.")
        else:
            figure = px.scatter(
                subset, x="sp_pc1", y="sp_pc2", color="model", symbol="role",
                hover_data=["conv_id", "turn_idx", "topic_bucket"], opacity=0.65,
                title="Turn-level projections",
            )
            figure.update_traces(marker={"size": 7})
            st.plotly_chart(figure, width="stretch")

    if _available(basin, TRACK2_TABLES["basin"]):
        st.subheader("Basin separation by topic")
        pivot = basin.pivot_table(index="model", columns="topic_bucket", values="S_basin")
        figure = go.Figure(go.Heatmap(
            z=pivot.to_numpy(), x=pivot.columns, y=pivot.index,
            colorscale="Viridis", text=pivot.round(2).to_numpy(), texttemplate="%{text}",
            colorbar={"title": "S basin"},
            hovertemplate="Model: %{y}<br>Topic: %{x}<br>S basin: %{z:.3f}<extra></extra>",
        ))
        figure.update_layout(height=max(430, 28 * len(pivot)), xaxis_title="Topic bucket")
        st.plotly_chart(figure, width="stretch")
        st.dataframe(basin.sort_values(["topic_bucket", "model"]), width="stretch", hide_index=True)

    pooled = tables["pooled_basin"]
    if not pooled.empty:
        with st.expander("Exploratory pooled basin estimates"):
            st.warning("These pooled estimates are exploratory and may mix topic and deployment confounds.")
            st.dataframe(pooled, width="stretch", hide_index=True)


def render_track2_accommodation(tables: dict[str, pd.DataFrame]) -> None:
    st.header("Track 2 · Accommodation")
    st.caption("Change in AI-to-human proximity over the course of each conversation.")
    summary = tables["accommodation_models"]
    progress = tables["accommodation_progress"]
    turns = tables["accommodation_turns"]
    if _available(summary, TRACK2_TABLES["accommodation_models"]):
        left, right = st.columns([2, 1])
        figure = px.bar(
            summary.sort_values("mean"), x="model", y="mean", error_y="std",
            hover_data=["count"], title="Mean accommodation by model",
        )
        figure.add_hline(y=0, line_dash="dash", line_color="#777")
        left.plotly_chart(figure, width="stretch")
        with right:
            _metric("Overall mean", f"{summary['mean'].mean():.3f}")
            _metric("Model estimates", len(summary))
            _metric("Turn observations", int(summary["count"].sum()))
    if _available(progress, TRACK2_TABLES["accommodation_progress"]):
        filtered = _model_filter(progress, "track2_accommodation_models")
        if not filtered.empty:
            figure = px.line(
                filtered, x="progress_bin", y="mean", color="model", markers=True,
                error_y="sem", hover_data=["n_conversations"],
                title="Accommodation over normalized conversation progress",
            )
            figure.add_hline(y=0, line_dash="dash", line_color="#777")
            st.plotly_chart(figure, width="stretch")
    if not turns.empty:
        with st.expander("Conversation-level explorer"):
            conversation = st.selectbox(
                "Conversation", sorted(turns["conv_id"].astype(str).unique()),
                key="track2_accommodation_conversation",
            )
            example = turns[turns["conv_id"].astype(str).eq(conversation)]
            st.plotly_chart(
                px.line(example, x="ai_turn_position", y="accommodation", markers=True,
                        color="model", title=f"Accommodation · {conversation}"),
                width="stretch",
            )
            st.dataframe(example, width="stretch", hide_index=True)


def render_track2_emotion(tables: dict[str, pd.DataFrame]) -> None:
    st.header("Track 2 · Emotion overlay")
    st.caption("Affiliative affect from the shared GoEmotions-derived scoring procedure.")
    roles = tables["emotion_roles"]
    gap = tables["emotion_gap"]
    models = tables["emotion_models"]
    if _available(roles, TRACK2_TABLES["emotion_roles"]):
        role_figure = px.line(
            roles, x="turn_bin", y="mean", color="role", markers=True, error_y="sem",
            hover_data=["count"], title="Human and assistant affiliative affect",
        )
        st.plotly_chart(role_figure, width="stretch")
    if _available(gap, TRACK2_TABLES["emotion_gap"]):
        gap_figure = px.bar(
            gap, x="turn_bin", y="mean", error_y="sem", hover_data=["count"],
            title="Assistant minus human affiliative affect",
        )
        gap_figure.add_hline(y=0, line_dash="dash", line_color="#777")
        st.plotly_chart(gap_figure, width="stretch")
    if _available(models, TRACK2_TABLES["emotion_models"]):
        filtered = _model_filter(models, "track2_emotion_models")
        if not filtered.empty:
            model_figure = px.line(
                filtered, x="turn_bin", y="mean", color="model", markers=True,
                error_y="sem", hover_data=["n_conversations"],
                title="Within-model affect change from conversation start",
            )
            model_figure.add_hline(y=0, line_dash="dash", line_color="#777")
            st.plotly_chart(model_figure, width="stretch")
    turns = tables["emotion_turns"]
    if not turns.empty:
        with st.expander("Raw turn-level emotion scores"):
            st.dataframe(turns, width="stretch", hide_index=True)


def render_track2_page(page: str, results_dir: str) -> None:
    tables = load_track2_results(results_dir)
    if page == "Track 2 · Overview":
        render_track2_overview(tables, results_dir)
    elif page == "Track 2 · Geometry":
        render_track2_geometry(tables)
    elif page == "Track 2 · Accommodation":
        render_track2_accommodation(tables)
    else:
        render_track2_emotion(tables)
