"""Read-only browser for generated Q1 conversations and corpus progress."""

from __future__ import annotations

import json
import os

from src.q1.corpus import (
    DEFAULT_TURN_RANGE_EDGES,
    corpus_inventory,
    load_q1_transcripts,
    turn_range_labels,
)


def transcript_markdown(transcript: dict) -> str:
    """Render a generated transcript without altering its response text."""
    lines = [
        f"# {transcript['conv_id']}",
        "",
        f"- Topic: {transcript.get('topic_id', 'unknown')}",
        f"- Condition: {transcript.get('condition', 'unknown')}",
        f"- Agent A: {transcript.get('agent_a_model', 'unknown')}",
        f"- Agent B: {transcript.get('agent_b_model', 'unknown')}",
        f"- Source: {transcript.get('_source_path', 'unknown')}",
        "",
    ]
    for turn in transcript.get("turns", []):
        speaker = str(turn.get("speaker", "?")).upper()
        model = turn.get(
            "model",
            transcript.get(f"agent_{turn.get('speaker')}_model", "unknown"),
        )
        lines.extend(
            [
                f"## Turn {turn.get('turn')} · Speaker {speaker} · {model}",
                "",
                f"Role: {turn.get('role', 'unknown')}  ",
                f"Stance: {turn.get('stance_score', 'NA')}  ",
                f"Confidence: {turn.get('stance_confidence', 'NA')}",
                "",
                str(turn.get("text", "")),
                "",
            ]
        )
    return "\n".join(lines)


def _turn_range(turn_index: int, turn_count: int) -> str:
    labels = turn_range_labels(DEFAULT_TURN_RANGE_EDGES)
    percentage = 100.0 * (turn_index + 1) / turn_count
    for upper, label in zip(DEFAULT_TURN_RANGE_EDGES[1:], labels):
        if percentage <= upper:
            return label
    return labels[-1]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Q1 conversations", layout="wide")
    st.title("Q1 generated conversation browser")
    st.caption(
        "Read-only view of generated Q1 JSON. Response text is displayed verbatim."
    )
    run_dir = st.sidebar.text_input(
        "Q1 run directory",
        os.environ.get("Q1_RUN_DIR", "data/q1_data/q1_minimum_v1"),
    )
    transcripts = load_q1_transcripts(run_dir)
    inventory = corpus_inventory(run_dir)
    if not inventory.empty:
        planned = int(inventory["planned"].sum())
        ready = int(inventory["analysis_ready"].sum())
        st.sidebar.metric("Analysis-ready", f"{ready} / {planned}")
        st.sidebar.progress(ready / planned if planned else 0.0)
    if not transcripts:
        st.info(f"No transcripts found under {run_dir}/q1_transcripts.")
        return

    topics = sorted({str(item["topic_id"]) for item in transcripts})
    conditions = sorted({str(item["condition"]) for item in transcripts})
    models = sorted(
        {
            str(item[key])
            for item in transcripts
            for key in ("agent_a_model", "agent_b_model")
        }
    )
    topic_filter = st.sidebar.multiselect("Topics", topics, default=topics)
    condition_filter = st.sidebar.multiselect(
        "Conditions", conditions, default=conditions
    )
    model_filter = st.sidebar.multiselect("Models", models, default=models)
    query = st.sidebar.text_input("Search response text").strip().lower()

    filtered = []
    for item in transcripts:
        raw_text = "\n".join(
            str(turn.get("text", "")) for turn in item.get("turns", [])
        ).lower()
        item_models = {
            str(item["agent_a_model"]),
            str(item["agent_b_model"]),
        }
        if (
            str(item["topic_id"]) in topic_filter
            and str(item["condition"]) in condition_filter
            and item_models.intersection(model_filter)
            and (not query or query in raw_text)
        ):
            filtered.append(item)
    st.sidebar.metric("Matching conversations", len(filtered))
    if not filtered:
        st.warning("No conversations match the filters.")
        return

    selected_id = st.selectbox(
        "Conversation", [item["conv_id"] for item in filtered]
    )
    item = next(value for value in filtered if value["conv_id"] == selected_id)
    metadata = st.columns(5)
    metadata[0].metric("Topic", item.get("topic_id", "unknown"))
    metadata[1].metric("Condition", item.get("condition", "unknown"))
    metadata[2].metric("Agent A", item.get("agent_a_model", "unknown"))
    metadata[3].metric("Agent B", item.get("agent_b_model", "unknown"))
    metadata[4].metric("Responses", len(item.get("turns", [])))

    raw = dict(item)
    raw.pop("_source_path", None)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download readable Markdown",
        transcript_markdown(item),
        file_name=f"{selected_id}.md",
        mime="text/markdown",
    )
    downloads[1].download_button(
        "Download original JSON",
        json.dumps(raw, indent=2, ensure_ascii=False),
        file_name=f"{selected_id}.json",
        mime="application/json",
    )

    turns = item.get("turns", [])
    compact = st.toggle("Compact responses", value=False)
    for turn in turns:
        turn_index = int(turn.get("turn", 0))
        model = turn.get(
            "model",
            item.get(f"agent_{turn.get('speaker')}_model", "unknown"),
        )
        title = (
            f"Turn {turn_index} · {_turn_range(turn_index, len(turns))} · "
            f"{model} · {turn.get('role', 'unknown')}"
        )
        with st.expander(title, expanded=not compact):
            st.write(turn.get("text", ""))
            st.caption(
                f"stance={turn.get('stance_score', 'NA')} · "
                f"confidence={turn.get('stance_confidence', 'NA')} · "
                f"agent_turn={turn.get('agent_turn', 'NA')} · "
                f"quality={turn.get('quality_gate_status', 'NA')}"
            )

    with st.expander("Original transcript JSON"):
        st.json(raw)


if __name__ == "__main__":
    main()
