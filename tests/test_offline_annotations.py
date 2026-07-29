import pandas as pd
import pytest

from src.track1_probing.offline_annotations import (
    ANNOTATION_SCHEMA,
    BINARY_FIELDS,
    BIG_FIVE_CONFIDENCE_FIELDS,
    BIG_FIVE_SCORE_FIELDS,
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    annotate_batch,
    complete_journal_rows,
    leakage_safe_batches,
    normalize_annotation,
    parse_json_array,
)
from src.q1.q1_annotations import (
    Q1_PERSONA_FIELDS,
    q1_annotation_prompt,
    resolve_q1_annotation_fields,
)
from src.track1_probing.variables import add_persona_baselines, derive_annotation_variables


def valid_annotation(item_id="c::1"):
    row = {"item_id": item_id}
    row.update({field: 2 for field in CONTINUOUS_FIELDS})
    row.update({field: sorted(choices)[0] for field, choices in CATEGORICAL_FIELDS.items()})
    row.update({field: 0 for field in BINARY_FIELDS})
    return row


def test_annotation_parser_and_schema_validation():
    row = valid_annotation()
    parsed = parse_json_array("json prefix\n[" + __import__("json").dumps(row) + "]")
    normalized = normalize_annotation(parsed[0], "c::1")
    assert set(normalized) == {"item_id", *CONTINUOUS_FIELDS, *CATEGORICAL_FIELDS, *BINARY_FIELDS}

    row["local_agreement"] = 5
    with pytest.raises(ValueError, match="between 0 and 4"):
        normalize_annotation(row, "c::1")


def test_direct_big_five_fields_are_scored_and_confidence_qualified():
    row = normalize_annotation(valid_annotation(), "c::1")
    assert set(BIG_FIVE_SCORE_FIELDS).issubset(row)
    assert set(BIG_FIVE_CONFIDENCE_FIELDS).issubset(row)


def test_annotation_journal_schema_upgrade_rejects_legacy_rows():
    complete = {
        "item_id": "c::1", "conv_id": "c", "turn": 1,
        "annotator_id": "gpt_pass_1", "annotation_schema": ANNOTATION_SCHEMA,
        **{key: value for key, value in valid_annotation().items() if key != "item_id"},
    }
    legacy = {key: value for key, value in complete.items() if key != "annotation_schema"}
    incomplete = {key: value for key, value in complete.items() if key != BIG_FIVE_SCORE_FIELDS[0]}
    assert complete_journal_rows([legacy, incomplete, complete]) == [complete]


def test_annotation_batches_never_include_two_turns_from_one_conversation():
    items = [
        {"conv_id": conv_id, "turn": turn}
        for conv_id in ("a", "b", "c")
        for turn in range(3)
    ]
    batches = leakage_safe_batches(items, batch_size=3)
    assert sum(map(len, batches)) == len(items)
    assert all(len({item["conv_id"] for item in batch}) == len(batch) for batch in batches)


def test_annotation_batches_rotate_across_topic_condition_strata():
    items = [
        {
            "conv_id": f"{topic}-{condition}-{conversation}",
            "turn": turn,
            "topic_id": topic,
            "condition": condition,
        }
        for topic in ("a", "b", "c", "d")
        for condition in ("mixed", "self")
        for conversation in range(2)
        for turn in range(3)
    ]
    batches = leakage_safe_batches(items, batch_size=8)

    assert len({item["topic_id"] for item in batches[0]}) == 4
    assert len({item["condition"] for item in batches[0]}) == 2
    assert len({item["conv_id"] for item in batches[0]}) == len(batches[0])


def test_annotation_batch_retries_only_missing_items(monkeypatch):
    class PartialClient:
        def __init__(self):
            self.requested_ids = []

        def chat(self, messages, max_tokens):
            payload = __import__("json").loads(messages[-1].content.split("\n")[-1])
            ids = [item["item_id"] for item in payload]
            self.requested_ids.append(ids)
            returned = ids[:-1] if len(self.requested_ids) == 1 else ids
            return __import__("json").dumps([valid_annotation(item_id) for item_id in returned])

    monkeypatch.setattr("src.track1_probing.offline_annotations.time.sleep", lambda _: None)
    batch = [
        {
            "item_id": item_id, "topic_id": "topic", "speaker": "speaker",
            "role": "role", "condition": "condition", "prior_context": [],
            "current_response": "response",
        }
        for item_id in ("a::1", "b::1", "c::1")
    ]
    client = PartialClient()
    result = annotate_batch(client, batch, max_retries=2)

    assert [row["item_id"] for row in result] == ["a::1", "b::1", "c::1"]
    assert client.requested_ids == [["a::1", "b::1", "c::1"], ["c::1"]]


def test_annotation_batch_retries_duplicate_item_id(monkeypatch):
    class DuplicateClient:
        calls = 0

        def chat(self, messages, max_tokens):
            self.calls += 1
            rows = [valid_annotation("a::1")]
            if self.calls == 1:
                rows.append(valid_annotation("a::1"))
            return __import__("json").dumps(rows)

    monkeypatch.setattr("src.track1_probing.offline_annotations.time.sleep", lambda _: None)
    item = {
        "item_id": "a::1", "topic_id": "topic", "speaker": "speaker",
        "role": "role", "condition": "condition", "prior_context": [],
        "current_response": "response",
    }
    client = DuplicateClient()

    assert annotate_batch(client, [item], max_retries=2)[0]["item_id"] == "a::1"
    assert client.calls == 2


def test_annotation_batch_splits_unresolved_items_after_retries(monkeypatch):
    class SizeLimitedClient:
        def __init__(self):
            self.requested_ids = []

        def chat(self, messages, max_tokens):
            payload = __import__("json").loads(messages[-1].content.split("\n")[-1])
            ids = [item["item_id"] for item in payload]
            self.requested_ids.append(ids)
            if len(ids) > 2:
                return '[{"item_id": "truncated"}'
            return __import__("json").dumps([valid_annotation(item_id) for item_id in ids])

    monkeypatch.setattr("src.track1_probing.offline_annotations.time.sleep", lambda _: None)
    batch = [
        {
            "item_id": f"c{index}::1", "topic_id": "topic", "speaker": "speaker",
            "role": "role", "condition": "condition", "prior_context": [],
            "current_response": "response",
        }
        for index in range(4)
    ]
    client = SizeLimitedClient()

    result = annotate_batch(client, batch, max_retries=2)

    assert [row["item_id"] for row in result] == [item["item_id"] for item in batch]
    assert client.requested_ids == [
        ["c0::1", "c1::1", "c2::1", "c3::1"],
        ["c0::1", "c1::1", "c2::1", "c3::1"],
        ["c0::1", "c1::1"],
        ["c2::1", "c3::1"],
    ]


def test_annotation_derivations_create_indices_and_priority_transitions():
    frame = pd.DataFrame([
        {
            "conv_id": "c", "turn": 0, "speaker": "a",
            "local_agreement": 0, "remaining_disagreement": 4,
            "affiliation": 0, "adversariality": 4,
            "realized_move": "rebut", "apparent_objective": "challenge",
            "explicit_synthesis": 0, "explicit_resolution": 0, "explicit_closure": 0,
        },
        {
            "conv_id": "c", "turn": 2, "speaker": "a",
            "local_agreement": 4, "remaining_disagreement": 0,
            "affiliation": 4, "adversariality": 0,
            "realized_move": "synthesize", "apparent_objective": "synthesize",
            "explicit_synthesis": 1, "explicit_resolution": 0, "explicit_closure": 0,
        },
        {
            "conv_id": "c", "turn": 4, "speaker": "a",
            "local_agreement": 4, "remaining_disagreement": 0,
            "affiliation": 4, "adversariality": 0,
            "realized_move": "close", "apparent_objective": "close",
            "explicit_synthesis": 0, "explicit_resolution": 1, "explicit_closure": 1,
        },
    ])

    result = derive_annotation_variables(frame)

    assert result.loc[0, "observed_conflict_index"] == 1
    assert result.loc[1, "observed_alignment_index"] == 1
    assert result.loc[1, "observable_transition"] == "synthesis"
    assert result.loc[2, "observable_transition"] == "closure"
    assert result.loc[2, "closure_evidence"] == pytest.approx(0.8)


def test_big_five_trailing_state_baseline_deviation_and_movement():
    frame = pd.DataFrame([
        {
            "conv_id": "c", "turn": turn, "speaker": "a", "model": "m",
            "condition": "self_play", "observer_big5_openness": score,
        }
        for turn, score in enumerate((1.0, 3.0, 2.0))
    ])
    result = add_persona_baselines(frame)
    assert result["observer_big5_openness_trailing3"].tolist() == [1.0, 2.0, 2.0]
    assert result["observer_big5_openness_self_play_baseline"].eq(5 / 3).all()
    assert result.loc[1, "observer_big5_openness_movement"] == 1.0


def test_q1_reduced_persona_categorical_annotation_schema():
    groups, fields = resolve_q1_annotation_fields(["persona", "categorical"])
    assert groups == ("persona", "categorical")
    assert fields == (*Q1_PERSONA_FIELDS, *CATEGORICAL_FIELDS)
    assert len(fields) == 11

    normalized = normalize_annotation(
        valid_annotation(), "c::1", output_fields=fields
    )
    assert set(normalized) == {"item_id", *fields}
    assert "local_agreement" not in normalized
    assert "observer_big5_openness" not in normalized

    prompt = q1_annotation_prompt(fields)
    assert "perceived_persona_warmth" in prompt
    assert "realized_move" in prompt
    assert "local_agreement" not in prompt
    assert "observer_big5_openness" not in prompt
