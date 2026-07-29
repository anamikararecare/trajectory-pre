import json

from src.track1_probing.offline_annotations import (
    ANNOTATION_SCHEMA,
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
)
from src.track1_probing.preliminary_annotations import (
    read_journal_snapshot,
    snapshot_preliminary_annotations,
)


def _annotation(item_id, conv_id, turn, annotator="gpt_pass_1"):
    row = {
        "item_id": item_id,
        "conv_id": conv_id,
        "turn": turn,
        "annotator_id": annotator,
        "annotation_schema": ANNOTATION_SCHEMA,
    }
    row.update({field: 2 for field in CONTINUOUS_FIELDS})
    row.update({field: sorted(choices)[0] for field, choices in CATEGORICAL_FIELDS.items()})
    row.update({field: 0 for field in BINARY_FIELDS})
    return row


def test_preliminary_snapshot_is_stratified_and_noncanonical(tmp_path):
    data = tmp_path / "data"
    transcripts = data / "transcripts"
    transcripts.mkdir(parents=True)
    journal = tmp_path / "journal.jsonl"
    rows = []
    for topic in ("a", "b", "c"):
        for condition in ("mixed_play", "self_play"):
            conv_id = f"{topic}-{condition}"
            transcript = {
                "conv_id": conv_id,
                "topic_id": topic,
                "condition": condition,
                "turns": [
                    {
                        "turn": turn,
                        "speaker": "a" if turn % 2 == 0 else "b",
                        "role": "supporter" if turn % 2 == 0 else "opposer",
                        "text": "example",
                    }
                    for turn in range(4)
                ],
            }
            (transcripts / f"{conv_id}.json").write_text(json.dumps(transcript))
            rows.extend(
                _annotation(f"{conv_id}::{turn}", conv_id, turn)
                for turn in range(4)
            )
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))

    manifest = snapshot_preliminary_annotations(
        str(data), str(journal), str(tmp_path / "preliminary"), max_turns=12
    )

    assert manifest["selected_turns"] == 12
    assert manifest["selected_topics"] == 3
    assert manifest["selected_conditions"] == 2
    assert manifest["status"] == "preliminary_partial_annotations"
    assert not (data / "annotations.csv").exists()
    assert (tmp_path / "preliminary" / "coverage.csv").is_file()


def test_journal_snapshot_ignores_an_in_progress_trailing_record(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_bytes(b'{"item_id":"complete"}\n{"item_id":"still-writing"')

    assert read_journal_snapshot(journal) == [{"item_id": "complete"}]


def test_preliminary_snapshot_can_select_a_manual_batch_prefix(tmp_path):
    data = tmp_path / "data"
    transcripts = data / "transcripts"
    transcripts.mkdir(parents=True)
    journal = tmp_path / "journal.jsonl"
    rows = []
    for conversation in range(4):
        conv_id = f"c-{conversation}"
        transcript = {
            "conv_id": conv_id,
            "topic_id": f"topic-{conversation % 2}",
            "condition": "self_play" if conversation % 2 else "mixed_play",
            "turns": [
                {
                    "turn": turn,
                    "speaker": "a",
                    "role": "supporter",
                    "text": "example",
                }
                for turn in range(4)
            ],
        }
        (transcripts / f"{conv_id}.json").write_text(json.dumps(transcript))
        rows.extend(
            _annotation(f"{conv_id}::{turn}", conv_id, turn)
            for turn in range(4)
        )
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))

    manifest = snapshot_preliminary_annotations(
        str(data),
        str(journal),
        str(tmp_path / "preliminary"),
        max_batches=2,
        batch_size=4,
    )

    assert manifest["selection_method"] == "journal_prefix_inferred_batches"
    assert manifest["requested_batches"] == 2
    assert manifest["selected_annotation_rows"] == 8
    assert manifest["available_complete_inferred_batches"] == 4
