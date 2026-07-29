"""Create auditable partial annotation snapshots without changing canonical data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.track1_probing.offline_annotations import (
    ANNOTATION_SCHEMA,
    complete_journal_rows,
    load_annotation_items,
    write_csvs,
)


def _rows_sha256(rows: list[dict]) -> str:
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_journal_snapshot(path: Path) -> list[dict]:
    """Read only newline-complete records while another process may append."""
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1] if b"\n" in data else b""
    rows = []
    for line_number, line in enumerate(complete.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed complete journal row at line {line_number}: {error}"
            ) from error
    return rows


def _turn_band(turn: int) -> str:
    if turn < 4:
        return "opening"
    if turn < 12:
        return "early_middle"
    if turn < 20:
        return "late_middle"
    return "closing"


def stratified_turn_ids(rows: list[dict], item_lookup: dict[str, dict], limit: int) -> list[str]:
    """Select turns round-robin across topic, condition, role, and turn band."""
    passes_by_item = Counter(row["item_id"] for row in rows)
    candidates = sorted(
        passes_by_item,
        key=lambda item_id: (
            -passes_by_item[item_id],
            item_lookup[item_id]["conv_id"],
            item_lookup[item_id]["turn"],
        ),
    )
    strata: dict[tuple[str, str, str, str], deque] = defaultdict(deque)
    for item_id in candidates:
        item = item_lookup[item_id]
        key = (
            str(item.get("topic_id", "")),
            str(item.get("condition", "")),
            str(item.get("role", "")),
            _turn_band(int(item["turn"])),
        )
        strata[key].append(item_id)
    selected = []
    keys = sorted(strata)
    while len(selected) < limit and any(strata.values()):
        for key in keys:
            if strata[key]:
                selected.append(strata[key].popleft())
                if len(selected) == limit:
                    break
    return selected


def snapshot_preliminary_annotations(
    data_dir: str,
    journal_path: str,
    out_dir: str,
    max_turns: int | None = None,
    max_batches: int | None = None,
    batch_size: int = 8,
) -> dict:
    journal = Path(journal_path)
    if not journal.is_file():
        raise FileNotFoundError(f"Annotation journal not found: {journal}")
    all_rows = complete_journal_rows(read_journal_snapshot(journal))
    if not all_rows:
        raise ValueError("The journal contains no complete rows under the current schema.")

    items = load_annotation_items(data_dir)
    item_lookup = {item["item_id"]: item for item in items}
    unknown = sorted({row["item_id"] for row in all_rows} - set(item_lookup))
    if unknown:
        raise ValueError(f"Journal rows do not match the frozen corpus: {unknown[:5]}")

    if max_turns is not None and max_batches is not None:
        raise ValueError("Choose either max_turns or max_batches, not both")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    available_turns = len({row["item_id"] for row in all_rows})
    available_complete_batches = len(all_rows) // batch_size
    if max_batches is not None:
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        if max_batches > available_complete_batches:
            raise ValueError(
                f"Requested {max_batches} batches, but only "
                f"{available_complete_batches} complete inferred batches are available"
            )
        selected_rows = all_rows[: max_batches * batch_size]
        selected_ids = {row["item_id"] for row in selected_rows}
        selection_method = "journal_prefix_inferred_batches"
    else:
        limit = available_turns if max_turns is None else min(max_turns, available_turns)
        if limit < 1:
            raise ValueError("max_turns must be positive")
        selected_ids = set(stratified_turn_ids(all_rows, item_lookup, limit))
        selected_rows = [row for row in all_rows if row["item_id"] in selected_ids]
        selection_method = "all_available" if max_turns is None else "stratified_turn_limit"

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_csv = output / "annotations_raw.csv"
    aggregated_csv = output / "annotations.csv"
    write_csvs(selected_rows, raw_csv, aggregated_csv)

    metadata_rows = []
    for item_id in sorted(selected_ids):
        item = item_lookup[item_id]
        metadata_rows.append({
            "item_id": item_id,
            "conv_id": item["conv_id"],
            "turn": item["turn"],
            "topic_id": item.get("topic_id"),
            "condition": item.get("condition"),
            "role": item.get("role"),
            "speaker": item.get("speaker"),
            "turn_band": _turn_band(int(item["turn"])),
            "passes_available": sum(row["item_id"] == item_id for row in selected_rows),
        })
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output / "selection.csv", index=False)

    coverage_rows = []
    for dimension in ("topic_id", "condition", "role", "turn_band"):
        for value, group in metadata.groupby(dimension, dropna=False):
            coverage_rows.append({
                "dimension": dimension,
                "value": value,
                "turns": len(group),
                "conversations": group["conv_id"].nunique(),
            })
    pd.DataFrame(coverage_rows).to_csv(output / "coverage.csv", index=False)

    topics = metadata["topic_id"].nunique()
    conversations = metadata["conv_id"].nunique()
    conditions = metadata["condition"].nunique()
    roles = metadata["role"].nunique()
    paired_turns = int((metadata["passes_available"] >= 2).sum())
    descriptive_ready = len(metadata) >= 30 and conditions >= 2 and roles >= 2
    probe_ready = len(metadata) >= 60 and topics >= 3 and conversations >= 12
    warnings = []
    if not probe_ready:
        warnings.append(
            "Cross-topic probe estimates are not yet interpretable; require at least "
            "60 turns, 3 topics, and 12 conversations."
        )
    if paired_turns == 0:
        warnings.append("No turns have two passes yet, so annotation reliability is unavailable.")
    if selection_method == "journal_prefix_inferred_batches":
        warnings.append(
            "Batch boundaries are inferred from journal order and the configured batch size; "
            "this prefix may be corpus-order biased."
        )
    manifest = {
        "status": "preliminary_partial_annotations",
        "annotation_schema": ANNOTATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "journal_path": str(journal),
        "complete_rows_sha256": _rows_sha256(all_rows),
        "selected_rows_sha256": _rows_sha256(selected_rows),
        "journal_complete_rows": len(all_rows),
        "selection_method": selection_method,
        "requested_batches": max_batches,
        "configured_batch_size": batch_size,
        "available_complete_inferred_batches": available_complete_batches,
        "available_turns": available_turns,
        "selected_annotation_rows": len(selected_rows),
        "selected_turns": len(metadata),
        "selected_conversations": conversations,
        "selected_topics": topics,
        "selected_conditions": conditions,
        "selected_roles": roles,
        "paired_turns": paired_turns,
        "annotator_rows": dict(Counter(row["annotator_id"] for row in selected_rows)),
        "descriptive_ready": descriptive_ready,
        "cross_topic_probe_ready": probe_ready,
        "warnings": warnings,
        "files": {
            "raw_annotations": str(raw_csv),
            "aggregated_annotations": str(aggregated_csv),
            "selection": str(output / "selection.csv"),
            "coverage": str(output / "coverage.csv"),
        },
    }
    (output / "preliminary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot complete journal rows for preliminary Track 1 analysis"
    )
    parser.add_argument("--data_dir", default="data/track1")
    parser.add_argument("--journal", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_turns", type=int)
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    manifest = snapshot_preliminary_annotations(
        args.data_dir,
        args.journal,
        args.out_dir,
        args.max_turns,
        args.max_batches,
        args.batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
