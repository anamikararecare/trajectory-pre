"""Live, read-only progress monitor for sharded Q1 corpus generation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import yaml


def _load_plan(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_ids(root: Path) -> set[str]:
    transcripts = {
        path.name.removeprefix("q1_transcript__").removesuffix(".json")
        for path in (root / "q1_transcripts").glob("q1_transcript__*.json")
    }
    activations = {
        path.name.removeprefix("q1_activations__").removesuffix(".npz")
        for path in (root / "q1_activations").glob("q1_activations__*.npz")
    }
    # A conversation counts only after both atomic outputs exist.
    return transcripts & activations


def _latest_journal(root: Path, shard: int) -> dict | None:
    path = root / f"q1_generation_journal__shard_{shard:02d}.jsonl"
    if not path.exists():
        return None
    latest = None
    with path.open() as handle:
        for line in handle:
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
    return latest


def render(protocol_path: str, run_id: str, num_shards: int) -> str:
    with open(protocol_path) as handle:
        protocol = yaml.safe_load(handle)
    root = Path(protocol["data_root"]) / run_id
    plan_path = root / "q1_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing Q1 plan: {plan_path}")
    plan = _load_plan(plan_path)
    completed = _completed_ids(root)
    lines = [
        f"Q1 progress — {run_id}",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"Overall: {len(completed):3d}/{len(plan):3d} "
        f"({100 * len(completed) / max(1, len(plan)):5.1f}%)",
        "",
    ]
    protocol_models = list(protocol["models"])
    for shard in range(num_shards):
        groups = [
            model
            for index, model in enumerate(protocol_models)
            if index % num_shards == shard
        ]
        expected_ids = {
            row["conv_id"] for row in plan if row["group_model"] in groups
        }
        done = len(expected_ids & completed)
        latest = _latest_journal(root, shard)
        if latest:
            latest_id = str(latest.get("conv_id", "unknown"))
            latest_id = latest_id if len(latest_id) <= 55 else latest_id[:52] + "..."
            activity = f"{latest.get('status', 'unknown')}: {latest_id}"
        else:
            activity = "no journal event yet"
        lines.extend(
            [
                f"GPU {shard} / shard {shard}: {done:3d}/{len(expected_ids):3d} "
                f"({100 * done / max(1, len(expected_ids)):5.1f}%)",
                f"  groups: {', '.join(groups)}",
                f"  latest: {activity}",
                "",
            ]
        )
    lines.append(
        "A conversation counts as complete only when both its transcript and "
        "activation file exist. Ctrl-C stops this monitor only."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Q1 generation")
    parser.add_argument(
        "--protocol", default="configs/q1_available_protocol.yaml"
    )
    parser.add_argument("--run-id", default="q1_minimum_v1")
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    try:
        while True:
            print("\033[2J\033[H" + render(
                args.protocol, args.run_id, args.num_shards
            ), flush=True)
            if args.once:
                return
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\nQ1 monitor stopped.")


if __name__ == "__main__":
    main()
