import csv
import json

import yaml

from src.q1.q1_monitor import render


def test_monitor_reports_per_shard_completion(tmp_path):
    data_root = tmp_path / "q1_data"
    run_root = data_root / "run"
    (run_root / "q1_transcripts").mkdir(parents=True)
    (run_root / "q1_activations").mkdir()
    protocol = {
        "data_root": str(data_root),
        "models": ["a", "b", "c"],
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    rows = [
        {"conv_id": "one", "group_model": "a"},
        {"conv_id": "two", "group_model": "b"},
        {"conv_id": "three", "group_model": "c"},
    ]
    with (run_root / "q1_plan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (run_root / "q1_transcripts" / "q1_transcript__one.json").write_text("{}")
    (run_root / "q1_activations" / "q1_activations__one.npz").write_bytes(b"x")
    (run_root / "q1_generation_journal__shard_00.jsonl").write_text(
        json.dumps({"status": "complete", "conv_id": "one"}) + "\n"
    )

    output = render(str(protocol_path), "run", 3)
    assert "Overall:   1/  3" in output
    assert "GPU 0 / shard 0:   1/  1" in output
    assert "GPU 1 / shard 1:   0/  1" in output
    assert "complete: one" in output
