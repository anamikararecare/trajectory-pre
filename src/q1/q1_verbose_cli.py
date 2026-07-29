"""Verbose wrapper around the Q1 CLI for long-running generation."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.common.llm_client import LocalHFClient
from src.q1 import q1_generate
from src.q1.design import load_protocol


_original_local_generate = LocalHFClient.generate_with_activations
_original_conversation_generate = q1_generate.generate_conversation
_original_shard_generate = q1_generate.run_generation_shard


def _verbose_local_generate(self, messages, max_tokens=400, layers=None):
    activation_bearing = bool(
        self.default_layers if layers is None else layers
    )
    if not activation_bearing:
        return _original_local_generate(
            self, messages, max_tokens=max_tokens, layers=layers
        )
    attempt = int(getattr(self, "_q1_response_attempt", 0)) + 1
    self._q1_response_attempt = attempt
    started = time.monotonic()
    print(
        f"    response attempt {attempt}: {self.hf_id} "
        f"(max_new_tokens={max_tokens})",
        flush=True,
    )
    text, activations = _original_local_generate(
        self, messages, max_tokens=max_tokens, layers=layers
    )
    elapsed = time.monotonic() - started
    print(
        f"    generated {len(text.split())} words in {elapsed:.1f}s; "
        "running stance and quality checks",
        flush=True,
    )
    return text, activations


def _verbose_conversation(*args, **kwargs):
    task = args[0]
    protocol = args[5]
    expected_turns = int(protocol["turns_per_agent"]) * 2
    print(
        f"\n[Q1] starting {task.conv_id}\n"
        f"     topic={task.topic_id} condition={task.condition}\n"
        f"     {task.model_a}/{task.role_a} vs "
        f"{task.model_b}/{task.role_b}; responses={expected_turns}",
        flush=True,
    )
    started = time.monotonic()
    try:
        transcript, arrays = _original_conversation_generate(*args, **kwargs)
    except Exception:
        print(
            f"[Q1] failed {task.conv_id} after "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
        raise
    turns = transcript["turns"]
    attempts = sum(int(turn["generation_attempts"]) for turn in turns)
    word_counts = [int(turn["word_count"]) for turn in turns]
    print(
        f"[Q1] completed {task.conv_id} in "
        f"{time.monotonic() - started:.1f}s "
        f"(responses={len(turns)}, attempts={attempts}, "
        f"words={min(word_counts)}–{max(word_counts)})",
        flush=True,
    )
    return transcript, arrays


def _verbose_shard(*args, **kwargs):
    protocol_path = kwargs.get("protocol_path", args[0] if args else None)
    run_id = kwargs.get("run_id", args[1] if len(args) > 1 else None)
    shard_index = int(
        kwargs.get("shard_index", args[2] if len(args) > 2 else 0)
    )
    num_shards = int(
        kwargs.get("num_shards", args[3] if len(args) > 3 else 1)
    )
    protocol = load_protocol(protocol_path)
    root = Path(protocol["data_root"]) / run_id
    plan = pd.read_csv(root / "q1_plan.csv")
    groups = [
        model
        for index, model in enumerate(protocol["models"])
        if index % num_shards == shard_index
    ]
    selected = plan[plan["group_model"].isin(groups)]
    completed = sum(
        (
            root
            / "q1_transcripts"
            / f"q1_transcript__{conv_id}.json"
        ).exists()
        for conv_id in selected["conv_id"]
    )
    print(
        f"[Q1] run={run_id} shard={shard_index + 1}/{num_shards} "
        f"groups={','.join(groups)} conversations={len(selected)} "
        f"already_complete={completed}",
        flush=True,
    )
    return _original_shard_generate(*args, **kwargs)


LocalHFClient.generate_with_activations = _verbose_local_generate
q1_generate.generate_conversation = _verbose_conversation
q1_generate.run_generation_shard = _verbose_shard


if __name__ == "__main__":
    from src.q1.q1_cli import main

    main()
