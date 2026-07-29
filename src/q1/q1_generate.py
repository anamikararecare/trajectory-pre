"""Resumable Q1-only conversation generation."""

from __future__ import annotations

import gc
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.debate_prompts import DebateTopic, load_topics
from src.common.llm_client import ChatMessage, build_client, load_model_registry
from src.common.questionnaire import (
    aggregate_stance_responses,
    build_item_prompt,
    build_stance_items,
    parse_stance_response,
)
from src.q1.design import load_protocol, write_plan
from src.q1.prompts import Q1_PROMPT_PROTOCOL, initial_history, q1_partner_message
from src.q1.quality import basic_quality, judge_quality


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _set_seed(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _model_revision(client) -> str | None:
    config = getattr(getattr(client, "model", None), "config", None)
    return getattr(config, "_commit_hash", None)


def _response_token_count(client, text: str) -> int:
    return len(client.tokenizer(text, add_special_tokens=False)["input_ids"])


def _global_stance(
    client,
    history: list[ChatMessage],
    response: str,
    topic: DebateTopic,
) -> dict:
    item = {
        "item_id": "support",
        "direction": "pro",
        "statement": topic.support_item or f"I support {topic.display_name}.",
    }
    raw = client.chat(
        history
        + [
            ChatMessage("assistant", response),
            ChatMessage(
                "user",
                build_item_prompt(item, topic.display_name, perspective="subjective"),
            ),
        ],
        max_tokens=180,
    )
    parsed = parse_stance_response(raw)
    if not 1 <= parsed.score <= 5 or not 1 <= parsed.confidence <= 5:
        raise ValueError("Stance response must lie on [1, 5]")
    return {
        **item,
        "score": parsed.score,
        "confidence": parsed.confidence,
        "raw": parsed.raw,
    }


def _full_stance_battery(
    client,
    history: list[ChatMessage],
    response: str,
    topic: DebateTopic,
    support_result: dict,
) -> tuple[list[dict], float, float]:
    results = []
    for item in build_stance_items(topic, paper_compatible=True):
        if item["item_id"] == "support":
            results.append(support_result)
            continue
        raw = client.chat(
            history
            + [
                ChatMessage("assistant", response),
                ChatMessage(
                    "user",
                    build_item_prompt(
                        item, topic.display_name, perspective="subjective"
                    ),
                ),
            ],
            max_tokens=180,
        )
        parsed = parse_stance_response(raw)
        results.append(
            {
                **item,
                "score": parsed.score,
                "confidence": parsed.confidence,
                "raw": parsed.raw,
            }
        )
    score, confidence = aggregate_stance_responses(results)
    return results, score, confidence


def _role_consistent(role: str, score: float) -> bool:
    return (role == "supporter" and score > 3) or (
        role == "opposer" and score < 3
    )


def _attempt_history(history: list[ChatMessage], attempt: int) -> list[ChatMessage]:
    if attempt == 1:
        return history
    revised = list(history)
    revised[-1] = ChatMessage(
        revised[-1].role,
        revised[-1].content
        + "\n\nYour previous draft was rejected. Write a new draft that "
        "unambiguously maintains your fixed global position, answers rather than "
        "continues the other participant, stays within the requested length, and "
        "ends cleanly.",
    )
    return revised


def generate_conversation(
    task,
    topic: DebateTopic,
    client_a,
    client_b,
    layers_by_model: dict[str, list[int]],
    protocol: dict,
    quality_judge=None,
) -> tuple[dict, dict[str, np.ndarray]]:
    min_words, max_words = map(int, protocol["requested_word_range"])
    max_tokens = int(protocol["max_response_tokens"])
    max_attempts = int(protocol["max_generation_attempts"])
    battery_turns = {int(value) for value in protocol["stance_battery_agent_turns"]}
    histories = {
        "a": initial_history(topic, task.role_a, min_words, max_words),
        "b": initial_history(topic, task.role_b, min_words, max_words),
    }
    roles = {"a": task.role_a, "b": task.role_b}
    models = {"a": task.model_a, "b": task.model_b}
    clients = {"a": client_a, "b": client_b}
    turns: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    _set_seed(int(task.seed))

    for turn_index in range(int(protocol["turns_per_agent"]) * 2):
        speaker = "a" if turn_index % 2 == 0 else "b"
        partner = "b" if speaker == "a" else "a"
        client = clients[speaker]
        role = roles[speaker]
        model = models[speaker]
        history = histories[speaker]
        partner_text = turns[-1]["text"] if turns else None
        accepted = None
        failures = []
        for attempt in range(1, max_attempts + 1):
            candidate_history = _attempt_history(history, attempt)
            text, activations = client.generate_with_activations(
                candidate_history,
                max_tokens=max_tokens,
                layers=layers_by_model[model],
            )
            basic = basic_quality(
                text,
                _response_token_count(client, text),
                max_tokens,
                min_words,
                max_words,
            )
            try:
                support = _global_stance(client, candidate_history, text, topic)
                stance_ok = _role_consistent(role, support["score"])
            except Exception as error:
                support = {
                    "item_id": "support",
                    "direction": "pro",
                    "statement": topic.support_item,
                    "score": None,
                    "confidence": None,
                    "raw": None,
                    "error": repr(error),
                }
                stance_ok = False
            external = None
            if quality_judge is not None:
                try:
                    external = judge_quality(
                        quality_judge, topic, role, partner_text, text
                    )
                except Exception as error:
                    external = {"error": repr(error)}
            external_ok = (
                True
                if quality_judge is None
                else bool(
                    external
                    and external.get("role_consistent")
                    and not external.get("mixed_global_stance")
                    and not external.get("continues_partner")
                    and external.get("self_contained")
                )
            )
            if (
                stance_ok
                and external_ok
                and not basic["hit_token_cap"]
                and basic["ends_with_terminal_punctuation"]
                and basic["within_word_tolerance"]
            ):
                accepted = {
                    "text": text,
                    "activations": activations,
                    "attempt": attempt,
                    "support": support,
                    "basic": basic,
                    "external": external,
                }
                break
            failures.append(
                {
                    "attempt": attempt,
                    "stance_ok": stance_ok,
                    "basic": basic,
                    "external": external,
                }
            )
        if accepted is None:
            raise RuntimeError(
                f"{task.conv_id} turn {turn_index} failed Q1 quality gate: "
                f"{failures}"
            )

        agent_turn = turn_index // 2 + 1
        battery_results = None
        battery_score = None
        battery_confidence = None
        if agent_turn in battery_turns:
            battery_results, battery_score, battery_confidence = (
                _full_stance_battery(
                    client,
                    history,
                    accepted["text"],
                    topic,
                    accepted["support"],
                )
            )
        for layer, vector in accepted["activations"].items():
            arrays[f"{int(layer)}__{turn_index}"] = vector
        turns.append(
            {
                "turn": turn_index,
                "agent_turn": agent_turn,
                "speaker": speaker,
                "model": model,
                "role": role,
                "text": accepted["text"],
                "stance_score": accepted["support"]["score"],
                "stance_confidence": accepted["support"]["confidence"],
                "stance_responses": [accepted["support"]],
                "stance_battery_score": battery_score,
                "stance_battery_confidence": battery_confidence,
                "stance_battery_responses": battery_results,
                "generation_attempts": accepted["attempt"],
                "quality_gate_status": "passed",
                "quality_gate_mode": (
                    "external_and_report"
                    if quality_judge is not None
                    else "report_only"
                ),
                **accepted["basic"],
                "external_quality": accepted["external"],
            }
        )
        histories[speaker].append(ChatMessage("assistant", accepted["text"]))
        histories[partner].append(
            ChatMessage("user", q1_partner_message(accepted["text"]))
        )

    transcript = {
        "schema": "q1_transcript_v1",
        "prompt_protocol": Q1_PROMPT_PROTOCOL,
        "conv_id": task.conv_id,
        "topic_id": task.topic_id,
        "condition": task.condition,
        "agent_a_model": task.model_a,
        "agent_b_model": task.model_b,
        "agent_a_role": task.role_a,
        "agent_b_role": task.role_b,
        "n_turns_per_agent": int(protocol["turns_per_agent"]),
        "seed": int(task.seed),
        "generation": {
            "max_new_tokens": max_tokens,
            "requested_word_range": [min_words, max_words],
            "max_attempts": max_attempts,
        },
        "model_specs": {
            "agent_a": {
                "registry_key": task.model_a,
                "hf_id": client_a.hf_id,
                "resolved_revision": _model_revision(client_a),
                "layers": layers_by_model[task.model_a],
            },
            "agent_b": {
                "registry_key": task.model_b,
                "hf_id": client_b.hf_id,
                "resolved_revision": _model_revision(client_b),
                "layers": layers_by_model[task.model_b],
            },
        },
        "turns": turns,
    }
    return transcript, arrays


def _write_outputs(
    root: Path, transcript: dict, arrays: dict[str, np.ndarray]
) -> None:
    transcript_dir = root / "q1_transcripts"
    activation_dir = root / "q1_activations"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    conv_id = transcript["conv_id"]
    destination = transcript_dir / f"q1_transcript__{conv_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    os.replace(temporary, destination)
    activation_destination = activation_dir / f"q1_activations__{conv_id}.npz"
    with tempfile.NamedTemporaryFile(
        dir=activation_dir, suffix=".npz", delete=False
    ) as handle:
        temporary_activation = Path(handle.name)
    np.savez_compressed(temporary_activation, **arrays)
    os.replace(temporary_activation, activation_destination)


def _journal(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_generation_shard(
    protocol_path: str,
    run_id: str,
    shard_index: int,
    num_shards: int,
    quality_judge_key: str | None,
    quality_judge_registry: str,
    load_in_4bit: bool = False,
) -> Path:
    protocol = load_protocol(protocol_path)
    root = Path(protocol["data_root"], run_id)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "q1_plan.csv"
    plan = (
        pd.read_csv(plan_path)
        if plan_path.exists()
        else write_plan(protocol_path, str(plan_path))
    )
    registry = load_model_registry(protocol["models_path"])
    topics = {topic.id: topic for topic in load_topics(protocol["topics_path"])}
    selected_groups = [
        model
        for index, model in enumerate(protocol["models"])
        if index % num_shards == shard_index
    ]
    journal_path = root / f"q1_generation_journal__shard_{shard_index:02d}.jsonl"
    quality_judge = (
        build_client(
            quality_judge_key,
            load_model_registry(quality_judge_registry),
        )
        if quality_judge_key and quality_judge_key.lower() != "none"
        else None
    )
    for group_model in selected_groups:
        group = plan[plan["group_model"].eq(group_model)]
        target_client = build_client(
            group_model, registry, load_in_4bit=load_in_4bit
        )
        anchor = protocol["anchor_model"]
        anchor_client = (
            target_client
            if group_model == anchor
            else build_client(anchor, registry, load_in_4bit=load_in_4bit)
        )
        clients = {group_model: target_client, anchor: anchor_client}
        layers = {
            key: list(registry[key].get("default_layers", []))
            for key in clients
        }
        for task in group.itertuples(index=False):
            transcript_path = (
                root / "q1_transcripts" / f"q1_transcript__{task.conv_id}.json"
            )
            if transcript_path.exists():
                _journal(
                    journal_path,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "conv_id": task.conv_id,
                        "status": "already_complete",
                    },
                )
                continue
            try:
                transcript, arrays = generate_conversation(
                    task,
                    topics[task.topic_id],
                    clients[task.model_a],
                    clients[task.model_b],
                    layers,
                    protocol,
                    quality_judge=quality_judge,
                )
                _write_outputs(root, transcript, arrays)
                _journal(
                    journal_path,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "conv_id": task.conv_id,
                        "status": "complete",
                        "turns": len(transcript["turns"]),
                    },
                )
            except Exception as error:
                _journal(
                    journal_path,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "conv_id": task.conv_id,
                        "status": "failed",
                        "error": repr(error),
                    },
                )
                raise
        if anchor_client is not target_client:
            del anchor_client
        del target_client
        gc.collect()
        _empty_cuda_cache()
    return root

