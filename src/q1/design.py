"""Deterministic Q1 corpus design and manifest creation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml


def load_protocol(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def conversation_id(
    topic: str,
    condition: str,
    model_a: str,
    model_b: str,
    role_a: str,
    role_b: str,
    seed: int,
) -> str:
    identity = "|".join(
        [topic, condition, model_a, model_b, role_a, role_b, str(seed)]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return (
        f"q1__{safe_name(topic)}__{condition}__{safe_name(model_a)}"
        f"__{safe_name(model_b)}__{role_a[:3]}_{role_b[:3]}__s{seed}__{digest}"
    )


def build_minimum_plan(protocol: dict) -> pd.DataFrame:
    topics = list(protocol["topic_ids"])
    models = list(protocol["models"])
    anchor = protocol["anchor_model"]
    role_orders = [tuple(order) for order in protocol["role_orders"]]
    seeds = [int(seed) for seed in protocol["seeds"]]
    if anchor not in models:
        raise ValueError("anchor_model must be present in models")
    rows = []
    for model in models:
        for topic in topics:
            for role_a, role_b in role_orders:
                for seed in seeds:
                    rows.append(
                        {
                            "group_model": model,
                            "condition": "self_play",
                            "topic_id": topic,
                            "model_a": model,
                            "model_b": model,
                            "role_a": role_a,
                            "role_b": role_b,
                            "seed": seed,
                        }
                    )
        if model == anchor:
            continue
        for topic in topics:
            for role_a, role_b in role_orders:
                for seed in seeds:
                    rows.append(
                        {
                            "group_model": model,
                            "condition": "mixed_play",
                            "topic_id": topic,
                            "model_a": model,
                            "model_b": anchor,
                            "role_a": role_a,
                            "role_b": role_b,
                            "seed": seed,
                        }
                    )
    plan = pd.DataFrame(rows)
    plan["conv_id"] = [
        conversation_id(
            row.topic_id,
            row.condition,
            row.model_a,
            row.model_b,
            row.role_a,
            row.role_b,
            int(row.seed),
        )
        for row in plan.itertuples(index=False)
    ]
    group_order = {model: index for index, model in enumerate(models)}
    plan["group_index"] = plan["group_model"].map(group_order)
    plan["task_index"] = range(len(plan))
    return plan.sort_values(
        ["group_index", "condition", "topic_id", "role_a", "seed"]
    ).reset_index(drop=True)


def write_plan(protocol_path: str, output_path: str) -> pd.DataFrame:
    plan = build_minimum_plan(load_protocol(protocol_path))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(destination, index=False)
    return plan

