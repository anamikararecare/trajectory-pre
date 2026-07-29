"""
Pulls a human-AI conversation corpus from Hugging Face and filters it down
to conversations comparable to the original paper's setup: multi-turn,
single consistent AI model, and touching on an opinion/controversial topic.

Supported datasets:
  - "lmsys-chat-1m": lmsys/lmsys-chat-1m (gated, accept terms on HF first)
  - "wildchat":       allenai/WildChat-1M (gated, accept terms on HF first)

Output is a flat JSONL, one line per conversation:
{
  "conv_id": ...,
  "model": "gpt-4" | "vicuna-13b" | ...,
  "topic_keywords_hit": [...],
  "turns": [{"role": "human"|"assistant", "text": ...}, ...]
}
"""

from __future__ import annotations

import json
import os

import yaml
from tqdm import tqdm


def _load_keywords(topics_path: str) -> dict[str, str]:
    with open(topics_path) as topics_file:
        cfg = yaml.safe_load(topics_file)
    keyword_topics = {
        keyword.lower(): topic["id"]
        for topic in cfg["topics"]
        for keyword in topic["keywords"]
    }
    keyword_topics.update(
        {keyword.lower(): "general" for keyword in cfg.get("general_controversy_keywords", [])}
    )
    return keyword_topics


def _matches_topic(
    turns: list[dict], keyword_topics: dict[str, str]
) -> tuple[list[str], list[str]]:
    text = " ".join(turn["text"] for turn in turns).lower()
    hits = [keyword for keyword in keyword_topics if keyword in text]
    buckets = sorted({keyword_topics[hit] for hit in hits if keyword_topics[hit] != "general"})
    if not buckets and hits:
        buckets = ["general"]
    return hits, buckets


def fetch_and_filter(
    dataset_name: str,
    out_path: str,
    topics_path: str = "configs/topics.yaml",
    min_turns: int = 6,
    max_conversations: int | None = 2000,
    streaming: bool = True,
) -> str:
    """Streams the dataset (avoids downloading the full multi-GB file up
    front), applies the filters, and writes matching conversations to
    `out_path` as JSONL. Returns out_path.
    """
    from datasets import load_dataset

    hf_name = {
        "lmsys-chat-1m": "lmsys/lmsys-chat-1m",
        "wildchat": "allenai/WildChat-1M",
    }[dataset_name]

    keyword_topics = _load_keywords(topics_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ds = load_dataset(hf_name, split="train", streaming=streaming, token=os.environ.get("HF_TOKEN"))

    n_written = 0
    with open(out_path, "w") as out_f:
        for row in tqdm(ds, desc=f"scanning {hf_name}"):
            conv = _normalize_row(dataset_name, row)
            if conv is None:
                continue
            if len(conv["turns"]) < min_turns:
                continue
            hits, buckets = _matches_topic(conv["turns"], keyword_topics)
            if not hits:
                continue
            conv["topic_keywords_hit"] = hits
            conv["topic_buckets"] = buckets
            conv["topic_bucket"] = buckets[0] if len(buckets) == 1 else "multi_topic"
            out_f.write(json.dumps(conv) + "\n")
            n_written += 1
            if max_conversations and n_written >= max_conversations:
                break

    print(f"Wrote {n_written} filtered conversations to {out_path}")
    return out_path


def _normalize_row(dataset_name: str, row: dict) -> dict | None:
    """Both datasets store an OpenAI-style `conversation` list; field names
    differ slightly between them and across dataset versions, so this is
    intentionally defensive. Returns None if the row can't be parsed.
    """
    try:
        if dataset_name == "lmsys-chat-1m":
            conv_id = row.get("conversation_id") or row.get("model") + str(hash(str(row["conversation"])))
            model = row.get("model", "unknown")
            raw_turns = row["conversation"]
        elif dataset_name == "wildchat":
            conv_id = row.get("conversation_hash") or str(hash(str(row["conversation"])))
            model = row.get("model", "unknown")
            raw_turns = row["conversation"]
        else:
            return None

        turns = []
        for t in raw_turns:
            raw_role = t.get("role")
            if raw_role not in {"user", "assistant"}:
                continue
            role = "human" if raw_role == "user" else "assistant"
            content = t.get("content", "")
            if isinstance(content, list):  # some rows store multimodal content blocks
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if content and content.strip():
                turns.append({"role": role, "text": content.strip()})

        if not turns:
            return None

        return {"conv_id": str(conv_id), "model": model, "turns": turns}
    except Exception:
        return None
