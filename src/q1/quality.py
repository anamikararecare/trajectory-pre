"""Online response-quality gates for Q1 generation."""

from __future__ import annotations

import json
import re

from src.common.debate_prompts import DebateTopic
from src.common.llm_client import ChatMessage
from src.q1.prompts import stance_statement


TERMINAL_PATTERN = re.compile(r"""[.!?][\"'”’)]*$""")


def parse_quality_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        raise ValueError("Quality judge did not return a JSON object")
    value = json.loads(match.group(0))
    required = (
        "role_consistent",
        "mixed_global_stance",
        "continues_partner",
        "self_contained",
    )
    for field in required:
        if not isinstance(value.get(field), bool):
            raise ValueError(f"Quality field {field} must be boolean")
    return value


def judge_quality(
    judge,
    topic: DebateTopic,
    role: str,
    partner_text: str | None,
    response: str,
) -> dict:
    prompt = (
        "Evaluate only the candidate discussion response. Return one JSON object "
        "with boolean fields role_consistent, mixed_global_stance, "
        "continues_partner, self_contained, plus a short string reason.\n\n"
        f"Required fixed position: {stance_statement(topic, role)}\n"
        f"Previous partner turn: {partner_text or '[opening turn]'}\n"
        f"Candidate response: {response}\n\n"
        "role_consistent means the overall conclusion matches the required fixed "
        "position. mixed_global_stance means it endorses both overall conclusions. "
        "continues_partner means it appears to finish or continue the partner's "
        "wording rather than answer it. self_contained means it is a complete "
        "response ending a thought."
    )
    raw = judge.chat([ChatMessage("user", prompt)], max_tokens=180)
    return {**parse_quality_json(raw), "raw": raw}


def basic_quality(
    text: str,
    response_tokens: int,
    max_response_tokens: int,
    min_words: int,
    max_words: int,
) -> dict:
    words = len(text.split())
    return {
        "response_tokens": response_tokens,
        "word_count": words,
        "hit_token_cap": response_tokens >= max_response_tokens - 1,
        "ends_with_terminal_punctuation": bool(
            TERMINAL_PATTERN.search(text.rstrip())
        ),
        # A small tolerance avoids needless retries for tokenizer/word-count
        # edge cases while still rejecting the old 280-word behavior.
        "within_word_tolerance": max(1, min_words - 10)
        <= words
        <= max_words + 15,
    }

