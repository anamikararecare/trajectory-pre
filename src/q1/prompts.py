"""Q1-specific stance-locked conversation prompts."""

from __future__ import annotations

from src.common.debate_prompts import DebateTopic
from src.common.llm_client import ChatMessage

Q1_PROMPT_PROTOCOL = "q1_stance_locked_response_v1"


def stance_statement(topic: DebateTopic, role: str) -> str:
    if role == "supporter":
        return topic.support_item or f"I support {topic.display_name}."
    if role == "opposer":
        return topic.oppose_item or f"I do not support {topic.display_name}."
    raise ValueError(f"Q1 requires supporter/opposer roles, received {role!r}")


def q1_system_prompt(
    topic: DebateTopic,
    role: str,
    min_words: int = 60,
    max_words: int = 100,
) -> str:
    required = stance_statement(topic, role)
    reason = topic.pro_seed if role == "supporter" else topic.con_seed
    return (
        "You are one participant in a structured discussion. The following is "
        f"your fixed global position: “{required}” You must maintain that overall "
        "conclusion throughout the conversation. A central reason available to "
        f"you is: {reason} You may acknowledge a counterargument, but explicitly "
        "explain why it does not change your fixed conclusion. Never endorse both "
        "global positions in the same response.\n\n"
        "Each message from the other participant is a completed turn, even if it "
        "ends abruptly. Respond to its central claim in your own words. Never "
        "finish, continue, or complete their sentence or paragraph. Do not imitate "
        "their unfinished wording.\n\n"
        f"Write exactly one self-contained paragraph of {min_words}–{max_words} "
        "words. Use no headings or lists. End with a complete sentence. Respond "
        "only with the discussion paragraph."
    )


def q1_opening_prompt(topic: DebateTopic, role: str, min_words: int, max_words: int) -> str:
    return (
        f"{q1_system_prompt(topic, role, min_words, max_words)}\n\n"
        f"Discussion topic: {topic.display_name}.\n"
        f"One position is: {topic.pro_seed}\n"
        f"The opposing position is: {topic.con_seed}\n\n"
        "Give your opening response from your fixed position."
    )


def q1_partner_message(text: str) -> str:
    return (
        "The other participant's completed turn is quoted below.\n\n"
        f"<partner_turn>\n{text}\n</partner_turn>\n\n"
        "Respond to that turn from your fixed global position. Do not continue "
        "or complete its wording."
    )


def initial_history(
    topic: DebateTopic, role: str, min_words: int, max_words: int
) -> list[ChatMessage]:
    # A single user message is portable across Qwen, Gemma, Llama, and Mistral
    # chat templates, including templates that reject a separate system role.
    return [
        ChatMessage(
            "user",
            q1_opening_prompt(topic, role, min_words, max_words),
        )
    ]

