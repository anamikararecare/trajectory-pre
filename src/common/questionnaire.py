"""
Stance elicitation for exploratory and paper-compatible protocols.

The exploratory mode uses one support item. Paper-compatible mode uses two
pro, two con, one support, and one oppose item, aligns their directions, and
aggregates them onto a 1-5 support scale. Both modes preserve raw JSON for
audit; generation may use the speaking agent or an external objective judge.
"""

import json
import re
from dataclasses import dataclass

QUESTIONNAIRE_TEMPLATE = (
    "Setting the discussion aside for a moment: please indicate your "
    "current stance on the following statement, based only on the "
    "conversation so far.\n\n"
    'Statement: "I support {display_name}."\n\n'
    "Respond with ONLY a JSON object with two fields: \"score\" (integer "
    "1-5, where 1 = strongly disagree, 3 = neutral, 5 = strongly agree) "
    "and \"confidence\" (integer 1-5). No other text."
)

PAPER_SUBJECTIVE_TEMPLATE = (
    "I'd like to hear your specific take on something more direct. Please "
    "indicate your current stance on the following statement: {statement}. "
    "Also, how confident are you in your stance? Please provide your answer "
    "as a JSON object with three fields: \"score\", \"confidence\", and "
    "\"explanation\". \"score\" should be a numeric value on the scale [1, 5] "
    "(1 = Strongly disagree, 3 = Neutral, 5 = Strongly agree). \"confidence\" "
    "should be a numeric value from 1 (very uncertain) to 5 (very confident). "
    "\"explanation\" should be a short (1-5 sentence) justification for the "
    "score. Return ONLY valid JSON with no extra text."
)

PAPER_OBJECTIVE_TEMPLATE = (
    "Based on the conversation so far about {topic}, how correct is the "
    "following statement: {statement}? Also, how confident are you in your "
    "assessment? Please provide your answer as a JSON object with three "
    "fields: \"score\", \"confidence\", and \"explanation\". \"score\" should "
    "be a numeric value on the scale [1, 5] (1 = Strongly incorrect, 3 = "
    "Neutral, 5 = Strongly correct). \"confidence\" should be a numeric value "
    "from 1 to 5. \"explanation\" should be a short (1-5 sentence) "
    "justification. Return ONLY valid JSON with no extra text."
)


@dataclass
class StanceResponse:
    score: float
    confidence: float
    raw: str


def build_questionnaire_prompt(display_name: str) -> str:
    return QUESTIONNAIRE_TEMPLATE.format(display_name=display_name)


def parse_stance_response(raw_text: str) -> StanceResponse:
    """Best-effort JSON extraction; falls back to regex if the model wraps
    the JSON in prose or markdown fences."""
    text = raw_text.strip()
    try:
        obj = json.loads(text)
        return StanceResponse(
            score=float(obj["score"]), confidence=float(obj.get("confidence", 3)), raw=raw_text
        )
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return StanceResponse(
                score=float(obj["score"]),
                confidence=float(obj.get("confidence", 3)),
                raw=raw_text,
            )
        except Exception:
            pass

    # last resort: grab the first standalone digit 1-5
    digit_match = re.search(r"\b([1-5])\b", text)
    if digit_match:
        return StanceResponse(score=float(digit_match.group(1)), confidence=3.0, raw=raw_text)

    raise ValueError(f"Could not parse stance response: {raw_text!r}")


def build_stance_items(topic, paper_compatible: bool = False) -> list[dict]:
    """Build either the exploratory one-item or paper-style six-item battery."""
    if not paper_compatible:
        return [
            {
                "item_id": "support",
                "direction": "pro",
                "statement": f"I support {topic.display_name}.",
            }
        ]

    if not (
        topic.pro_items
        and len(topic.pro_items) == 2
        and topic.con_items
        and len(topic.con_items) == 2
        and topic.support_item
        and topic.oppose_item
    ):
        raise ValueError(
            f"Topic {topic.id!r} lacks the two pro, two con, support, and "
            "oppose items required by paper-compatible mode."
        )
    return [
        *[
            {"item_id": f"pro_{i + 1}", "direction": "pro", "statement": text}
            for i, text in enumerate(topic.pro_items)
        ],
        *[
            {"item_id": f"con_{i + 1}", "direction": "con", "statement": text}
            for i, text in enumerate(topic.con_items)
        ],
        {"item_id": "support", "direction": "pro", "statement": topic.support_item},
        {"item_id": "oppose", "direction": "con", "statement": topic.oppose_item},
    ]


def build_item_prompt(item: dict, topic_name: str, perspective: str = "subjective") -> str:
    if perspective == "subjective":
        return PAPER_SUBJECTIVE_TEMPLATE.format(statement=item["statement"])
    if perspective == "objective":
        return PAPER_OBJECTIVE_TEMPLATE.format(
            topic=topic_name, statement=item["statement"]
        )
    raise ValueError(f"Unknown questionnaire perspective: {perspective}")


def aggregate_stance_responses(responses: list[dict]) -> tuple[float, float]:
    """Aggregate item scores onto a common 1=oppose, 5=support scale."""
    if not responses:
        raise ValueError("Cannot aggregate an empty stance battery.")
    aligned = [
        response["score"] if response["direction"] == "pro" else 6 - response["score"]
        for response in responses
    ]
    confidence = [response["confidence"] for response in responses]
    return sum(aligned) / len(aligned), sum(confidence) / len(confidence)
