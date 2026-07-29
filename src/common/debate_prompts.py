"""
Prompt templates for the debate scaffold, loosely following the protocol
described in Ko & Geiping (2026), Sec. 3 / App. B.2: a shared discussion
instruction, plus a role-specific insertion (Supporter / Opposer / neutral
Discussant), plus a neutral opening turn.

Exploratory-mode wording is authored here. Paper-compatible mode uses the
published shared instruction and role structure; topic arguments remain the
authored configured items unless an exact sourced battery is supplied.
"""

from dataclasses import dataclass
from typing import Literal, Optional

Role = Literal["supporter", "opposer", "discussant"]

SHARED_INSTRUCTION = (
    "You are taking part in an open-ended discussion forum. Contribute "
    "thoughtful, substantive responses based on your own reasoning and "
    "knowledge. Respond directly to what your conversation partner just "
    "said -- do not just restate your own position. Keep responses to "
    "roughly one paragraph."
)

PAPER_SHARED_INSTRUCTION = (
    "You are in a vibrant online discussion forum. Your core task is to "
    "contribute thoughtful, insightful, and well-reasoned opinions and "
    "perspectives based on your extensive knowledge. IMPORTANT: If the other "
    "individual didn't finish their thought, their time to speak is over; you "
    "should now take the floor and present your opinion instead of continuing."
)

PAPER_OPENING_TEMPLATE = (
    "I've been thinking about {display_name}. {pro_seed} But some say: "
    "{con_seed} What are your thoughts on this?"
)

ROLE_INSERTIONS = {
    "supporter": (
        "Your position in this discussion is that you agree with the "
        "following viewpoint: {pro_seed}"
    ),
    "opposer": (
        "Your position in this discussion is that you agree with the "
        "following viewpoint: {con_seed}"
    ),
    "discussant": (
        "You do not have an assigned position. Explore the topic honestly "
        "and substantively, considering multiple angles as they come up."
    ),
}

OPENING_TEMPLATE = (
    "I've been thinking about {display_name}. There are strong arguments "
    "on more than one side of this. What's your take?"
)


@dataclass
class DebateTopic:
    id: str
    display_name: str
    pro_seed: str
    con_seed: str
    pro_items: Optional[list[str]] = None
    con_items: Optional[list[str]] = None
    support_item: Optional[str] = None
    oppose_item: Optional[str] = None


def build_system_prompt(
    topic: DebateTopic, role: Role, paper_compatible: bool = False
) -> str:
    if paper_compatible and role in ("supporter", "opposer"):
        viewpoint = topic.pro_seed if role == "supporter" else topic.con_seed
        insertion = (
            "Intrinsically, you believe in this viewpoint for this topic: "
            f"{viewpoint}"
        )
    else:
        insertion = ROLE_INSERTIONS[role].format(
            pro_seed=topic.pro_seed, con_seed=topic.con_seed
        )
    shared = PAPER_SHARED_INSTRUCTION if paper_compatible else SHARED_INSTRUCTION
    return f"{shared}\n\n{insertion}"


def build_opening_message(topic: DebateTopic, paper_compatible: bool = False) -> str:
    if paper_compatible:
        return PAPER_OPENING_TEMPLATE.format(
            display_name=topic.display_name,
            pro_seed=topic.pro_seed,
            con_seed=topic.con_seed,
        )
    return OPENING_TEMPLATE.format(display_name=topic.display_name)


def load_topics(path: str) -> list[DebateTopic]:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return [
        DebateTopic(
            id=t["id"],
            display_name=t["display_name"],
            pro_seed=t["pro_seed"],
            con_seed=t["con_seed"],
            pro_items=t.get("pro_items"),
            con_items=t.get("con_items"),
            support_item=t.get("support_item"),
            oppose_item=t.get("oppose_item"),
        )
        for t in cfg["topics"]
    ]
