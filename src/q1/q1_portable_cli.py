"""Final Q1 generator with portable alternating-role chat histories."""

from __future__ import annotations

from src.common.llm_client import LocalHFClient
from src.q1 import q1_mechanical_gate_cli  # noqa: F401 - installs final gates


def _portable_chat_template(self, messages):
    normalized = []
    for message in messages:
        item = {"role": message.role, "content": message.content}
        if normalized and normalized[-1]["role"] == item["role"]:
            # Gemma rejects adjacent same-role messages. Combining them is
            # semantically lossless here: both are prompt context supplied
            # before the model's next response.
            normalized[-1]["content"] += "\n\n" + item["content"]
        else:
            normalized.append(item)
    return self.tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
    )


LocalHFClient._apply_chat_template = _portable_chat_template


if __name__ == "__main__":
    from src.q1.q1_cli import main

    main()
