from types import SimpleNamespace

from src.q1.q1_portable_cli import _portable_chat_template


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages


def test_portable_template_merges_adjacent_same_role_messages():
    client = SimpleNamespace(tokenizer=_Tokenizer())
    messages = [
        SimpleNamespace(role="user", content="Role instruction."),
        SimpleNamespace(role="user", content="Partner turn."),
        SimpleNamespace(role="assistant", content="Response."),
        SimpleNamespace(role="user", content="Next partner turn."),
    ]
    result = _portable_chat_template(client, messages)
    assert [item["role"] for item in result] == [
        "user",
        "assistant",
        "user",
    ]
    assert result[0]["content"] == "Role instruction.\n\nPartner turn."
