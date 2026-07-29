"""
Unified client interface so debate generation code doesn't care whether an
agent is an API model or a local hookable HF model.

- ApiClient wraps Anthropic / OpenAI chat completion.
- LocalHFClient wraps a local transformers model, and additionally exposes
  `generate_with_activations`, which returns the generated text *and* the
  mean-pooled residual-stream hidden states at a chosen set of layers, for
  the newly generated tokens. This is the piece Track 1 depends on.

Both clients share a `chat(messages) -> str` method for plain generation
(used identically by Track 1's non-activation partner and by Track 2, if you
ever want to generate synthetic comparisons).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


# ---------------------------------------------------------------------------
# API backends
# ---------------------------------------------------------------------------


class ApiClient:
    def __init__(self, backend: str, api_model: str):
        self.backend = backend
        self.api_model = api_model
        if backend == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif backend == "openai":
            import openai

            self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown API backend: {backend}")

    def chat(self, messages: list[ChatMessage], max_tokens: int = 400) -> str:
        if self.backend == "anthropic":
            system = next((m.content for m in messages if m.role == "system"), None)
            turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
            resp = self._client.messages.create(
                model=self.api_model,
                system=system,
                messages=turns,
                max_tokens=max_tokens,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        elif self.backend == "openai":
            turns = [{"role": m.role, "content": m.content} for m in messages]
            resp = self._client.chat.completions.create(
                model=self.api_model, messages=turns, max_tokens=max_tokens
            )
            return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Local HF backend, with activation extraction
# ---------------------------------------------------------------------------


class LocalHFClient:
    def __init__(
        self,
        hf_id: str,
        default_layers: Optional[list[int]] = None,
        load_in_4bit: bool = False,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.device = device or os.environ.get("TRACK1_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.default_layers = default_layers or []
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.load_in_4bit = load_in_4bit

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, token=hf_token, revision=self.tokenizer_revision
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            token=hf_token,
            revision=revision,
            dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            **quant_kwargs,
        )
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

    def _apply_chat_template(self, messages: list[ChatMessage]) -> str:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )

    def chat(self, messages: list[ChatMessage], max_tokens: int = 400) -> str:
        text, _ = self.generate_with_activations(messages, max_tokens=max_tokens, layers=[])
        return text

    def generate_with_activations(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        layers: Optional[list[int]] = None,
    ) -> tuple[str, dict[int, np.ndarray]]:
        """Generate a reply, then run a single forward pass over
        prompt+reply to extract mean-pooled hidden states (over the
        newly-generated tokens only) at the requested layers.

        Returns (generated_text, {layer_idx: pooled_vector}).
        """
        import torch

        layers = self.default_layers if layers is None else layers
        prompt_text = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full_ids = gen_out[0]
        generated_ids = full_ids[prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        activations: dict[int, np.ndarray] = {}
        if layers:
            with torch.no_grad():
                out = self.model(full_ids.unsqueeze(0), output_hidden_states=True)
            # hidden_states: tuple of (num_layers+1) tensors, (1, seq_len, hidden_dim)
            for layer_idx in layers:
                h = out.hidden_states[layer_idx][0]  # (seq_len, hidden_dim)
                gen_hidden = h[prompt_len:]  # only the newly generated tokens
                if gen_hidden.shape[0] == 0:
                    gen_hidden = h[-1:]
                pooled = gen_hidden.mean(dim=0).float().cpu().numpy()
                activations[layer_idx] = pooled

        return generated_text, activations

    def teacher_force_snapshots(
        self,
        messages: list[ChatMessage],
        response_text: str,
        layers: Optional[list[int]] = None,
        original_max_new_tokens: Optional[int] = None,
    ) -> tuple[dict[tuple[int, str], np.ndarray], dict]:
        """Teacher-force recorded text; this method never calls ``generate``.

        Response-position states are after consuming the corresponding token.
        EOS and other response special tokens are excluded from primary pools.
        """
        import torch

        layers = self.default_layers if layers is None else layers
        prompt_text = self._apply_chat_template(messages)
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"][0]
        response_ids = self.tokenizer(
            response_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        if response_ids.numel() == 0:
            raise ValueError("Cannot replay an empty recorded response.")
        prompt_len, response_len = int(prompt_ids.numel()), int(response_ids.numel())
        legacy_include_eos = bool(
            original_max_new_tokens
            and response_len < original_max_new_tokens
            and self.tokenizer.eos_token_id is not None
        )
        suffix = (
            torch.tensor([self.tokenizer.eos_token_id], dtype=response_ids.dtype)
            if legacy_include_eos else response_ids[:0]
        )
        full_ids = torch.cat([prompt_ids, response_ids, suffix]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(full_ids, output_hidden_states=True)

        snapshots = {}
        for layer_idx in layers:
            h = out.hidden_states[layer_idx][0]
            response_h = h[prompt_len : prompt_len + response_len]
            pools = {
                "pre_generation": h[prompt_len - 1],
                "early_response": response_h[: min(16, response_len)].mean(dim=0),
                "full_response": response_h.mean(dim=0),
                "final_window": response_h[-min(8, response_len) :].mean(dim=0),
                "final_token": response_h[-1],
                "legacy_full_response": h[
                    prompt_len : prompt_len + response_len + int(legacy_include_eos)
                ].mean(dim=0),
            }
            for name, value in pools.items():
                snapshots[(int(layer_idx), name)] = value.float().cpu().numpy()

        metadata = {
            "prompt_token_count": prompt_len,
            "response_token_count": response_len,
            "response_start_index": prompt_len,
            "response_end_index": prompt_len + response_len - 1,
            "early_response_window": min(16, response_len),
            "early_response_text": self.tokenizer.decode(
                response_ids[: min(16, response_len)], skip_special_tokens=True
            ),
            "full_response_window": response_len,
            "final_window": min(8, response_len),
            "final_token_window": 1,
            "response_special_tokens_included": False,
            "eos_included": False,
            "legacy_validation_eos_included": legacy_include_eos,
            "state_semantics": "response states are after consuming the indexed recorded token",
            "chat_template_sha256": hashlib.sha256(
                (self.tokenizer.chat_template or "").encode("utf-8")
            ).hexdigest(),
        }
        return snapshots, metadata


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


def load_model_registry(path: str = "configs/models.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["models"]


def build_client(model_key: str, registry: Optional[dict] = None, load_in_4bit: bool = False):
    registry = registry or load_model_registry()
    spec = registry[model_key]
    if spec["backend"] == "hf_local":
        return LocalHFClient(
            hf_id=spec["hf_id"],
            default_layers=spec.get("default_layers"),
            load_in_4bit=load_in_4bit,
            revision=spec.get("revision"),
            tokenizer_revision=spec.get("tokenizer_revision"),
        )
    elif spec["backend"] in ("anthropic", "openai"):
        return ApiClient(backend=spec["backend"], api_model=spec["api_model"])
    else:
        raise ValueError(f"Unknown backend for {model_key}: {spec['backend']}")
