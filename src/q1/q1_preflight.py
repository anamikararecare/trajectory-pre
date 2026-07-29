"""Credential and checkpoint-access preflight for Q1 generation."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

from src.common.llm_client import load_model_registry


def main() -> None:
    load_dotenv(override=False)
    token = (os.environ.get("HF_TOKEN") or "").strip()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not token:
        raise SystemExit("FAIL: HF_TOKEN is missing from the environment/.env")
    if not openai_key:
        raise SystemExit(
            "FAIL: OPENAI_API_KEY is missing from the environment/.env"
        )

    registry_path = Path("configs/q1_minimum_models.yaml")
    registry = load_model_registry(str(registry_path))
    account = HfApi(token=token).whoami()["name"]
    print(f"Hugging Face account: {account}")
    denied = []
    for key, spec in registry.items():
        hf_id = spec["hf_id"]
        try:
            get_hf_file_metadata(
                hf_hub_url(hf_id, "config.json"),
                token=token,
            )
        except Exception as error:
            denied.append((key, hf_id, type(error).__name__))
            print(f"DENIED  {key:18s} {hf_id} ({type(error).__name__})")
        else:
            print(f"OK      {key:18s} {hf_id}")

    if denied:
        print(
            f"\nFAIL: {len(denied)} of {len(registry)} Q1 checkpoints are "
            "not accessible. Accept their repository terms using the account "
            f"shown above, then rerun this preflight."
        )
        raise SystemExit(1)
    print(
        f"\nPASS: credentials are present and all {len(registry)} Q1 "
        "checkpoint config files are accessible."
    )


if __name__ == "__main__":
    main()
