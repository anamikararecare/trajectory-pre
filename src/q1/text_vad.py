"""Deterministic, cached text VAD scoring for generated Q1 responses."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VAD_MODEL = "RobroKools/vad-bert"
VAD_COLUMNS = (
    "expressed_valence",
    "expressed_arousal",
    "expressed_dominance",
)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_vad_cache_path(
    run_dir: str | Path,
    model_name: str = DEFAULT_VAD_MODEL,
) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return Path(run_dir) / "derived" / f"text_vad__{safe_model}.csv"


def _predict_transformer_vad(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> tuple[np.ndarray, str | None]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {
                name: tensor.to(resolved_device)
                for name, tensor in inputs.items()
            }
            logits = model(**inputs).logits.detach().float().cpu().numpy()
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            if logits.shape[1] != 3:
                raise ValueError(
                    f"VAD model {model_name!r} returned {logits.shape[1]} "
                    "outputs; expected [valence, arousal, dominance]."
                )
            predictions.append(logits)
    revision = getattr(model.config, "_commit_hash", None)
    return np.concatenate(predictions, axis=0), revision


def _read_cache(path: Path, model_name: str) -> pd.DataFrame:
    columns = [
        "text_sha256",
        "vad_model",
        "vad_model_revision",
        *VAD_COLUMNS,
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    cache = pd.read_csv(path)
    missing = set(columns).difference(cache.columns)
    if missing:
        raise ValueError(f"VAD cache is missing columns: {sorted(missing)}")
    cache = cache[cache["vad_model"].eq(model_name)].copy()
    if cache["text_sha256"].duplicated().any():
        raise ValueError(f"VAD cache has duplicate text hashes: {path}")
    return cache[columns]


def _write_cache(cache: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
    ) as handle:
        temporary = Path(handle.name)
        cache.to_csv(handle, index=False)
    temporary.replace(path)


def add_text_vad_scores(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_VAD_MODEL,
    cache_path: str | Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    predictor: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Add expressed VAD scores, inferring only uncached unique response texts.

    Cache rows contain SHA-256 hashes rather than response text. ``predictor`` is
    an injectable deterministic test hook; production uses the HF checkpoint.
    """
    if "text" not in frame:
        raise ValueError("Text VAD scoring requires a text column.")
    if batch_size <= 0:
        raise ValueError("VAD batch size must be positive.")
    output = frame.copy()
    texts = output["text"].fillna("").astype(str)
    hashes = texts.map(_text_hash)
    cache_file = Path(cache_path) if cache_path is not None else None
    cache = (
        _read_cache(cache_file, model_name)
        if cache_file is not None
        else _read_cache(Path("__missing_vad_cache__"), model_name)
    )
    cached_hashes = set(cache["text_sha256"])
    unique_missing = (
        pd.DataFrame({"text_sha256": hashes, "text": texts})
        .drop_duplicates("text_sha256")
        .loc[lambda values: ~values["text_sha256"].isin(cached_hashes)]
    )
    if not unique_missing.empty:
        missing_texts = unique_missing["text"].tolist()
        if predictor is None:
            predictions, revision = _predict_transformer_vad(
                missing_texts,
                model_name=model_name,
                batch_size=batch_size,
                device=device,
            )
        else:
            predictions = np.asarray(predictor(missing_texts), dtype=float)
            revision = "test_predictor"
        if predictions.shape != (len(unique_missing), 3):
            raise ValueError(
                "VAD predictor must return shape "
                f"({len(unique_missing)}, 3), received {predictions.shape}."
            )
        additions = pd.DataFrame(
            {
                "text_sha256": unique_missing["text_sha256"].to_numpy(),
                "vad_model": model_name,
                "vad_model_revision": revision,
                **{
                    column: predictions[:, index]
                    for index, column in enumerate(VAD_COLUMNS)
                },
            }
        )
        cache = pd.concat([cache, additions], ignore_index=True)
        if cache_file is not None:
            _write_cache(cache, cache_file)
    lookup = cache.set_index("text_sha256")
    for column in VAD_COLUMNS:
        output[column] = pd.to_numeric(hashes.map(lookup[column]), errors="raise")
    output["vad_model"] = model_name
    output["vad_model_revision"] = hashes.map(lookup["vad_model_revision"])
    output["text_sha256"] = hashes
    return output

