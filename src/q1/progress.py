"""Thread-safe live and JSONL progress reporting for long Q1 experiments."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProgressReporter:
    """Emit concise terminal updates and an append-only machine-readable log."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.started = time.monotonic()
        self._stage_started: dict[str, float] = {}
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(
        self,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        **details: Any,
    ) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.started
            stage_started = self._stage_started.setdefault(stage, self.started)
            stage_elapsed = now - stage_started
            payload: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "elapsed_seconds": round(elapsed, 3),
                **details,
            }
            summary = f"[progress] {stage}"
            if completed is not None:
                payload["completed"] = int(completed)
                summary += f" {completed}"
            if total is not None:
                payload["total"] = int(total)
                summary += f"/{total}"
            if completed is not None and total is not None and total > 0:
                fraction = min(max(completed / total, 0.0), 1.0)
                width = 30
                filled = min(width, int(fraction * width))
                bar = "#" * filled + "-" * (width - filled)
                payload["fraction_complete"] = fraction
                summary += f" [{bar}] {fraction:6.2%}"
            if (
                completed is not None
                and total is not None
                and completed > 0
                and total >= completed
            ):
                rate = completed / max(stage_elapsed, 1e-9)
                eta = (total - completed) / max(rate, 1e-9)
                payload["rate_per_second"] = rate
                payload["eta_seconds"] = eta
                summary += f" elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
            identity = " ".join(
                f"{key}={value}"
                for key, value in details.items()
                if value is not None
            )
            if identity:
                summary += f" {identity}"
            print(summary, flush=True)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, default=str) + "\n")

