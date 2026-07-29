"""Compatibility entry point for the Q1 transcript browser.

The browser was renamed to ``q1_transcript_browser.py``, but some existing
workspace port configurations still launch this historical path.
"""

from pathlib import Path
import sys

# Streamlit may execute this file with only ``src/q1`` on ``sys.path``.
# Add the repository root so imports through the ``src`` package still work.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.q1.q1_transcript_browser import main, transcript_markdown

__all__ = ["main", "transcript_markdown"]


if __name__ == "__main__":
    main()
