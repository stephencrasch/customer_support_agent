"""Stable import shim for the current diagnostic tutor graph.

This keeps a short, predictable module name (`agents`) for scripts/tests while
the implementation lives in `agents_diagnostic.py`.
"""

from __future__ import annotations

from agents_diagnostic import StudyState, app

__all__ = ["StudyState", "app"]
