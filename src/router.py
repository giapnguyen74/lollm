"""
router.py — shared infra. Probe the architecture, route to the family.

This is the *entire* job: read the model_type the loader probed, look it up in the
family registry, return the family. It never touches tensor names or weights — all
weight handling (and every format quirk) lives in the family.
"""

from __future__ import annotations

import models                # noqa: F401  — importing registers the models


def route(model_type: str):
    """model_type → Family (raises ValueError if unknown — fail loud)."""
    return models.get(model_type)
