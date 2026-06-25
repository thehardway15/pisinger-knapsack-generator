"""Canonical JSON text serialization shared across modules."""

import json
from typing import Any


def dict_to_text(data: dict[str, Any]) -> str:
    """Serialize a dict to canonical JSON text: sorted keys, 4-space indent, trailing newline."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=4) + "\n"
