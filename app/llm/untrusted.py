"""Serialize untrusted conversational text for a model prompt.

The JSON keeps the text machine-readable and escapes angle brackets so a user
turn cannot spoof the delimiter tags used around untrusted blocks. The escaped
form still decodes to the original characters.
"""

from __future__ import annotations

import json
from typing import Any


def untrusted_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


__all__ = ["untrusted_json"]
