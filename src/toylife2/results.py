"""Owned host results. Validity is explicit, including resource-limited assays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayResult:
    values: np.ndarray
    status: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("values", "status"):
            array = np.array(getattr(self, name), copy=True)
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @property
    def complete(self) -> bool:
        return bool(np.all(self.status == "complete"))
