from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import load_yaml


def load_financial_constants(path: str | Path = "configs/financial.yaml") -> dict[str, Any]:
    return load_yaml(path)

