from __future__ import annotations


def expected_annual_loss(probability: float, impact: float) -> float:
    return float(max(probability, 0) * max(impact, 0))

