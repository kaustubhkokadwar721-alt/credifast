"""History-coverage confidence routing, separate from credit-risk prediction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistoryConfidencePolicy:
    policy_version: str = "history-confidence-0.1.0"
    minimum_sources_for_standard_route: int = 4
    high_confidence_boundary_distance: float = 0.01
    risk_band_boundaries: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)

    @classmethod
    def from_json(cls, path: str | Path) -> HistoryConfidencePolicy:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            policy_version=str(values["policy_version"]),
            minimum_sources_for_standard_route=int(
                values["minimum_sources_for_standard_route"]
            ),
            high_confidence_boundary_distance=float(
                values["high_confidence_boundary_distance"]
            ),
            risk_band_boundaries=tuple(float(value) for value in values["risk_band_boundaries"]),
        )


@dataclass(frozen=True)
class ConfidenceAssessment:
    history_segment: str
    available_sources: int
    confidence: str
    route: str
    reasons: tuple[str, ...]
    policy_version: str


def _available_source_count(source_flags: Mapping[str, int | bool | None]) -> int:
    values = [value for key, value in source_flags.items() if key.startswith("HAS_")]
    if not values:
        raise ValueError("at least one HAS_* source flag is required")
    for value in values:
        if value not in (0, 1, False, True, None):
            raise ValueError("source flags must be binary or null")
    return sum(int(bool(value)) for value in values)


def assess_history_confidence(
    probability: float,
    source_flags: Mapping[str, int | bool | None],
    policy: HistoryConfidencePolicy | None = None,
) -> ConfidenceAssessment:
    """Route incomplete histories to review without changing predicted risk."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    active_policy = policy or HistoryConfidencePolicy()
    available = _available_source_count(source_flags)
    if available <= 1:
        segment = "thin_0_to_1_sources"
    elif available <= 3:
        segment = "partial_2_to_3_sources"
    else:
        segment = "full_4_to_5_sources"

    if available < active_policy.minimum_sources_for_standard_route:
        return ConfidenceAssessment(
            history_segment=segment,
            available_sources=available,
            confidence="LOW",
            route="MANUAL_REVIEW_DATA_LIMITED",
            reasons=(
                "Historical source coverage is below the standard scoring-route minimum",
                "Segment calibration error is higher than for full-history applications",
            ),
            policy_version=active_policy.policy_version,
        )

    distance = min(abs(probability - boundary) for boundary in active_policy.risk_band_boundaries)
    if distance < active_policy.high_confidence_boundary_distance:
        return ConfidenceAssessment(
            history_segment=segment,
            available_sources=available,
            confidence="MEDIUM",
            route="MANUAL_REVIEW_NEAR_RISK_BOUNDARY",
            reasons=("Predicted risk is close to a configured grade boundary",),
            policy_version=active_policy.policy_version,
        )
    return ConfidenceAssessment(
        history_segment=segment,
        available_sources=available,
        confidence="HIGH",
        route="STANDARD_REVIEW",
        reasons=("At least four historical source families are available",),
        policy_version=active_policy.policy_version,
    )
