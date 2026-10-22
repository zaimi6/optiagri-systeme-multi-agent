from __future__ import annotations

from typing import Dict

import numpy as np


class WaterAgent:
    """Estimate water sufficiency and water stress for each crop allocation."""

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def _normalize(self, value: float, low: float, high: float) -> float:
        if np.isclose(low, high):
            return 0.5
        return self._clamp((value - low) / (high - low), 0.0, 1.0)

    @staticmethod
    def _label(score: float) -> str:
        if score < 0.35:
            return "low"
        if score < 0.65:
            return "medium"
        return "high"

    def assess_crop(
        self,
        crop_profile: Dict[str, float],
        context: Dict[str, object],
        allocation_share: float,
    ) -> Dict[str, float]:
        water_context = self._normalize(float(context.get("water_availability_index", 0.90)), 0.50, 1.18)
        precipitation_resilience = self._clamp(float(context.get("precipitation_stress_index", 0.55)), 0.0, 1.0)
        crop_demand_score = self._normalize(float(crop_profile.get("water_demand", 0.86)), 0.70, 1.02)
        allocation_pressure = self._clamp((allocation_share - 0.40) / 0.60, 0.0, 1.0)

        supply_score = self._clamp(0.65 * water_context + 0.35 * precipitation_resilience, 0.0, 1.0)
        normalized_score = self._clamp(
            0.60 * supply_score
            + 0.24 * (1.0 - crop_demand_score)
            + 0.16 * (1.0 - allocation_pressure),
            0.0,
            1.0,
        )
        stress = self._clamp(
            max(crop_demand_score - supply_score, 0.0) + 0.35 * allocation_pressure,
            0.0,
            1.0,
        )
        factor = self._clamp(0.78 + 0.28 * normalized_score - 0.12 * stress, 0.70, 1.08)

        return {
            "normalized_score": round(normalized_score, 4),
            "availability_score": round(normalized_score, 4),
            "supply_score": round(supply_score, 4),
            "water_demand_score": round(crop_demand_score, 4),
            "allocation_pressure": round(allocation_pressure, 4),
            "stress": round(stress, 4),
            "factor": round(factor, 4),
            "label": self._label(normalized_score),
            "note": (
                f"Water sufficiency for {crop_profile.get('canonical_crop', 'crop')} is "
                f"{normalized_score:.2f} with stress {stress:.2f}."
            ),
        }
