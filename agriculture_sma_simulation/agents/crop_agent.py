from __future__ import annotations

from typing import Dict

import numpy as np


class CropAgent:
    """Manage crop-specific compatibility logic inside each evaluated scenario."""

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def describe_scenario(scenario: Dict[str, object]) -> str:
        allocation = dict(scenario["allocation"])
        ordered = sorted(allocation.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{int(round(share * 100))}% of land to {crop}" for crop, share in ordered)

    @staticmethod
    def focus_crop(allocation: Dict[str, float]) -> str:
        return max(allocation, key=allocation.get)

    @staticmethod
    def _label(score: float) -> str:
        if score < 0.35:
            return "weak"
        if score < 0.65:
            return "medium"
        return "strong"

    def assess_compatibility(
        self,
        crop_profile: Dict[str, object],
        soil_summary: Dict[str, float],
        water_summary: Dict[str, float],
        climate_summary: Dict[str, float],
        farm_summary: Dict[str, float],
    ) -> Dict[str, float]:
        base_potential = float(
            0.42 * crop_profile.get("normalized_base_yield_score", 0.5)
            + 0.28 * crop_profile.get("normalized_stability_score", 0.5)
            + 0.30 * crop_profile.get("normalized_trend_score", 0.5)
        )
        demand_pressure = self._clamp(
            (float(crop_profile.get("water_demand", 0.86)) - 0.70) / 0.32,
            0.0,
            1.0,
        )
        resource_support = self._clamp(
            0.28 * float(soil_summary["normalized_score"])
            + 0.28 * float(water_summary["normalized_score"])
            + 0.24 * float(climate_summary["normalized_score"])
            + 0.20 * float(farm_summary["normalized_score"]),
            0.0,
            1.0,
        )
        compatibility_score = self._clamp(
            0.52 * resource_support + 0.30 * base_potential + 0.18 * (1.0 - demand_pressure),
            0.0,
            1.0,
        )
        factor = self._clamp(0.80 + 0.26 * compatibility_score, 0.76, 1.10)

        return {
            "normalized_score": round(compatibility_score, 4),
            "compatibility_score": round(compatibility_score, 4),
            "base_potential_score": round(base_potential, 4),
            "resource_support_score": round(resource_support, 4),
            "demand_pressure": round(demand_pressure, 4),
            "factor": round(factor, 4),
            "label": self._label(compatibility_score),
            "note": (
                f"Crop compatibility for {crop_profile.get('canonical_crop', 'crop')} is "
                f"{compatibility_score:.2f} ({self._label(compatibility_score)})."
            ),
        }
