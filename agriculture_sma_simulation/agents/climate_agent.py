from __future__ import annotations

from typing import Dict

import numpy as np


class ClimateAgent:
    """Estimate climate favorability using normalized Moroccan context indicators."""

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
            return "unfavorable"
        if score < 0.65:
            return "average"
        return "favorable"

    def assess_crop(self, crop_profile: Dict[str, float], context: Dict[str, object]) -> Dict[str, float]:
        climate_context = self._normalize(float(context.get("climate_favorability_index", 0.88)), 0.58, 1.16)
        precipitation_resilience = self._clamp(float(context.get("precipitation_stress_index", 0.55)), 0.0, 1.0)
        crop_preference = self._normalize(float(crop_profile.get("climate_preference", 0.90)), 0.84, 0.97)
        trend_score = float(
            crop_profile.get(
                "normalized_trend_score",
                self._normalize(float(crop_profile.get("trend_score", 1.0)), 0.82, 1.18),
            )
        )

        normalized_score = self._clamp(
            0.44 * climate_context
            + 0.24 * precipitation_resilience
            + 0.20 * crop_preference
            + 0.12 * trend_score,
            0.0,
            1.0,
        )
        factor = self._clamp(0.80 + 0.28 * normalized_score, 0.74, 1.10)

        return {
            "normalized_score": round(normalized_score, 4),
            "favorability_score": round(normalized_score, 4),
            "context_score": round(climate_context, 4),
            "trend_score": round(trend_score, 4),
            "factor": round(factor, 4),
            "label": self._label(normalized_score),
            "note": (
                f"Climate favorability for {crop_profile.get('canonical_crop', 'crop')} is "
                f"{normalized_score:.2f} ({self._label(normalized_score)})."
            ),
        }
