from __future__ import annotations

from typing import Dict

import numpy as np


class SoilAgent:
    """Estimate normalized soil suitability for each crop under the current context."""

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
            return "poor"
        if score < 0.65:
            return "moderate"
        return "good"

    def assess_crop(self, crop_profile: Dict[str, float], context: Dict[str, object]) -> Dict[str, float]:
        context_score = self._normalize(float(context.get("soil_quality_index", 0.90)), 0.70, 1.08)
        crop_preference = self._normalize(float(crop_profile.get("soil_preference", 0.90)), 0.82, 0.98)
        stability_score = float(
            crop_profile.get(
                "normalized_stability_score",
                self._normalize(float(crop_profile.get("yield_stability", 0.80)), 0.55, 1.0),
            )
        )
        surface_confidence = float(crop_profile.get("normalized_surface_score", 0.5))

        normalized_score = self._clamp(
            0.46 * context_score
            + 0.28 * crop_preference
            + 0.18 * stability_score
            + 0.08 * surface_confidence,
            0.0,
            1.0,
        )
        factor = self._clamp(0.82 + 0.28 * normalized_score, 0.78, 1.10)

        return {
            "normalized_score": round(normalized_score, 4),
            "suitability_score": round(normalized_score, 4),
            "context_score": round(context_score, 4),
            "crop_preference_score": round(crop_preference, 4),
            "stability_score": round(stability_score, 4),
            "factor": round(factor, 4),
            "label": self._label(normalized_score),
            "note": (
                f"Soil suitability for {crop_profile.get('canonical_crop', 'crop')} is "
                f"{normalized_score:.2f} ({self._label(normalized_score)})."
            ),
        }
