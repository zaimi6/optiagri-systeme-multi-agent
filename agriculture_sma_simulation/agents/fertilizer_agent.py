from __future__ import annotations

from typing import Dict

import numpy as np


class FertilizerAgent:
    """Estimate fertilizer contribution while clearly marking derived logic."""

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
        water_summary: Dict[str, float],
        farm_summary: Dict[str, float],
    ) -> Dict[str, float]:
        baseline = self._normalize(float(context.get("fertilizer_efficiency_index", 0.92)), 0.75, 1.12)
        responsiveness = self._normalize(float(crop_profile.get("fertilizer_responsiveness", 0.92)), 0.82, 1.02)
        water_support = float(water_summary.get("normalized_score", 0.5))
        farm_support = float(farm_summary.get("fertilizer_support_score", 0.5))
        resource_balance = self._normalize(float(context.get("resource_balance_index", 0.85)), 0.40, 1.00)

        normalized_score = self._clamp(
            0.34 * baseline
            + 0.24 * responsiveness
            + 0.18 * water_support
            + 0.14 * farm_support
            + 0.10 * resource_balance,
            0.0,
            1.0,
        )
        factor = self._clamp(0.82 + 0.22 * normalized_score, 0.78, 1.08)

        return {
            "normalized_score": round(normalized_score, 4),
            "efficiency_score": round(normalized_score, 4),
            "baseline_score": round(baseline, 4),
            "responsiveness_score": round(responsiveness, 4),
            "factor": round(factor, 4),
            "is_derived": True,
            "label": self._label(normalized_score),
            "note": (
                "Fertilizer efficiency is derived for the simulation and scaled to "
                f"{normalized_score:.2f} ({self._label(normalized_score)})."
            ),
        }
