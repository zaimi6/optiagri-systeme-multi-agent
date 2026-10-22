from __future__ import annotations

from typing import Dict

import numpy as np


class FarmTypeAgent:
    """Select the farm strategy that best matches resources and crop demand."""

    STRATEGIES = {
        "rainfed": {
            "base_factor": 0.93,
            "water_support_score": 0.38,
            "fertilizer_support_score": 0.42,
            "resource_focus_score": 0.48,
        },
        "irrigated": {
            "base_factor": 1.04,
            "water_support_score": 0.84,
            "fertilizer_support_score": 0.64,
            "resource_focus_score": 0.70,
        },
        "intensive": {
            "base_factor": 1.06,
            "water_support_score": 0.66,
            "fertilizer_support_score": 0.86,
            "resource_focus_score": 0.72,
        },
        "mixed": {
            "base_factor": 1.00,
            "water_support_score": 0.58,
            "fertilizer_support_score": 0.58,
            "resource_focus_score": 0.60,
        },
    }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def _normalize(self, value: float, low: float, high: float) -> float:
        if np.isclose(low, high):
            return 0.5
        return self._clamp((value - low) / (high - low), 0.0, 1.0)

    def recommend(
        self,
        allocation: Dict[str, float],
        crop_profiles: Dict[str, Dict[str, object]],
        context: Dict[str, object],
    ) -> Dict[str, float]:
        weighted_demand = sum(float(crop_profiles[crop]["water_demand"]) * share for crop, share in allocation.items())
        demand_score = self._normalize(weighted_demand, 0.70, 1.02)
        water_context = self._normalize(float(context.get("water_availability_index", 0.90)), 0.50, 1.18)
        soil_context = self._normalize(float(context.get("soil_quality_index", 0.95)), 0.70, 1.08)
        fertilizer_context = self._normalize(float(context.get("fertilizer_efficiency_index", 0.92)), 0.75, 1.12)
        resource_balance = self._normalize(float(context.get("resource_balance_index", 0.85)), 0.40, 1.00)
        diversification_score = self._clamp((len(allocation) - 1) / 2.0, 0.0, 1.0)

        if water_context < 0.45 and demand_score < 0.50:
            strategy = "rainfed"
        elif water_context > 0.58 and demand_score > 0.62:
            strategy = "irrigated"
        elif soil_context > 0.65 and fertilizer_context > 0.60 and diversification_score < 0.40:
            strategy = "intensive"
        else:
            strategy = "mixed"

        config = self.STRATEGIES[strategy]
        resource_fit_score = self._clamp(
            1.0
            - 0.55 * abs(demand_score - float(config["water_support_score"]))
            - 0.25 * abs(resource_balance - float(config["resource_focus_score"])),
            0.0,
            1.0,
        )
        normalized_score = self._clamp(
            0.52 * resource_fit_score
            + 0.24 * resource_balance
            + 0.14 * diversification_score
            + 0.10 * fertilizer_context,
            0.0,
            1.0,
        )
        factor = self._clamp(
            float(config["base_factor"]) + 0.16 * (normalized_score - 0.5),
            0.86,
            1.10,
        )

        return {
            "strategy": strategy,
            "weighted_demand": round(weighted_demand, 4),
            "demand_score": round(demand_score, 4),
            "resource_fit_score": round(resource_fit_score, 4),
            "water_support_score": round(float(config["water_support_score"]), 4),
            "fertilizer_support_score": round(float(config["fertilizer_support_score"]), 4),
            "normalized_score": round(normalized_score, 4),
            "factor": round(factor, 4),
            "note": (
                f"Farm strategy '{strategy}' selected with normalized resource fit "
                f"{resource_fit_score:.2f}."
            ),
        }
