from __future__ import annotations

from typing import Dict, List

import numpy as np


class YieldAgent:
    """Combine normalized agent outputs into interpretable crop and scenario yields."""

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def estimate_crop_output(
        self,
        crop_profile: Dict[str, object],
        context: Dict[str, object],
        allocation_share: float,
        decision_surface_ha: float,
        soil_summary: Dict[str, float],
        water_summary: Dict[str, float],
        climate_summary: Dict[str, float],
        fertilizer_summary: Dict[str, float],
        farm_summary: Dict[str, float],
        compatibility_summary: Dict[str, float],
        fuzzy_summary: Dict[str, object],
    ) -> Dict[str, float]:
        base_yield = float(crop_profile.get("base_yield_t_ha", 1.0))
        normalized_base = float(crop_profile.get("normalized_base_yield_score", 0.5))
        historical_mean_yield = float(crop_profile.get("historical_mean_yield_t_ha", base_yield))
        yield_reference = float(context.get("yield_reference_t_ha", base_yield))

        normalized_component = self._clamp(
            0.18 * float(soil_summary["normalized_score"])
            + 0.20 * float(water_summary["normalized_score"])
            + 0.16 * float(climate_summary["normalized_score"])
            + 0.10 * float(fertilizer_summary["normalized_score"])
            + 0.10 * float(farm_summary["normalized_score"])
            + 0.12 * float(compatibility_summary["normalized_score"])
            + 0.14 * float(fuzzy_summary["normalized_score"]),
            0.0,
            1.0,
        )
        multiplicative_effect = (
            float(soil_summary["factor"])
            * float(water_summary["factor"])
            * float(climate_summary["factor"])
            * float(fertilizer_summary["factor"])
            * float(farm_summary["factor"])
            * float(compatibility_summary["factor"])
            * float(fuzzy_summary["modifier"])
        )
        damped_multiplier = self._clamp(
            0.74 + 0.24 * normalized_component + 0.18 * (multiplicative_effect - 1.0),
            0.72,
            1.28,
        )
        crop_scale_cap = max(base_yield * 1.30, historical_mean_yield * 1.18, yield_reference * (0.85 + 0.55 * normalized_base))
        expected_yield_t_ha = self._clamp(
            base_yield * damped_multiplier,
            min(base_yield, historical_mean_yield) * 0.75,
            crop_scale_cap,
        )
        allocated_surface_ha = decision_surface_ha * allocation_share
        expected_output_tonnes = allocated_surface_ha * expected_yield_t_ha

        return {
            "allocated_surface_ha": round(allocated_surface_ha, 3),
            "expected_yield_t_ha": round(expected_yield_t_ha, 4),
            "expected_output_tonnes": round(expected_output_tonnes, 3),
            "yield_multiplier": round(damped_multiplier, 4),
            "normalized_component_score": round(normalized_component, 4),
            "fuzzy_suitability_score": round(float(fuzzy_summary["normalized_score"]), 4),
            "resource_balance_score": round(float(fuzzy_summary["resource_balance_score"]), 4),
        }

    def aggregate_scenario(
        self,
        crop_details: List[Dict[str, object]],
        farm_summary: Dict[str, float],
        decision_surface_ha: float,
    ) -> Dict[str, object]:
        total_output = float(sum(float(item["expected_output_tonnes"]) for item in crop_details))
        weighted_yield = total_output / max(decision_surface_ha, 1e-6)

        def weighted_average(field: str) -> float:
            return float(sum(float(item[field]) * float(item["allocation_share"]) for item in crop_details))

        best_crop = max(crop_details, key=lambda item: float(item["expected_output_tonnes"])) if crop_details else {}

        return {
            "decision_surface_ha": round(decision_surface_ha, 3),
            "weighted_expected_yield_t_ha": round(weighted_yield, 4),
            "total_expected_output_tonnes": round(total_output, 3),
            "average_soil_score": round(weighted_average("soil_score"), 4),
            "average_water_score": round(weighted_average("water_score"), 4),
            "average_climate_score": round(weighted_average("climate_score"), 4),
            "average_fertilizer_score": round(weighted_average("fertilizer_score"), 4),
            "average_compatibility_score": round(weighted_average("compatibility_score"), 4),
            "average_fuzzy_score": round(weighted_average("fuzzy_score"), 4),
            "resource_balance_score": round(weighted_average("resource_balance_score"), 4),
            "average_water_stress": round(weighted_average("water_stress"), 4),
            "farm_type": farm_summary["strategy"],
            "farm_factor": farm_summary["factor"],
            "best_crop": best_crop.get("crop", "n/a"),
            "best_crop_output_tonnes": round(float(best_crop.get("expected_output_tonnes", 0.0)), 3),
            "crop_details": crop_details,
        }
