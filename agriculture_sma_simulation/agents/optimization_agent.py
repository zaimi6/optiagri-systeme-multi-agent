from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class OptimizationAgent:
    """Generate agricultural allocation scenarios and rank them transparently."""

    PAIR_RATIOS: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.5),
        (0.7, 0.3),
        (0.6, 0.4),
    )
    TRIPLE_RATIOS: Tuple[Tuple[float, float, float], ...] = (
        (0.5, 0.25, 0.25),
        (0.34, 0.33, 0.33),
    )
    MAX_SCENARIOS = 16
    ML_BLEND_WEIGHT = 0.18

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _normalize_array(values: Sequence[float]) -> List[float]:
        array = np.array(list(values), dtype=float)
        if array.size == 0:
            return []
        if np.isclose(array.min(), array.max()):
            return [0.5 for _ in array]
        normalized = (array - array.min()) / (array.max() - array.min())
        return [float(item) for item in normalized]

    @staticmethod
    def _scenario_name(allocation: Dict[str, float]) -> str:
        ordered = sorted(allocation.items(), key=lambda item: (-item[1], item[0]))
        return " + ".join(f"{int(round(share * 100))}% {crop}" for crop, share in ordered)

    @staticmethod
    def _allocation_text(allocation: Dict[str, float]) -> str:
        ordered = sorted(allocation.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{int(round(share * 100))}% of land to {crop}" for crop, share in ordered)

    def _crop_priority(self, crop_profile: Dict[str, object]) -> float:
        return float(
            0.40 * crop_profile.get("normalized_base_yield_score", 0.5)
            + 0.20 * crop_profile.get("normalized_stability_score", 0.5)
            + 0.20 * crop_profile.get("normalized_trend_score", 0.5)
            + 0.20 * crop_profile.get("normalized_surface_score", 0.5)
        )

    def generate_scenarios(
        self,
        crop_profiles: Dict[str, Dict[str, object]],
        decision_surface_ha: float,
    ) -> List[Dict[str, object]]:
        ranked_crops = sorted(
            crop_profiles,
            key=lambda crop: self._crop_priority(crop_profiles[crop]),
            reverse=True,
        )
        preferred_crops = [crop for crop in ("Wheat", "Oranges", "Tomatoes", "Olives") if crop in crop_profiles]
        candidate_crops = (preferred_crops + [crop for crop in ranked_crops if crop not in preferred_crops])[
            : min(4, len(ranked_crops))
        ]
        scenarios: List[Dict[str, object]] = []
        seen = set()

        def add_scenario(crops: Sequence[str], ratios: Sequence[float], mix_type: str) -> None:
            allocation = {crop: round(float(ratio), 4) for crop, ratio in zip(crops, ratios)}
            if not allocation or not np.isclose(sum(allocation.values()), 1.0):
                return
            name = self._scenario_name(allocation)
            if name in seen:
                return
            seen.add(name)
            scenarios.append(
                {
                    "name": name,
                    "allocation": allocation,
                    "allocation_text": self._allocation_text(allocation),
                    "decision_surface_ha": float(decision_surface_ha),
                    "mix_type": mix_type,
                    "focus_crop": max(allocation, key=allocation.get),
                }
            )

        for crop in candidate_crops:
            add_scenario((crop,), (1.0,), "single_crop")

        for crop_a, crop_b in combinations(candidate_crops, 2):
            for ratios in self.PAIR_RATIOS:
                add_scenario((crop_a, crop_b), ratios, "two_crop_mix")
                if not np.isclose(ratios[0], ratios[1]):
                    add_scenario((crop_b, crop_a), ratios, "two_crop_mix")

        if len(candidate_crops) >= 3:
            top_three = tuple(candidate_crops[:3])
            for ratios in self.TRIPLE_RATIOS:
                add_scenario(top_three, ratios, "three_crop_mix")

        ordered = sorted(
            scenarios,
            key=lambda scenario: (
                len(scenario["allocation"]),
                -max(scenario["allocation"].values()),
                scenario["name"],
            ),
        )
        return ordered[: self.MAX_SCENARIOS]

    def build_scenario_record(
        self,
        scenario: Dict[str, object],
        aggregated: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "scenario_name": str(scenario["name"]),
            "allocation": dict(scenario["allocation"]),
            "allocation_text": str(scenario["allocation_text"]),
            "focus_crop": str(scenario["focus_crop"]),
            "mix_type": str(scenario["mix_type"]),
            "weighted_expected_yield_t_ha": float(aggregated["weighted_expected_yield_t_ha"]),
            "total_expected_output_tonnes": float(aggregated["total_expected_output_tonnes"]),
            "decision_surface_ha": float(aggregated["decision_surface_ha"]),
            "average_soil_score": float(aggregated["average_soil_score"]),
            "average_water_score": float(aggregated["average_water_score"]),
            "average_climate_score": float(aggregated["average_climate_score"]),
            "average_fertilizer_score": float(aggregated["average_fertilizer_score"]),
            "average_compatibility_score": float(aggregated["average_compatibility_score"]),
            "average_fuzzy_score": float(aggregated["average_fuzzy_score"]),
            "resource_balance_score": float(aggregated["resource_balance_score"]),
            "average_water_stress": float(aggregated["average_water_stress"]),
            "farm_type": str(aggregated["farm_type"]),
            "farm_factor": float(aggregated["farm_factor"]),
            "ml_enabled": bool(aggregated.get("ml_enabled", False)),
            "ml_model_name": str(aggregated.get("ml_model_name", "unavailable")),
            "ml_predicted_yield_t_ha": float(aggregated.get("ml_predicted_yield_t_ha", aggregated["weighted_expected_yield_t_ha"])),
            "ml_total_output_tonnes": float(aggregated.get("ml_total_output_tonnes", aggregated["total_expected_output_tonnes"])),
            "crop_details": list(aggregated["crop_details"]),
        }

    def rank_scenarios(self, scenario_results: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        results = [dict(item) for item in scenario_results]
        if not results:
            return []

        output_scores = self._normalize_array(item["total_expected_output_tonnes"] for item in results)
        yield_scores = self._normalize_array(item["weighted_expected_yield_t_ha"] for item in results)
        fuzzy_scores = self._normalize_array(item["average_fuzzy_score"] for item in results)
        resource_scores = self._normalize_array(item["resource_balance_score"] for item in results)
        ml_scores = self._normalize_array(
            item.get("ml_predicted_yield_t_ha", item["weighted_expected_yield_t_ha"]) for item in results
        )
        ml_enabled = any(bool(item.get("ml_enabled", False)) for item in results)

        scored_results: List[Dict[str, object]] = []
        for index, item in enumerate(results):
            diversification_bonus = 0.025 * max(len(item["allocation"]) - 1, 0)
            stress_penalty = 0.07 * float(item["average_water_stress"])
            agent_score = self._clamp(
                0.50 * output_scores[index]
                + 0.22 * yield_scores[index]
                + 0.18 * fuzzy_scores[index]
                + 0.10 * resource_scores[index]
                + diversification_bonus
                - stress_penalty,
                0.0,
                1.2,
            )
            optimization_score = (
                self._clamp(
                    (1.0 - self.ML_BLEND_WEIGHT) * agent_score + self.ML_BLEND_WEIGHT * ml_scores[index],
                    0.0,
                    1.2,
                )
                if ml_enabled
                else agent_score
            )
            scored_results.append(
                {
                    **item,
                    "normalized_output_score": round(output_scores[index], 4),
                    "normalized_yield_score": round(yield_scores[index], 4),
                    "normalized_fuzzy_score": round(fuzzy_scores[index], 4),
                    "normalized_resource_score": round(resource_scores[index], 4),
                    "agent_score": round(agent_score, 4),
                    "ml_prediction_score": round(ml_scores[index], 4),
                    "ml_weight": round(self.ML_BLEND_WEIGHT if ml_enabled else 0.0, 4),
                    "optimization_score": round(optimization_score, 4),
                }
            )

        ranked = sorted(
            scored_results,
            key=lambda item: (
                float(item["optimization_score"]),
                float(item["total_expected_output_tonnes"]),
                float(item["weighted_expected_yield_t_ha"]),
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            item["win_reason"] = self._build_win_reason(item)
        return ranked

    @staticmethod
    def select_best(ranked_results: List[Dict[str, object]]) -> Dict[str, object]:
        return ranked_results[0] if ranked_results else {}

    def current_best(self, scenario_results: Iterable[Dict[str, object]]) -> Dict[str, object]:
        return self.select_best(self.rank_scenarios(scenario_results))

    def compute_decision_metrics(self, ranked_results: Sequence[Dict[str, object]]) -> Dict[str, float]:
        if not ranked_results:
            return {
                "dcg_score": 0.0,
                "best_scenario_score": 0.0,
                "score_margin": 0.0,
            }

        relevance_scores = [
            max(float(item.get("optimization_score", 0.0)), 0.0)
            for item in ranked_results
        ]
        dcg_score = sum(
            relevance / np.log2(index + 2)
            for index, relevance in enumerate(relevance_scores)
        )
        best_score = relevance_scores[0]
        second_score = relevance_scores[1] if len(relevance_scores) > 1 else 0.0
        return {
            "dcg_score": round(float(dcg_score), 4),
            "best_scenario_score": round(float(best_score), 4),
            "score_margin": round(float(best_score - second_score), 4),
        }

    def _build_win_reason(self, result: Dict[str, object]) -> str:
        components = [
            f"optimization score {float(result['optimization_score']):.2f}",
            f"output {float(result['total_expected_output_tonnes']):.1f} t",
            f"yield {float(result['weighted_expected_yield_t_ha']):.2f} t/ha",
            f"fuzzy suitability {float(result['average_fuzzy_score']):.2f}",
            f"resource balance {float(result['resource_balance_score']):.2f}",
        ]
        if bool(result.get("ml_enabled", False)):
            components.append(
                f"ML-predicted yield {float(result.get('ml_predicted_yield_t_ha', 0.0)):.2f} t/ha"
            )
        return "Selected because it combines " + ", ".join(components) + "."
