from __future__ import annotations

from typing import Dict, List

import numpy as np


class FuzzyLogicAgent:
    """Apply simple explainable fuzzy reasoning over normalized agent outputs."""

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _triangle(value: float, left: float, peak: float, right: float) -> float:
        if value <= left or value >= right:
            return 0.0
        if np.isclose(value, peak):
            return 1.0
        if value < peak:
            return (value - left) / max(peak - left, 1e-9)
        return (right - value) / max(right - peak, 1e-9)

    @staticmethod
    def _left_shoulder(value: float, full: float, zero: float) -> float:
        if value <= full:
            return 1.0
        if value >= zero:
            return 0.0
        return (zero - value) / max(zero - full, 1e-9)

    @staticmethod
    def _right_shoulder(value: float, zero: float, full: float) -> float:
        if value <= zero:
            return 0.0
        if value >= full:
            return 1.0
        return (value - zero) / max(full - zero, 1e-9)

    def _build_memberships(self, value: float) -> Dict[str, float]:
        clipped = self._clamp(value, 0.0, 1.0)
        return {
            "low": round(self._left_shoulder(clipped, 0.25, 0.50), 4),
            "medium": round(self._triangle(clipped, 0.25, 0.50, 0.75), 4),
            "high": round(self._right_shoulder(clipped, 0.50, 0.75), 4),
        }

    @staticmethod
    def _rule(label: str, strength: float, output: float, reason: str) -> Dict[str, object]:
        return {
            "label": label,
            "strength": round(strength, 4),
            "output": round(output, 4),
            "reason": reason,
        }

    def evaluate(
        self,
        crop_profile: Dict[str, object],
        soil_summary: Dict[str, float],
        water_summary: Dict[str, float],
        climate_summary: Dict[str, float],
        compatibility_summary: Dict[str, float],
        fertilizer_summary: Dict[str, float],
        farm_summary: Dict[str, float],
    ) -> Dict[str, object]:
        soil_score = float(soil_summary.get("normalized_score", 0.5))
        water_score = float(water_summary.get("normalized_score", 0.5))
        climate_score = float(climate_summary.get("normalized_score", 0.5))
        compatibility_score = float(compatibility_summary.get("normalized_score", 0.5))
        fertilizer_score = float(fertilizer_summary.get("normalized_score", 0.5))
        farm_score = float(farm_summary.get("normalized_score", 0.5))
        water_demand = self._clamp(
            (float(crop_profile.get("water_demand", 0.86)) - 0.70) / 0.32,
            0.0,
            1.0,
        )

        soil_membership = self._build_memberships(soil_score)
        water_membership = self._build_memberships(water_score)
        climate_membership = self._build_memberships(climate_score)
        compatibility_membership = self._build_memberships(compatibility_score)
        fertilizer_membership = self._build_memberships(fertilizer_score)
        farm_membership = self._build_memberships(farm_score)
        demand_membership = self._build_memberships(water_demand)

        rules: List[Dict[str, object]] = [
            self._rule(
                "water_penalty",
                min(water_membership["low"], demand_membership["high"]),
                0.20,
                "Low water combined with a water-intensive crop produces a strong penalty.",
            ),
            self._rule(
                "climate_water_penalty",
                min(water_membership["low"], climate_membership["low"]),
                0.24,
                "Poor water and unfavorable climate jointly reduce expected performance.",
            ),
            self._rule(
                "soil_climate_boost",
                min(soil_membership["high"], climate_membership["high"]),
                0.84,
                "Good soil and favorable climate create a strong agronomic boost.",
            ),
            self._rule(
                "balanced_conditions",
                min(
                    soil_membership["medium"],
                    water_membership["medium"],
                    climate_membership["medium"],
                ),
                0.56,
                "Balanced medium conditions keep the scenario near a stable neutral zone.",
            ),
            self._rule(
                "limited_fertilizer_recovery",
                min(fertilizer_membership["medium"], soil_membership["low"]),
                0.42,
                "Medium fertilizer efficiency cannot fully compensate for poor soil quality.",
            ),
            self._rule(
                "system_alignment",
                min(compatibility_membership["high"], farm_membership["high"]),
                0.78,
                "Strong crop compatibility and a suitable farm strategy improve reliability.",
            ),
            self._rule(
                "weak_alignment",
                min(compatibility_membership["low"], farm_membership["low"]),
                0.30,
                "Weak compatibility with a weak farm fit lowers scenario confidence.",
            ),
        ]

        active_rules = [rule for rule in rules if float(rule["strength"]) > 0.0]
        if not active_rules:
            active_rules = [
                self._rule(
                    "neutral",
                    1.0,
                    0.55,
                    "No strong fuzzy rule dominated, so the scenario stays close to neutral.",
                )
            ]

        numerator = sum(float(rule["strength"]) * float(rule["output"]) for rule in active_rules)
        denominator = sum(float(rule["strength"]) for rule in active_rules)
        fuzzy_score = self._clamp(numerator / max(denominator, 1e-9), 0.0, 1.0)
        resource_balance = self._clamp(
            0.28 * soil_score + 0.30 * water_score + 0.24 * climate_score + 0.18 * farm_score,
            0.0,
            1.0,
        )
        modifier = self._clamp(0.82 + 0.34 * fuzzy_score + 0.04 * (resource_balance - 0.5), 0.76, 1.16)

        dominant_rule = max(active_rules, key=lambda rule: (float(rule["strength"]), float(rule["output"])))
        explanation = (
            f"Fuzzy score {fuzzy_score:.2f} for {crop_profile.get('canonical_crop', 'crop')} "
            f"driven mainly by rule '{dominant_rule['label']}'."
        )

        return {
            "normalized_score": round(fuzzy_score, 4),
            "fuzzy_suitability_score": round(fuzzy_score, 4),
            "resource_balance_score": round(resource_balance, 4),
            "modifier": round(modifier, 4),
            "dominant_rule": str(dominant_rule["label"]),
            "dominant_reason": str(dominant_rule["reason"]),
            "active_rules": active_rules[:4],
            "note": explanation,
        }
