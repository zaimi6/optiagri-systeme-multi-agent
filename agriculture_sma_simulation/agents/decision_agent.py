from __future__ import annotations

from typing import Dict, List


class DecisionAgent:
    """Convert optimization outputs into readable academic summaries."""

    def summarize_scenario(self, scenario_result: Dict[str, object]) -> str:
        summary = (
            f"{scenario_result['scenario_name']} under {scenario_result['farm_type']} strategy reaches "
            f"{float(scenario_result['weighted_expected_yield_t_ha']):.2f} t/ha and "
            f"{float(scenario_result['total_expected_output_tonnes']):.1f} tonnes, with fuzzy suitability "
            f"{float(scenario_result['average_fuzzy_score']):.2f} and resource balance "
            f"{float(scenario_result['resource_balance_score']):.2f}."
        )
        if bool(scenario_result.get("ml_enabled", False)):
            summary += (
                f" The supervised regression layer estimates "
                f"{float(scenario_result.get('ml_predicted_yield_t_ha', 0.0)):.2f} t/ha "
                f"and contributes to the hybrid ranking score."
            )
        return summary

    def explain_winner(
        self,
        best_result: Dict[str, object],
        second_best: Dict[str, object] | None,
        context: Dict[str, object],
    ) -> str:
        margin = 0.0
        if second_best is not None:
            margin = float(best_result["total_expected_output_tonnes"]) - float(second_best["total_expected_output_tonnes"])
        explanation = (
            f"For {context['region']} Morocco in {context['year']}, the optimizer recommends "
            f"{best_result['scenario_name']} because it combines the highest optimization score "
            f"({float(best_result['optimization_score']):.2f}), strong fuzzy suitability "
            f"({float(best_result['average_fuzzy_score']):.2f}), and expected production of "
            f"{float(best_result['total_expected_output_tonnes']):.1f} tonnes. "
            f"The margin over the next scenario is {margin:.1f} tonnes."
        )
        if bool(best_result.get("ml_enabled", False)):
            explanation += (
                f" Its supervised ML estimate is "
                f"{float(best_result.get('ml_predicted_yield_t_ha', 0.0)):.2f} t/ha "
                f"from {best_result.get('ml_model_name', 'the regression model')}."
            )
        return explanation

    def build_ranked_summaries(self, ranked_results: List[Dict[str, object]]) -> List[str]:
        return [self.summarize_scenario(item) for item in ranked_results]
