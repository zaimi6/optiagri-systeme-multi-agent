from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .climate_agent import ClimateAgent
from .crop_agent import CropAgent
from .data_agent import DataAgent
from .decision_agent import DecisionAgent
from .farm_type_agent import FarmTypeAgent
from .fertilizer_agent import FertilizerAgent
from .fuzzy_logic_agent import FuzzyLogicAgent
from .ml_agent import MLAgent
from .optimization_agent import OptimizationAgent
from .soil_agent import SoilAgent
from .visualization_agent import VisualizationAgent
from .water_agent import WaterAgent
from .yield_agent import YieldAgent


class CoordinatorAgent:
    """Orchestrate the full agricultural simulation workflow."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.data_agent = DataAgent(project_root=self.project_root)
        self.soil_agent = SoilAgent()
        self.water_agent = WaterAgent()
        self.climate_agent = ClimateAgent()
        self.crop_agent = CropAgent()
        self.fertilizer_agent = FertilizerAgent()
        self.farm_type_agent = FarmTypeAgent()
        self.fuzzy_logic_agent = FuzzyLogicAgent()
        self.ml_agent = MLAgent(results_dir=self.results_dir)
        self.yield_agent = YieldAgent()
        self.optimization_agent = OptimizationAgent()
        self.decision_agent = DecisionAgent()
        self.visualization_agent = VisualizationAgent(results_dir=self.results_dir)

        self.scenario_results_path = self.results_dir / "scenario_results.csv"
        self.ml_evaluation_path = self.results_dir / "ml_evaluation.json"
        self.report_path = self.results_dir / "best_allocation_report.txt"
        self.summary_path = self.results_dir / "simulation_summary.json"
        self.ml_overview: Dict[str, object] = {}

    @staticmethod
    def _to_profile_map(dataframe: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        return {str(row["canonical_crop"]): row.to_dict() for _, row in dataframe.iterrows()}

    @staticmethod
    def _safe_metric(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_metrics(
        self,
        context: Dict[str, object],
        scenario_index: int,
        scenario_total: int,
        *,
        expected_yield_t_ha: float = 0.0,
        total_output_tonnes: float = 0.0,
        farm_type: str = "pending",
        extra: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        metrics = {
            "expected_yield_t_ha": expected_yield_t_ha,
            "total_output_tonnes": total_output_tonnes,
            "farm_type": farm_type,
            "decision_surface_ha": context.get("decision_surface_ha", 0.0),
            "scenario_index": scenario_index,
            "scenario_total": scenario_total,
            "best_crop": "pending",
            "best_allocation": "pending",
            "fuzzy_suitability_score": 0.0,
            "resource_balance_score": max(
                0.0,
                min(1.0, (float(context.get("resource_balance_index", 0.4)) - 0.40) / 0.60),
            ),
            "optimization_score": 0.0,
            "agent_score": 0.0,
            "ml_predicted_yield_t_ha": 0.0,
            "ml_prediction_score": 0.0,
            "ml_model_name": "pending",
            "ml_test_mae": 0.0,
            "ml_test_rmse": 0.0,
            "ml_test_r2": 0.0,
            "ml_rmsle_display": "pending",
            "dcg_score": 0.0,
            "best_scenario_score": 0.0,
            "score_margin": 0.0,
            "water_stress": 0.0,
            "dominant_fuzzy_rule": "pending",
            "why_selected": "Scenario ranking will appear after optimization.",
        }
        if self.ml_overview:
            metrics.update(self.ml_overview)
        if extra:
            metrics.update(extra)
        return metrics

    def _ml_metrics_overview(self, ml_summary: Dict[str, object]) -> Dict[str, object]:
        test_metrics = dict(ml_summary.get("metrics", {}).get("test", {}))
        rmsle_value = test_metrics.get("rmsle")
        if isinstance(rmsle_value, (int, float)):
            rmsle_display = f"{float(rmsle_value):.4f}"
        else:
            rmsle_display = str(test_metrics.get("rmsle_note", "n/a"))
        return {
            "ml_model_name": str(ml_summary.get("model_name", "unavailable")),
            "ml_test_mae": self._safe_metric(test_metrics.get("mae")),
            "ml_test_rmse": self._safe_metric(test_metrics.get("rmse")),
            "ml_test_r2": self._safe_metric(test_metrics.get("r2")),
            "ml_rmsle_display": rmsle_display,
        }

    @staticmethod
    def _active_agents_for_phase(phase: str) -> List[str]:
        mapping = {
            "initialization": ["Data", "Coordinator"],
            "scenario_setup": ["Coordinator", "Crop"],
            "soil_assessment": ["Soil"],
            "water_assessment": ["Water"],
            "climate_assessment": ["Climate"],
            "farm_assessment": ["FarmType"],
            "fertilizer_assessment": ["Fertilizer"],
            "crop_assessment": ["Crop"],
            "fuzzy_assessment": ["Fuzzy"],
            "yield_assessment": ["Yield"],
            "optimization": ["Optimizer", "Coordinator"],
            "recommendation": ["Optimizer", "Coordinator"],
        }
        return mapping.get(phase, ["Coordinator"])

    def _save_report(
        self,
        context: Dict[str, object],
        best_result: Dict[str, object],
        ranked_results: List[Dict[str, object]],
        data_bundle: Dict[str, object],
        ml_summary: Dict[str, object],
        final_explanation: str,
    ) -> None:
        lines = [
            "Multi-Agent System for Moroccan Agricultural Optimization",
            "",
            f"Data mode: {context['data_mode']}",
            f"Context: {context['region']} Morocco, {context['year']}",
            f"Precipitation: {context['precipitation_mm']:.1f} mm",
            f"Decision surface: {context['decision_surface_ha']:.0f} ha",
            "",
            "Best Scenario",
            f"- Scenario: {best_result.get('scenario_name', 'n/a')}",
            f"- Allocation: {best_result.get('allocation_text', 'n/a')}",
            f"- Focus crop: {best_result.get('focus_crop', 'n/a')}",
            f"- Farm type: {best_result.get('farm_type', 'n/a')}",
            f"- Weighted expected yield: {self._safe_metric(best_result.get('weighted_expected_yield_t_ha')):.2f} t/ha",
            f"- Total expected output: {self._safe_metric(best_result.get('total_expected_output_tonnes')):.1f} tonnes",
            f"- Fuzzy suitability score: {self._safe_metric(best_result.get('average_fuzzy_score')):.2f}",
            f"- Resource balance score: {self._safe_metric(best_result.get('resource_balance_score')):.2f}",
            f"- Optimization score: {self._safe_metric(best_result.get('optimization_score')):.2f}",
            f"- ML predicted yield: {self._safe_metric(best_result.get('ml_predicted_yield_t_ha')):.2f} t/ha",
            f"- Why it won: {best_result.get('win_reason', 'n/a')}",
            "",
            "Supervised ML Layer",
            f"- Status: {ml_summary.get('status', 'unknown')}",
            f"- Message: {ml_summary.get('message', 'n/a')}",
            f"- Target: {ml_summary.get('target', 'base_yield_t_ha')}",
            f"- Model: {ml_summary.get('model_name', 'n/a')}",
            f"- Total rows: {ml_summary.get('total_rows', 0)}",
            f"- Train rows: {ml_summary.get('training_rows', 0)}",
            f"- Test rows: {ml_summary.get('test_rows', 0)}",
            f"- Test MAE: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('mae')):.4f}",
            f"- Test MSE: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('mse')):.4f}",
            f"- Test RMSE: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('rmse')):.4f}",
            f"- Test R^2: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('r2')):.4f}",
            (
                f"- Test RMSLE: {float(ml_summary['metrics']['test']['rmsle']):.4f}"
                if isinstance(ml_summary.get("metrics", {}).get("test", {}).get("rmsle"), (int, float))
                else f"- Test RMSLE: {ml_summary.get('metrics', {}).get('test', {}).get('rmsle_note', 'n/a')}"
            ),
            "",
            "Final Explanation",
            final_explanation,
            "",
            "Scenario Ranking",
        ]
        for index, item in enumerate(ranked_results, start=1):
            lines.append(
                f"{index}. {item['scenario_name']} | {item['farm_type']} | "
                f"score {float(item['optimization_score']):.2f} | "
                f"{float(item['weighted_expected_yield_t_ha']):.2f} t/ha | "
                f"{float(item['total_expected_output_tonnes']):.1f} tonnes"
            )

        lines.append("")
        lines.append("Official Files Loaded")
        if data_bundle["used_files"]:
            lines.extend(f"- {name}" for name in data_bundle["used_files"])
        else:
            lines.append("- Fallback demo sample")

        lines.append("")
        lines.append("Real Official Columns Used")
        lines.extend(f"- {name}" for name in data_bundle.get("real_columns", []))
        lines.append("")
        lines.append("Derived Columns Used")
        lines.extend(f"- {name}" for name in data_bundle.get("derived_columns", []))
        lines.append("")
        lines.append("Derived Variables")
        for variable in data_bundle["derived_variables"]:
            lines.append(f"- {variable['name']}: {variable['reason']}")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def _save_summary(
        self,
        context: Dict[str, object],
        best_result: Dict[str, object],
        ranked_results: List[Dict[str, object]],
        data_bundle: Dict[str, object],
        ml_summary: Dict[str, object],
        visualization_outputs: Dict[str, Path],
        final_explanation: str,
    ) -> None:
        summary = {
            "project_title": "Multi-Agent System for Moroccan agricultural optimization using real official Moroccan data and a live simulation interface",
            "data_mode": context["data_mode"],
            "context": context,
            "best_result": best_result,
            "ranked_results": ranked_results,
            "used_official_files": data_bundle["used_files"],
            "real_columns": data_bundle.get("real_columns", []),
            "derived_columns": data_bundle.get("derived_columns", []),
            "derived_variables": data_bundle["derived_variables"],
            "data_load_log": data_bundle["load_log"],
            "ml_summary": ml_summary,
            "final_explanation": final_explanation,
            "saved_outputs": {
                "merged_agriculture_dataset": str(self.data_agent.merged_csv_path),
                "scenario_results": str(self.scenario_results_path),
                "ml_evaluation": str(self.ml_evaluation_path),
                "best_allocation_report": str(self.report_path),
                "simulation_summary": str(self.summary_path),
                "agents_simulation_png": str(visualization_outputs["png"]),
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _update_visualization(
        self,
        keep_running: bool,
        *,
        phase: str,
        context: Dict[str, object],
        scenario_name: str,
        allocation_text: str,
        narrative: str,
        raw_scenario_results: List[Dict[str, object]],
        scenario_index: int,
        scenario_total: int,
        expected_yield_t_ha: float = 0.0,
        total_output_tonnes: float = 0.0,
        farm_type: str = "pending",
        extra_metrics: Dict[str, object] | None = None,
    ) -> bool:
        if not keep_running or not self.visualization_agent.is_running():
            return False

        ranked_view = self.optimization_agent.rank_scenarios(raw_scenario_results)
        best_result = self.optimization_agent.select_best(ranked_view)
        decision_metrics = self.optimization_agent.compute_decision_metrics(ranked_view)
        return self.visualization_agent.update(
            phase=phase,
            scenario_name=scenario_name,
            allocation_text=allocation_text,
            narrative=narrative,
            metrics=self._build_metrics(
                context,
                scenario_index,
                scenario_total,
                expected_yield_t_ha=expected_yield_t_ha,
                total_output_tonnes=total_output_tonnes,
                farm_type=farm_type,
                extra={**decision_metrics, **(extra_metrics or {})},
            ),
            scenario_results=ranked_view,
            best_result=best_result,
            active_agents=self._active_agents_for_phase(phase),
        )

    def _serialize_results(self, ranked_results: List[Dict[str, object]]) -> pd.DataFrame:
        serialized_rows = []
        for item in ranked_results:
            row = {}
            for key, value in item.items():
                if key in {"allocation", "crop_details"}:
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            serialized_rows.append(row)
        return pd.DataFrame(serialized_rows)

    def run(self) -> None:
        data_bundle = self.data_agent.load_agricultural_data()
        crop_profiles_df: pd.DataFrame = data_bundle["merged_dataset"]
        context: Dict[str, object] = data_bundle["context"]
        crop_profiles = self._to_profile_map(crop_profiles_df)
        ml_dataset = pd.DataFrame(data_bundle.get("ml_dataset", pd.DataFrame()))
        ml_summary = self.ml_agent.fit(ml_dataset)
        self.ml_overview = self._ml_metrics_overview(ml_summary)
        self.ml_evaluation_path.write_text(json.dumps(ml_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        scenarios = self.optimization_agent.generate_scenarios(
            crop_profiles=crop_profiles,
            decision_surface_ha=float(context["decision_surface_ha"]),
        )
        if not scenarios:
            raise RuntimeError("No valid agricultural scenarios could be generated from the available crops.")

        self.visualization_agent.start(context)
        scenario_results: List[Dict[str, object]] = []

        keep_running = self.visualization_agent.update(
            phase="initialization",
            scenario_name="Official Data Load",
            allocation_text="Loading local Moroccan production and sector indicator workbooks.",
            narrative=(
                "DataAgent cleaned the official files and prepared a normalized agricultural dataset, "
                "then MLAgent trained the supervised yield regressor used by the hybrid ranking."
            ),
            metrics=self._build_metrics(context, 0, len(scenarios)),
            scenario_results=[],
            best_result=None,
            active_agents=self._active_agents_for_phase("initialization"),
        )

        for scenario_index, scenario in enumerate(scenarios, start=1):
            if not keep_running or not self.visualization_agent.is_running():
                break

            allocation = dict(scenario["allocation"])
            scenario_name = str(scenario["name"])
            allocation_text = self.crop_agent.describe_scenario(scenario)

            keep_running = self._update_visualization(
                keep_running,
                phase="scenario_setup",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="CoordinatorAgent activates the crop allocation while CropAgent prepares crop-specific evaluation.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                extra_metrics={
                    "best_allocation": allocation_text,
                    "best_crop": self.crop_agent.focus_crop(allocation),
                },
            )
            if not keep_running:
                break

            soil_summaries = {crop: self.soil_agent.assess_crop(crop_profiles[crop], context) for crop in allocation}
            keep_running = self._update_visualization(
                keep_running,
                phase="soil_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="SoilAgent converts the Moroccan context and crop preferences into normalized soil suitability scores.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                extra_metrics={
                    "best_crop": self.crop_agent.focus_crop(allocation),
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                },
            )
            if not keep_running:
                break

            water_summaries = {
                crop: self.water_agent.assess_crop(crop_profiles[crop], context, allocation_share=share)
                for crop, share in allocation.items()
            }
            keep_running = self._update_visualization(
                keep_running,
                phase="water_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="WaterAgent evaluates irrigation sufficiency and penalizes water-intensive allocations under limited rainfall.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                extra_metrics={
                    "best_crop": self.crop_agent.focus_crop(allocation),
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                    "average_water_score": sum(item["normalized_score"] * allocation[crop] for crop, item in water_summaries.items()),
                    "water_stress": sum(item["stress"] * allocation[crop] for crop, item in water_summaries.items()),
                },
            )
            if not keep_running:
                break

            climate_summaries = {crop: self.climate_agent.assess_crop(crop_profiles[crop], context) for crop in allocation}
            keep_running = self._update_visualization(
                keep_running,
                phase="climate_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="ClimateAgent translates precipitation conditions and crop trends into normalized climate favorability.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                extra_metrics={
                    "best_crop": self.crop_agent.focus_crop(allocation),
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                    "average_water_score": sum(item["normalized_score"] * allocation[crop] for crop, item in water_summaries.items()),
                    "average_climate_score": sum(item["normalized_score"] * allocation[crop] for crop, item in climate_summaries.items()),
                    "water_stress": sum(item["stress"] * allocation[crop] for crop, item in water_summaries.items()),
                },
            )
            if not keep_running:
                break

            farm_summary = self.farm_type_agent.recommend(allocation, crop_profiles, context)
            keep_running = self._update_visualization(
                keep_running,
                phase="farm_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative=f"FarmTypeAgent selects the '{farm_summary['strategy']}' strategy to match crop demand and resource balance.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                farm_type=str(farm_summary["strategy"]),
                extra_metrics={
                    "best_crop": self.crop_agent.focus_crop(allocation),
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                    "average_water_score": sum(item["normalized_score"] * allocation[crop] for crop, item in water_summaries.items()),
                    "average_climate_score": sum(item["normalized_score"] * allocation[crop] for crop, item in climate_summaries.items()),
                    "resource_balance_score": float(farm_summary["normalized_score"]),
                    "water_stress": sum(item["stress"] * allocation[crop] for crop, item in water_summaries.items()),
                },
            )
            if not keep_running:
                break

            fertilizer_summaries = {
                crop: self.fertilizer_agent.assess_crop(crop_profiles[crop], context, water_summaries[crop], farm_summary)
                for crop in allocation
            }
            keep_running = self._update_visualization(
                keep_running,
                phase="fertilizer_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="FertilizerAgent applies derived fertilizer efficiency logic while keeping it explicitly labeled as derived.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                farm_type=str(farm_summary["strategy"]),
                extra_metrics={
                    "best_crop": self.crop_agent.focus_crop(allocation),
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                    "average_water_score": sum(item["normalized_score"] * allocation[crop] for crop, item in water_summaries.items()),
                    "average_climate_score": sum(item["normalized_score"] * allocation[crop] for crop, item in climate_summaries.items()),
                    "average_fertilizer_score": sum(item["normalized_score"] * allocation[crop] for crop, item in fertilizer_summaries.items()),
                    "resource_balance_score": float(farm_summary["normalized_score"]),
                    "water_stress": sum(item["stress"] * allocation[crop] for crop, item in water_summaries.items()),
                },
            )
            if not keep_running:
                break

            compatibility_summaries = {
                crop: self.crop_agent.assess_compatibility(
                    crop_profiles[crop],
                    soil_summaries[crop],
                    water_summaries[crop],
                    climate_summaries[crop],
                    farm_summary,
                )
                for crop in allocation
            }
            keep_running = self._update_visualization(
                keep_running,
                phase="crop_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="CropAgent evaluates how well each crop fits the current soil, water, climate, and farm strategy combination.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                farm_type=str(farm_summary["strategy"]),
                extra_metrics={
                    "best_crop": self.crop_agent.focus_crop(allocation),
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                    "average_water_score": sum(item["normalized_score"] * allocation[crop] for crop, item in water_summaries.items()),
                    "average_climate_score": sum(item["normalized_score"] * allocation[crop] for crop, item in climate_summaries.items()),
                    "average_fertilizer_score": sum(item["normalized_score"] * allocation[crop] for crop, item in fertilizer_summaries.items()),
                    "average_compatibility_score": sum(item["normalized_score"] * allocation[crop] for crop, item in compatibility_summaries.items()),
                    "resource_balance_score": float(farm_summary["normalized_score"]),
                    "water_stress": sum(item["stress"] * allocation[crop] for crop, item in water_summaries.items()),
                },
            )
            if not keep_running:
                break

            fuzzy_summaries = {
                crop: self.fuzzy_logic_agent.evaluate(
                    crop_profiles[crop],
                    soil_summaries[crop],
                    water_summaries[crop],
                    climate_summaries[crop],
                    compatibility_summaries[crop],
                    fertilizer_summaries[crop],
                    farm_summary,
                )
                for crop in allocation
            }
            weighted_fuzzy_score = sum(
                float(item["normalized_score"]) * allocation[crop] for crop, item in fuzzy_summaries.items()
            )
            dominant_fuzzy_crop = max(
                allocation,
                key=lambda crop: float(fuzzy_summaries[crop]["normalized_score"]),
            )
            keep_running = self._update_visualization(
                keep_running,
                phase="fuzzy_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative=(
                    f"FuzzyLogicAgent softens rigid thresholds. Dominant fuzzy rule for "
                    f"{dominant_fuzzy_crop} is '{fuzzy_summaries[dominant_fuzzy_crop]['dominant_rule']}'."
                ),
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                farm_type=str(farm_summary["strategy"]),
                extra_metrics={
                    "best_crop": dominant_fuzzy_crop,
                    "average_soil_score": sum(item["normalized_score"] * allocation[crop] for crop, item in soil_summaries.items()),
                    "average_water_score": sum(item["normalized_score"] * allocation[crop] for crop, item in water_summaries.items()),
                    "average_climate_score": sum(item["normalized_score"] * allocation[crop] for crop, item in climate_summaries.items()),
                    "average_fertilizer_score": sum(item["normalized_score"] * allocation[crop] for crop, item in fertilizer_summaries.items()),
                    "average_compatibility_score": sum(item["normalized_score"] * allocation[crop] for crop, item in compatibility_summaries.items()),
                    "fuzzy_suitability_score": weighted_fuzzy_score,
                    "resource_balance_score": sum(
                        float(item["resource_balance_score"]) * allocation[crop] for crop, item in fuzzy_summaries.items()
                    ),
                    "dominant_fuzzy_rule": str(fuzzy_summaries[dominant_fuzzy_crop]["dominant_rule"]),
                    "water_stress": sum(item["stress"] * allocation[crop] for crop, item in water_summaries.items()),
                },
            )
            if not keep_running:
                break

            ml_scenario = self.ml_agent.evaluate_scenario(allocation, crop_profiles, context)
            ml_prediction_map = {
                str(item["crop"]): item for item in ml_scenario.get("ml_crop_predictions", [])
            }
            crop_details: List[Dict[str, object]] = []
            for crop, share in allocation.items():
                yield_output = self.yield_agent.estimate_crop_output(
                    crop_profile=crop_profiles[crop],
                    context=context,
                    allocation_share=share,
                    decision_surface_ha=float(context["decision_surface_ha"]),
                    soil_summary=soil_summaries[crop],
                    water_summary=water_summaries[crop],
                    climate_summary=climate_summaries[crop],
                    fertilizer_summary=fertilizer_summaries[crop],
                    farm_summary=farm_summary,
                    compatibility_summary=compatibility_summaries[crop],
                    fuzzy_summary=fuzzy_summaries[crop],
                )
                ml_prediction = ml_prediction_map.get(
                    crop,
                    {
                        "ml_predicted_yield_t_ha": float(crop_profiles[crop].get("base_yield_t_ha", 0.0)),
                        "ml_predicted_output_tonnes": 0.0,
                    },
                )
                crop_details.append(
                    {
                        "crop": crop,
                        "allocation_share": round(share, 4),
                        "soil_score": soil_summaries[crop]["normalized_score"],
                        "water_score": water_summaries[crop]["normalized_score"],
                        "climate_score": climate_summaries[crop]["normalized_score"],
                        "fertilizer_score": fertilizer_summaries[crop]["normalized_score"],
                        "compatibility_score": compatibility_summaries[crop]["normalized_score"],
                        "fuzzy_score": fuzzy_summaries[crop]["normalized_score"],
                        "resource_balance_score": fuzzy_summaries[crop]["resource_balance_score"],
                        "water_stress": water_summaries[crop]["stress"],
                        "dominant_fuzzy_rule": fuzzy_summaries[crop]["dominant_rule"],
                        "ml_predicted_yield_t_ha": ml_prediction["ml_predicted_yield_t_ha"],
                        "ml_predicted_output_tonnes": ml_prediction["ml_predicted_output_tonnes"],
                        **yield_output,
                    }
                )

            aggregated = self.yield_agent.aggregate_scenario(
                crop_details=crop_details,
                farm_summary=farm_summary,
                decision_surface_ha=float(context["decision_surface_ha"]),
            )
            aggregated.update(
                {
                    "ml_enabled": bool(ml_scenario.get("ml_enabled", False)),
                    "ml_model_name": str(ml_scenario.get("ml_model_name", "unavailable")),
                    "ml_predicted_yield_t_ha": float(
                        ml_scenario.get("ml_predicted_yield_t_ha", aggregated["weighted_expected_yield_t_ha"])
                    ),
                    "ml_total_output_tonnes": float(
                        ml_scenario.get("ml_total_output_tonnes", aggregated["total_expected_output_tonnes"])
                    ),
                }
            )
            keep_running = self._update_visualization(
                keep_running,
                phase="yield_assessment",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative="YieldAgent combines normalized agent outputs, fuzzy suitability, and bounded scaling into expected yield and production.",
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                expected_yield_t_ha=float(aggregated["weighted_expected_yield_t_ha"]),
                total_output_tonnes=float(aggregated["total_expected_output_tonnes"]),
                farm_type=str(farm_summary["strategy"]),
                extra_metrics={
                    "best_crop": str(aggregated["best_crop"]),
                    "average_soil_score": aggregated["average_soil_score"],
                    "average_water_score": aggregated["average_water_score"],
                    "average_climate_score": aggregated["average_climate_score"],
                    "average_fertilizer_score": aggregated["average_fertilizer_score"],
                    "average_compatibility_score": aggregated["average_compatibility_score"],
                    "fuzzy_suitability_score": aggregated["average_fuzzy_score"],
                    "resource_balance_score": aggregated["resource_balance_score"],
                    "ml_predicted_yield_t_ha": aggregated["ml_predicted_yield_t_ha"],
                    "water_stress": aggregated["average_water_stress"],
                },
            )
            if not keep_running:
                break

            scenario_record = self.optimization_agent.build_scenario_record(scenario, aggregated)
            scenario_record["summary"] = self.decision_agent.summarize_scenario(scenario_record)
            scenario_results.append(scenario_record)

            ranked_partial = self.optimization_agent.rank_scenarios(scenario_results)
            current_best = self.optimization_agent.select_best(ranked_partial)
            keep_running = self._update_visualization(
                keep_running,
                phase="optimization",
                context=context,
                scenario_name=scenario_name,
                allocation_text=allocation_text,
                narrative=current_best.get("win_reason", "OptimizationAgent is updating the ranking."),
                raw_scenario_results=scenario_results,
                scenario_index=scenario_index,
                scenario_total=len(scenarios),
                expected_yield_t_ha=float(scenario_record["weighted_expected_yield_t_ha"]),
                total_output_tonnes=float(scenario_record["total_expected_output_tonnes"]),
                farm_type=str(scenario_record["farm_type"]),
                extra_metrics={
                    "best_crop": str(current_best.get("focus_crop", scenario_record["focus_crop"])),
                    "best_allocation": str(current_best.get("allocation_text", allocation_text)),
                    "average_soil_score": scenario_record["average_soil_score"],
                    "average_water_score": scenario_record["average_water_score"],
                    "average_climate_score": scenario_record["average_climate_score"],
                    "average_fertilizer_score": scenario_record["average_fertilizer_score"],
                    "average_compatibility_score": scenario_record["average_compatibility_score"],
                    "fuzzy_suitability_score": scenario_record["average_fuzzy_score"],
                    "resource_balance_score": scenario_record["resource_balance_score"],
                    "ml_predicted_yield_t_ha": scenario_record["ml_predicted_yield_t_ha"],
                    "ml_prediction_score": current_best.get("ml_prediction_score", 0.0),
                    "agent_score": current_best.get("agent_score", 0.0),
                    "optimization_score": current_best.get("optimization_score", 0.0),
                    "why_selected": current_best.get("win_reason", "Optimization ranking in progress."),
                    "water_stress": scenario_record["average_water_stress"],
                },
            )

        ranked_results = self.optimization_agent.rank_scenarios(scenario_results)
        best_result = self.optimization_agent.select_best(ranked_results)
        second_best = ranked_results[1] if len(ranked_results) > 1 else None

        if ranked_results:
            final_explanation = self.decision_agent.explain_winner(best_result, second_best, context)
        else:
            best_result = {
                "scenario_name": "No completed scenario",
                "allocation_text": "Simulation interrupted before scenario completion",
                "farm_type": "n/a",
                "weighted_expected_yield_t_ha": 0.0,
                "total_expected_output_tonnes": 0.0,
                "average_fuzzy_score": 0.0,
                "resource_balance_score": 0.0,
                "optimization_score": 0.0,
                "win_reason": "Simulation interrupted before optimization finished.",
            }
            final_explanation = "Simulation interrupted before any scenario was completed."

        if self.visualization_agent.is_running():
            self._update_visualization(
                True,
                phase="recommendation",
                context=context,
                scenario_name=str(best_result.get("scenario_name", "No completed scenario")),
                allocation_text=str(best_result.get("allocation_text", "n/a")),
                narrative=final_explanation,
                raw_scenario_results=ranked_results,
                scenario_index=len(ranked_results),
                scenario_total=len(scenarios),
                expected_yield_t_ha=self._safe_metric(best_result.get("weighted_expected_yield_t_ha")),
                total_output_tonnes=self._safe_metric(best_result.get("total_expected_output_tonnes")),
                farm_type=str(best_result.get("farm_type", "n/a")),
                extra_metrics={
                    "best_crop": str(best_result.get("focus_crop", "n/a")),
                    "best_allocation": str(best_result.get("allocation_text", "n/a")),
                    "fuzzy_suitability_score": self._safe_metric(best_result.get("average_fuzzy_score")),
                    "resource_balance_score": self._safe_metric(best_result.get("resource_balance_score")),
                    "ml_predicted_yield_t_ha": self._safe_metric(best_result.get("ml_predicted_yield_t_ha")),
                    "ml_prediction_score": self._safe_metric(best_result.get("ml_prediction_score")),
                    "agent_score": self._safe_metric(best_result.get("agent_score")),
                    "optimization_score": self._safe_metric(best_result.get("optimization_score")),
                    "why_selected": str(best_result.get("win_reason", final_explanation)),
                },
            )

        scenario_results_df = self._serialize_results(ranked_results)
        scenario_results_df.to_csv(self.scenario_results_path, index=False)
        self._save_report(context, best_result, ranked_results, data_bundle, ml_summary, final_explanation)
        visualization_outputs = self.visualization_agent.finalize()
        self._save_summary(context, best_result, ranked_results, data_bundle, ml_summary, visualization_outputs, final_explanation)

        print("Moroccan Agricultural Optimization Simulation")
        print(f"Data mode: {context['data_mode']}")
        print(f"Context: {context['region']} Morocco, {context['year']}")
        print(
            "Official files loaded: "
            f"{', '.join(data_bundle['used_files']) if data_bundle['used_files'] else 'fallback demo sample'}"
        )
        if ranked_results:
            print(f"Best scenario: {best_result['scenario_name']}")
            print(f"Best allocation: {best_result['allocation_text']}")
            print(f"Farm type: {best_result['farm_type']}")
            print(f"Weighted expected yield: {float(best_result['weighted_expected_yield_t_ha']):.2f} t/ha")
            print(f"ML predicted yield: {float(best_result.get('ml_predicted_yield_t_ha', 0.0)):.2f} t/ha")
            print(f"Total expected output: {float(best_result['total_expected_output_tonnes']):.1f} tonnes")
            print(f"Fuzzy suitability score: {float(best_result['average_fuzzy_score']):.2f}")
            print(f"Optimization score: {float(best_result['optimization_score']):.2f}")
            if ml_summary.get("status") == "trained":
                print(
                    f"ML model: {ml_summary.get('model_name')} | "
                    f"Test MAE: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('mae')):.4f} | "
                    f"Test RMSE: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('rmse')):.4f} | "
                    f"Test R^2: {self._safe_metric(ml_summary.get('metrics', {}).get('test', {}).get('r2')):.4f}"
                )
            else:
                print(f"ML layer status: {ml_summary.get('status')} ({ml_summary.get('message')})")
        else:
            print("Best scenario: simulation interrupted before completion")
        print(f"Results folder: {self.results_dir}")
