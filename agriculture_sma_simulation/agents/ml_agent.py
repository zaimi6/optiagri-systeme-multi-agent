from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split


class MLAgent:
    """Train and serve a compact supervised regression layer for crop-yield estimation."""

    TARGET_COLUMN = "base_yield_t_ha"
    NUMERIC_COLUMNS: Tuple[str, ...] = (
        "season_year",
        "precipitation_mm",
        "historical_precipitation_mm",
        "cultivated_area_kha",
        "cultivated_intensity_pct",
        "soil_quality_index",
        "water_availability_index",
        "climate_favorability_index",
        "fertilizer_efficiency_index",
        "precipitation_stress_index",
        "resource_balance_index",
        "historical_mean_yield_t_ha",
        "yield_stability",
        "trend_score",
        "base_surface_ha",
        "soil_preference",
        "water_demand",
        "climate_preference",
        "fertilizer_responsiveness",
    )
    CATEGORICAL_COLUMNS: Tuple[str, ...] = ("canonical_crop",)
    MIN_ROWS_FOR_TRAINING = 8
    ML_WEIGHT = 0.18

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_name = "untrained"
        self.feature_columns: List[str] = []
        self.numeric_fill_values = pd.Series(dtype=float)
        self.training_summary: Dict[str, object] = {
            "status": "not_trained",
            "target": self.TARGET_COLUMN,
            "model_name": self.model_name,
            "ml_weight": self.ML_WEIGHT,
            "message": "ML training has not started.",
        }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _select_model(self, row_count: int) -> Tuple[object, str]:
        if row_count >= 24:
            return (
                RandomForestRegressor(
                    n_estimators=240,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=42,
                ),
                "RandomForestRegressor",
            )
        if row_count >= 14:
            return (GradientBoostingRegressor(random_state=42), "GradientBoostingRegressor")
        return (LinearRegression(), "LinearRegression")

    def _prepare_feature_frame(self, dataframe: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = dataframe.copy()
        for column in self.NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

        if fit:
            self.numeric_fill_values = frame.loc[:, self.NUMERIC_COLUMNS].median(numeric_only=True).fillna(0.0)

        numeric = frame.loc[:, self.NUMERIC_COLUMNS].fillna(self.numeric_fill_values)
        categorical = pd.get_dummies(
            frame.loc[:, self.CATEGORICAL_COLUMNS].fillna("unknown"),
            prefix=self.CATEGORICAL_COLUMNS,
            dtype=float,
        )
        features = pd.concat([numeric, categorical], axis=1)

        if fit:
            self.feature_columns = list(features.columns)
        else:
            features = features.reindex(columns=self.feature_columns, fill_value=0.0)

        return features.astype(float)

    def _evaluate_split(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
        metrics: Dict[str, object] = {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        }
        metrics["rmse"] = round(float(np.sqrt(metrics["mse"])), 4)
        try:
            metrics["r2"] = round(float(r2_score(y_true, y_pred)), 4)
        except Exception:
            metrics["r2"] = None

        if np.all(y_true >= 0.0) and np.all(y_pred >= 0.0):
            metrics["rmsle"] = round(float(np.sqrt(mean_squared_log_error(y_true, y_pred))), 4)
            metrics["rmsle_note"] = "RMSLE computed on non-negative test predictions."
        else:
            metrics["rmsle"] = None
            metrics["rmsle_note"] = "RMSLE not available: negative values detected."
        return metrics

    def fit(self, training_data: pd.DataFrame) -> Dict[str, object]:
        dataframe = training_data.copy()
        required_columns = list(self.NUMERIC_COLUMNS) + list(self.CATEGORICAL_COLUMNS) + [self.TARGET_COLUMN]
        for column in required_columns:
            if column not in dataframe.columns:
                dataframe[column] = np.nan

        dataframe[self.TARGET_COLUMN] = pd.to_numeric(dataframe[self.TARGET_COLUMN], errors="coerce")
        dataframe = dataframe.dropna(subset=[self.TARGET_COLUMN]).copy()

        if len(dataframe) < self.MIN_ROWS_FOR_TRAINING or dataframe[self.TARGET_COLUMN].nunique() < 2:
            self.model = None
            self.model_name = "unavailable"
            self.training_summary = {
                "status": "insufficient_data",
                "target": self.TARGET_COLUMN,
                "model_name": self.model_name,
                "ml_weight": 0.0,
                "message": "Not enough distinct crop-year observations for supervised regression training.",
                "total_rows": int(len(dataframe)),
                "feature_columns": [],
                "metrics": {},
            }
            return dict(self.training_summary)

        test_size = 0.25 if len(dataframe) >= 16 else 0.33
        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=42, shuffle=True)
        if train_df.empty or test_df.empty:
            self.model = None
            self.model_name = "unavailable"
            self.training_summary = {
                "status": "split_failed",
                "target": self.TARGET_COLUMN,
                "model_name": self.model_name,
                "ml_weight": 0.0,
                "message": "Train/test split did not produce usable partitions.",
                "total_rows": int(len(dataframe)),
                "feature_columns": [],
                "metrics": {},
            }
            return dict(self.training_summary)

        x_train = self._prepare_feature_frame(train_df, fit=True)
        x_test = self._prepare_feature_frame(test_df, fit=False)
        y_train = train_df[self.TARGET_COLUMN].astype(float).to_numpy()
        y_test = test_df[self.TARGET_COLUMN].astype(float).to_numpy()

        self.model, self.model_name = self._select_model(len(dataframe))
        self.model.fit(x_train, y_train)

        train_predictions = np.asarray(self.model.predict(x_train), dtype=float)
        test_predictions = np.asarray(self.model.predict(x_test), dtype=float)

        self.training_summary = {
            "status": "trained",
            "target": self.TARGET_COLUMN,
            "model_name": self.model_name,
            "ml_weight": self.ML_WEIGHT,
            "message": "Supervised regression layer trained successfully.",
            "total_rows": int(len(dataframe)),
            "training_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_columns": list(self.feature_columns),
            "metrics": {
                "train": self._evaluate_split(y_train, train_predictions),
                "test": self._evaluate_split(y_test, test_predictions),
            },
        }
        return dict(self.training_summary)

    def is_ready(self) -> bool:
        return self.model is not None and bool(self.feature_columns)

    def predict_crop_yield(self, crop_profile: Dict[str, object], context: Dict[str, object]) -> Dict[str, object]:
        baseline_yield = self._safe_float(crop_profile.get("base_yield_t_ha"), 0.0)
        historical_mean = self._safe_float(crop_profile.get("historical_mean_yield_t_ha"), baseline_yield)

        if not self.is_ready():
            return {
                "crop": str(crop_profile.get("canonical_crop", "crop")),
                "ml_enabled": False,
                "model_name": self.model_name,
                "ml_predicted_yield_t_ha": baseline_yield,
                "raw_ml_prediction_t_ha": baseline_yield,
                "note": "ML layer unavailable; prediction falls back to the current crop baseline without affecting ranking weight.",
            }

        row = pd.DataFrame(
            [
                {
                    "canonical_crop": str(crop_profile.get("canonical_crop", "crop")),
                    "season_year": int(context.get("year", crop_profile.get("reference_year", 0)) or 0),
                    "precipitation_mm": self._safe_float(context.get("precipitation_mm")),
                    "historical_precipitation_mm": self._safe_float(context.get("historical_precipitation_mm")),
                    "cultivated_area_kha": self._safe_float(context.get("cultivated_area_kha")),
                    "cultivated_intensity_pct": self._safe_float(context.get("cultivated_intensity_pct")),
                    "soil_quality_index": self._safe_float(context.get("soil_quality_index")),
                    "water_availability_index": self._safe_float(context.get("water_availability_index")),
                    "climate_favorability_index": self._safe_float(context.get("climate_favorability_index")),
                    "fertilizer_efficiency_index": self._safe_float(context.get("fertilizer_efficiency_index")),
                    "precipitation_stress_index": self._safe_float(context.get("precipitation_stress_index")),
                    "resource_balance_index": self._safe_float(context.get("resource_balance_index")),
                    "historical_mean_yield_t_ha": historical_mean,
                    "yield_stability": self._safe_float(crop_profile.get("yield_stability"), 0.80),
                    "trend_score": self._safe_float(crop_profile.get("trend_score"), 1.0),
                    "base_surface_ha": self._safe_float(crop_profile.get("base_surface_ha")),
                    "soil_preference": self._safe_float(crop_profile.get("soil_preference")),
                    "water_demand": self._safe_float(crop_profile.get("water_demand")),
                    "climate_preference": self._safe_float(crop_profile.get("climate_preference")),
                    "fertilizer_responsiveness": self._safe_float(crop_profile.get("fertilizer_responsiveness")),
                }
            ]
        )

        features = self._prepare_feature_frame(row, fit=False)
        raw_prediction = float(self.model.predict(features)[0])
        upper_cap = max(
            baseline_yield * 1.45,
            historical_mean * 1.25,
            self._safe_float(context.get("yield_scale_cap_t_ha"), baseline_yield) * 1.10,
            0.5,
        )
        clipped_prediction = self._clamp(raw_prediction, 0.0, upper_cap)
        return {
            "crop": str(crop_profile.get("canonical_crop", "crop")),
            "ml_enabled": True,
            "model_name": self.model_name,
            "ml_predicted_yield_t_ha": round(clipped_prediction, 4),
            "raw_ml_prediction_t_ha": round(raw_prediction, 4),
            "note": f"ML predicts {clipped_prediction:.2f} t/ha for {crop_profile.get('canonical_crop', 'crop')}.",
        }

    def evaluate_scenario(
        self,
        allocation: Dict[str, float],
        crop_profiles: Dict[str, Dict[str, object]],
        context: Dict[str, object],
    ) -> Dict[str, object]:
        decision_surface_ha = self._safe_float(context.get("decision_surface_ha"), 0.0)
        crop_predictions: List[Dict[str, object]] = []
        for crop, share in allocation.items():
            prediction = self.predict_crop_yield(crop_profiles[crop], context)
            allocated_surface = decision_surface_ha * float(share)
            predicted_output = allocated_surface * float(prediction["ml_predicted_yield_t_ha"])
            crop_predictions.append(
                {
                    **prediction,
                    "allocation_share": round(float(share), 4),
                    "allocated_surface_ha": round(allocated_surface, 4),
                    "ml_predicted_output_tonnes": round(predicted_output, 4),
                }
            )

        weighted_yield = float(
            sum(float(item["ml_predicted_yield_t_ha"]) * float(item["allocation_share"]) for item in crop_predictions)
        )
        total_output = float(sum(float(item["ml_predicted_output_tonnes"]) for item in crop_predictions))
        return {
            "ml_enabled": self.is_ready(),
            "ml_model_name": self.model_name,
            "ml_target": self.TARGET_COLUMN,
            "ml_weight": self.training_summary.get("ml_weight", 0.0),
            "ml_predicted_yield_t_ha": round(weighted_yield, 4),
            "ml_total_output_tonnes": round(total_output, 4),
            "ml_crop_predictions": crop_predictions,
        }
