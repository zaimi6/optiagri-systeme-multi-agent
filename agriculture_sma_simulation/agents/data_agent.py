from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataAgent:
    """Load and prepare official Moroccan agricultural data for simulation."""

    CROP_LIBRARY = {
        "Wheat": {
            "patterns": ("ble", "wheat"),
            "soil_preference": 0.90,
            "water_demand": 0.82,
            "climate_preference": 0.88,
            "fertilizer_responsiveness": 0.94,
        },
        "Oranges": {
            "patterns": ("orange",),
            "soil_preference": 0.96,
            "water_demand": 1.02,
            "climate_preference": 0.94,
            "fertilizer_responsiveness": 0.91,
        },
        "Tomatoes": {
            "patterns": ("tomate",),
            "soil_preference": 0.93,
            "water_demand": 0.98,
            "climate_preference": 0.92,
            "fertilizer_responsiveness": 1.02,
        },
        "Olives": {
            "patterns": ("olive", "olivier"),
            "soil_preference": 0.95,
            "water_demand": 0.74,
            "climate_preference": 0.96,
            "fertilizer_responsiveness": 0.84,
        },
        "Barley": {
            "patterns": ("orge", "barley"),
            "soil_preference": 0.83,
            "water_demand": 0.70,
            "climate_preference": 0.84,
            "fertilizer_responsiveness": 0.82,
        },
        "Potatoes": {
            "patterns": ("pomme de terre", "potato"),
            "soil_preference": 0.89,
            "water_demand": 0.96,
            "climate_preference": 0.87,
            "fertilizer_responsiveness": 1.01,
        },
    }

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.merged_csv_path = self.results_dir / "merged_agriculture_dataset.csv"
        self.load_log: List[str] = []
        self.derived_variables: List[Dict[str, str]] = []
        self.used_files: List[str] = []

    @staticmethod
    def _normalize_text(value: object) -> str:
        text = "" if value is None else str(value)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    @staticmethod
    def _clean_display_text(value: object) -> str:
        text = "" if value is None else str(value)
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        return text

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def _record_derived(self, name: str, reason: str) -> None:
        self.derived_variables.append({"name": name, "reason": reason})

    @staticmethod
    def _min_max_scale(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.dropna().empty:
            return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
        low = float(numeric.min())
        high = float(numeric.max())
        if np.isclose(low, high):
            return pd.Series(np.full(len(values), 0.5), index=values.index, dtype=float)
        return (numeric - low) / (high - low)

    def _resolve_file(self, patterns: Tuple[str, ...]) -> Optional[Path]:
        files = sorted(self.data_dir.glob("*.xlsx"))
        for path in files:
            normalized_name = self._normalize_text(path.name)
            if all(self._normalize_text(pattern) in normalized_name for pattern in patterns):
                return path
        return None

    @staticmethod
    def _parse_year_end(label: object) -> Optional[int]:
        text = "" if label is None else str(label).strip()
        if not text:
            return None
        match = re.search(r"(\d{4})\s*/\s*(\d{2,4})", text)
        if match:
            start_year = int(match.group(1))
            end_text = match.group(2)
            end_year = int(end_text) if len(end_text) == 4 else int(str(start_year)[:2] + end_text)
            return end_year
        match = re.search(r"(\d{4})\s*-\s*(\d{2,4})", text)
        if match:
            start_year = int(match.group(1))
            end_text = match.group(2)
            end_year = int(end_text) if len(end_text) == 4 else int(str(start_year)[:2] + end_text)
            return end_year
        match = re.search(r"(20\d{2})", text)
        if match:
            return int(match.group(1))
        return None

    def _load_production_dataset(self) -> pd.DataFrame:
        path = self._resolve_file(("production", "vegetale"))
        if path is None:
            raise FileNotFoundError("Official production dataset not found in data/.")

        raw = pd.read_excel(path)
        cleaned_columns = {column: self._normalize_text(column) for column in raw.columns}
        dataframe = raw.rename(columns=cleaned_columns).copy()

        required = {"occurrence", "filiere", "produit", "indicateur", "valeur"}
        if not required.issubset(set(dataframe.columns)):
            raise ValueError("Production dataset columns could not be mapped to the expected structure.")

        dataframe["occurrence"] = dataframe["occurrence"].map(self._clean_display_text)
        dataframe["filiere"] = dataframe["filiere"].map(self._clean_display_text)
        dataframe["produit"] = dataframe["produit"].map(self._clean_display_text)
        dataframe["indicateur"] = dataframe["indicateur"].map(self._clean_display_text)
        dataframe["valeur"] = pd.to_numeric(dataframe["valeur"], errors="coerce")
        dataframe = dataframe.dropna(subset=["valeur"]).copy()
        dataframe["season_year"] = dataframe["occurrence"].map(self._parse_year_end)
        dataframe["indicator_key"] = dataframe["indicateur"].map(self._normalize_text)

        dataframe["metric_name"] = np.where(
            dataframe["indicator_key"].str.contains("production"),
            "production_tonnes",
            np.where(dataframe["indicator_key"].str.contains("superficie"), "surface_ha", "other_metric"),
        )

        filtered = dataframe[dataframe["metric_name"].isin({"production_tonnes", "surface_ha"})].copy()
        pivot = (
            filtered.pivot_table(
                index=["occurrence", "season_year", "filiere", "produit"],
                columns="metric_name",
                values="valeur",
                aggfunc="sum",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        if "production_tonnes" not in pivot.columns:
            pivot["production_tonnes"] = np.nan
        if "surface_ha" not in pivot.columns:
            pivot["surface_ha"] = np.nan

        pivot["yield_t_per_ha"] = pivot["production_tonnes"] / pivot["surface_ha"].replace(0, np.nan)
        pivot["yield_t_per_ha"] = pivot["yield_t_per_ha"].replace([np.inf, -np.inf], np.nan)
        pivot = pivot.dropna(subset=["season_year"]).copy()
        pivot["season_year"] = pivot["season_year"].astype(int)
        pivot["official_dataset"] = path.name

        self.load_log.append(
            f"Loaded official production dataset '{path.name}' with {len(pivot)} crop-year records after cleaning."
        )
        self.used_files.append(path.name)
        return pivot

    def _load_sector_dataset(self) -> pd.DataFrame:
        path = self._resolve_file(("indicateurs", "agriculture"))
        if path is None:
            raise FileNotFoundError("Official sector indicators dataset not found in data/.")

        raw = pd.read_excel(path, header=None)
        year_row_index: Optional[int] = None
        year_columns: Dict[int, Dict[str, object]] = {}

        for row_index in range(min(12, len(raw))):
            matches: Dict[int, Dict[str, object]] = {}
            for column_index, value in enumerate(raw.iloc[row_index].tolist()):
                text = self._clean_display_text(value)
                year_end = self._parse_year_end(text)
                if year_end is None:
                    continue
                matches[column_index] = {
                    "label": text,
                    "year": year_end,
                    "is_average": "-" in text and "/" not in text,
                }
            if len(matches) >= 3:
                year_row_index = row_index
                year_columns = matches
                break

        if year_row_index is None or not year_columns:
            raise ValueError("Sector indicators dataset year header row could not be detected.")

        records: List[Dict[str, object]] = []
        current_group = "general"

        for row_index in range(year_row_index + 1, len(raw)):
            row = raw.iloc[row_index]
            indicator_name = self._clean_display_text(row.iloc[0]) if len(row) > 0 else ""
            if indicator_name.lower().startswith("source"):
                break

            values: List[float] = []
            for column_index in year_columns:
                numeric = pd.to_numeric(row.iloc[column_index], errors="coerce")
                if pd.notna(numeric):
                    values.append(float(numeric))

            if indicator_name and not values:
                current_group = indicator_name
                continue
            if not indicator_name or not values:
                continue

            for column_index, meta in year_columns.items():
                numeric = pd.to_numeric(row.iloc[column_index], errors="coerce")
                if pd.isna(numeric):
                    continue
                records.append(
                    {
                        "indicator_group": current_group,
                        "indicator_name": indicator_name,
                        "year_label": str(meta["label"]),
                        "year": int(meta["year"]),
                        "is_average": bool(meta["is_average"]),
                        "value": float(numeric),
                        "official_dataset": path.name,
                    }
                )

        sector = pd.DataFrame(records)
        if sector.empty:
            raise ValueError("Sector indicators dataset could not be converted into usable records.")

        sector["indicator_group_key"] = sector["indicator_group"].map(self._normalize_text)
        sector["indicator_key"] = sector["indicator_name"].map(self._normalize_text)

        self.load_log.append(
            f"Loaded official sector indicators dataset '{path.name}' with {len(sector)} normalized indicator-year rows."
        )
        self.used_files.append(path.name)
        return sector

    @staticmethod
    def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
        clean = clean[clean["weight"] > 0]
        if clean.empty:
            non_null = values.dropna()
            return float(non_null.mean()) if not non_null.empty else 0.0
        return float(np.average(clean["value"], weights=clean["weight"]))

    def _extract_sector_value(
        self,
        sector: pd.DataFrame,
        patterns: Tuple[str, ...],
        year: int,
        prefer_average: bool = False,
    ) -> float:
        key_pattern = "|".join(self._normalize_text(pattern) for pattern in patterns)
        subset = sector[sector["indicator_key"].str.contains(key_pattern, na=False)].copy()
        if prefer_average:
            subset = subset[subset["is_average"]]
        else:
            subset = subset[(~subset["is_average"]) & (subset["year"] == year)]
        if subset.empty:
            subset = sector[sector["indicator_key"].str.contains(key_pattern, na=False)].copy()
            if subset.empty:
                return 0.0
            subset = subset.sort_values(["is_average", "year"], ascending=[True, False])
        return float(subset.iloc[0]["value"])

    def _match_crop_records(self, production: pd.DataFrame, patterns: Tuple[str, ...]) -> pd.DataFrame:
        pattern = "|".join(self._normalize_text(item) for item in patterns)
        return production[
            production["produit"].map(self._normalize_text).str.contains(pattern, na=False)
            | production["filiere"].map(self._normalize_text).str.contains(pattern, na=False)
        ].copy()

    def _derive_context_values(self, sector: pd.DataFrame, year: int) -> Dict[str, float]:
        precipitation_mm = self._extract_sector_value(sector, ("precipitations", "precipitation"), year)
        precipitation_mean = self._extract_sector_value(
            sector,
            ("precipitations", "precipitation"),
            year,
            prefer_average=True,
        )
        cultivated_area_kha = self._extract_sector_value(sector, ("superficie cultivee totale",), year)
        cultivated_intensity_pct = self._extract_sector_value(
            sector,
            ("superficie cultivee cultivable",),
            year,
            prefer_average=True,
        )
        if cultivated_intensity_pct <= 0:
            cultivated_intensity_pct = self._extract_sector_value(
                sector,
                ("superficie cultivee cultivable",),
                year,
            )

        precipitation_ratio = precipitation_mm / max(precipitation_mean, 1.0)
        soil_quality_index = self._clamp(0.70 + 0.0034 * cultivated_intensity_pct, 0.70, 1.08)
        water_availability_index = self._clamp(0.48 + precipitation_mm / 600.0, 0.50, 1.18)
        climate_favorability_index = self._clamp(0.56 + 0.52 * precipitation_ratio, 0.58, 1.16)
        fertilizer_efficiency_index = self._clamp(
            0.74 + 0.0015 * cultivated_intensity_pct + 0.18 * precipitation_ratio,
            0.75,
            1.12,
        )
        precipitation_stress_index = self._clamp(precipitation_ratio, 0.30, 1.00)
        resource_balance_index = self._clamp(
            0.34 * soil_quality_index + 0.38 * water_availability_index + 0.28 * climate_favorability_index,
            0.40,
            1.00,
        )
        return {
            "precipitation_mm": round(precipitation_mm, 3),
            "historical_precipitation_mm": round(precipitation_mean, 3),
            "cultivated_area_kha": round(cultivated_area_kha, 3),
            "cultivated_intensity_pct": round(cultivated_intensity_pct, 3),
            "soil_quality_index": round(soil_quality_index, 3),
            "water_availability_index": round(water_availability_index, 3),
            "climate_favorability_index": round(climate_favorability_index, 3),
            "fertilizer_efficiency_index": round(fertilizer_efficiency_index, 3),
            "precipitation_stress_index": round(precipitation_stress_index, 3),
            "resource_balance_index": round(resource_balance_index, 3),
        }

    def _build_context(self, production: pd.DataFrame, sector: pd.DataFrame) -> Dict[str, object]:
        production_years = sorted(int(year) for year in production["season_year"].dropna().unique().tolist())
        sector_actual_years = sorted(int(year) for year in sector.loc[~sector["is_average"], "year"].dropna().unique().tolist())
        overlapping_years = sorted(set(production_years).intersection(sector_actual_years))
        context_year = overlapping_years[-1] if overlapping_years else production_years[-1]
        context_values = self._derive_context_values(sector, context_year)

        self._record_derived(
            "soil_quality_index_derived",
            "Derived from official cultivated intensity indicators to create a simulation-ready soil suitability baseline.",
        )
        self._record_derived(
            "water_availability_index_derived",
            "Derived from official precipitation indicators to approximate national water sufficiency in the simulation context.",
        )
        self._record_derived(
            "climate_favorability_index_derived",
            "Derived from the ratio between current and historical precipitation to represent climate favorability.",
        )
        self._record_derived(
            "fertilizer_efficiency_index_derived",
            "Derived because the official files do not directly provide fertilizer usage data for this simulation.",
        )
        self._record_derived(
            "resource_balance_index_derived",
            "Derived from normalized soil, water, and climate context scores to support scaled optimization.",
        )

        return {
            "country": "Morocco",
            "region": "National",
            "year": context_year,
            "season_label": f"{context_year - 1}/{context_year}",
            **context_values,
            "data_mode": "official_moroccan_data",
        }

    def _build_crop_profiles(self, production: pd.DataFrame, context: Dict[str, object]) -> pd.DataFrame:
        context_year = int(context["year"])
        crop_rows: List[Dict[str, object]] = []

        for crop_name, spec in self.CROP_LIBRARY.items():
            matches = self._match_crop_records(production, spec["patterns"])
            if matches.empty:
                continue

            context_matches = matches[matches["season_year"] == context_year].copy()
            latest_year = context_year
            if context_matches.empty:
                latest_year = int(matches["season_year"].max())
                context_matches = matches[matches["season_year"] == latest_year].copy()

            yearly = (
                matches.groupby("season_year")
                .apply(
                    lambda frame: pd.Series(
                        {
                            "weighted_yield": self._weighted_mean(frame["yield_t_per_ha"], frame["surface_ha"]),
                            "surface_ha": frame["surface_ha"].sum(),
                        }
                    )
                )
                .reset_index()
            )

            historical_mean_yield = self._weighted_mean(matches["yield_t_per_ha"], matches["surface_ha"])
            latest_yield = self._weighted_mean(context_matches["yield_t_per_ha"], context_matches["surface_ha"])
            latest_surface = float(context_matches["surface_ha"].sum())
            latest_production = float(context_matches["production_tonnes"].sum())

            stability = 1.0
            if len(yearly) >= 2 and yearly["weighted_yield"].notna().any():
                mean_yield = float(yearly["weighted_yield"].mean())
                std_yield = float(yearly["weighted_yield"].std(ddof=0))
                stability = self._clamp(1.0 - std_yield / max(mean_yield, 1e-6), 0.58, 1.05)

            trend_score = 1.0
            if len(yearly) >= 2:
                x_values = yearly["season_year"].astype(float).to_numpy()
                y_values = yearly["weighted_yield"].fillna(yearly["weighted_yield"].mean()).astype(float).to_numpy()
                slope, _ = np.polyfit(x_values, y_values, 1)
                trend_score = self._clamp(1.0 + slope / max(np.mean(y_values), 1.0), 0.82, 1.18)

            official_matches = sorted(context_matches["produit"].dropna().astype(str).unique().tolist())
            crop_rows.append(
                {
                    "canonical_crop": crop_name,
                    "matched_products": "; ".join(official_matches),
                    "matched_filiere": "; ".join(sorted(context_matches["filiere"].dropna().astype(str).unique().tolist())),
                    "reference_year": latest_year,
                    "base_yield_t_ha": round(latest_yield if latest_yield > 0 else historical_mean_yield, 4),
                    "historical_mean_yield_t_ha": round(historical_mean_yield, 4),
                    "base_surface_ha": round(latest_surface, 3),
                    "base_production_tonnes": round(latest_production, 3),
                    "yield_stability": round(stability, 4),
                    "trend_score": round(trend_score, 4),
                    "soil_preference": spec["soil_preference"],
                    "water_demand": spec["water_demand"],
                    "climate_preference": spec["climate_preference"],
                    "fertilizer_responsiveness": spec["fertilizer_responsiveness"],
                    "official_dataset_flag": "official_crop_profile",
                }
            )

        crop_profiles = pd.DataFrame(crop_rows)
        if crop_profiles.empty:
            raise ValueError("No target crops could be extracted from the official Moroccan production dataset.")

        decision_surface_ha = round(self._clamp(float(crop_profiles["base_surface_ha"].median()) * 0.08, 800.0, 2200.0), 2)
        context["decision_surface_ha"] = decision_surface_ha
        context["yield_reference_t_ha"] = round(float(crop_profiles["base_yield_t_ha"].median()), 4)
        context["yield_scale_cap_t_ha"] = round(float(crop_profiles["base_yield_t_ha"].quantile(0.90)), 4)
        self._record_derived(
            "decision_surface_ha_derived",
            "Derived from the median official cultivated surface of the selected crops to keep scenario comparison readable in the live simulation.",
        )
        self._record_derived(
            "yield_reference_t_ha_derived",
            "Derived from the median official crop yield to keep scenario yield scaling consistent across agents.",
        )

        crop_profiles["normalized_base_yield_score"] = self._min_max_scale(crop_profiles["base_yield_t_ha"]).round(4)
        crop_profiles["normalized_surface_score"] = self._min_max_scale(crop_profiles["base_surface_ha"]).round(4)
        crop_profiles["normalized_production_score"] = self._min_max_scale(crop_profiles["base_production_tonnes"]).round(4)
        crop_profiles["normalized_stability_score"] = np.clip(
            (crop_profiles["yield_stability"] - 0.55) / 0.45,
            0.0,
            1.0,
        ).round(4)
        crop_profiles["normalized_trend_score"] = np.clip(
            (crop_profiles["trend_score"] - 0.80) / 0.40,
            0.0,
            1.0,
        ).round(4)
        crop_profiles["real_columns_used"] = "occurrence,filiere,produit,indicateur,valeur,production_tonnes,surface_ha"
        crop_profiles["derived_columns_used"] = (
            "yield_t_per_ha,soil_quality_index_derived,water_availability_index_derived,"
            "climate_favorability_index_derived,fertilizer_efficiency_index_derived,"
            "resource_balance_index_derived,"
            "normalized_base_yield_score,normalized_surface_score,normalized_production_score,"
            "normalized_stability_score,normalized_trend_score,decision_surface_ha_derived,"
            "yield_reference_t_ha_derived"
        )

        crop_profiles["context_year"] = context["year"]
        crop_profiles["country"] = context["country"]
        crop_profiles["region"] = context["region"]
        crop_profiles["precipitation_mm"] = context["precipitation_mm"]
        crop_profiles["cultivated_area_kha"] = context["cultivated_area_kha"]
        crop_profiles["soil_quality_index_derived"] = context["soil_quality_index"]
        crop_profiles["water_availability_index_derived"] = context["water_availability_index"]
        crop_profiles["climate_favorability_index_derived"] = context["climate_favorability_index"]
        crop_profiles["fertilizer_efficiency_index_derived"] = context["fertilizer_efficiency_index"]
        crop_profiles["resource_balance_index_derived"] = context["resource_balance_index"]
        crop_profiles["decision_surface_ha_derived"] = context["decision_surface_ha"]

        return crop_profiles.sort_values("canonical_crop").reset_index(drop=True)

    def _build_ml_training_dataset(self, production: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
        ml_rows: List[Dict[str, object]] = []

        for crop_name, spec in self.CROP_LIBRARY.items():
            matches = self._match_crop_records(production, spec["patterns"])
            if matches.empty:
                continue

            yearly = (
                matches.groupby("season_year")
                .apply(
                    lambda frame: pd.Series(
                        {
                            "base_yield_t_ha": self._weighted_mean(frame["yield_t_per_ha"], frame["surface_ha"]),
                            "base_surface_ha": frame["surface_ha"].sum(),
                            "base_production_tonnes": frame["production_tonnes"].sum(),
                        }
                    )
                )
                .reset_index()
                .sort_values("season_year")
                .reset_index(drop=True)
            )
            if yearly.empty:
                continue

            for row_index, row in yearly.iterrows():
                year = int(row["season_year"])
                base_yield = float(row["base_yield_t_ha"]) if pd.notna(row["base_yield_t_ha"]) else np.nan
                if pd.isna(base_yield) or base_yield <= 0:
                    continue

                prior = yearly.iloc[:row_index].copy()
                prior_yields = pd.to_numeric(prior["base_yield_t_ha"], errors="coerce").dropna()
                historical_mean = float(prior_yields.mean()) if not prior_yields.empty else np.nan
                stability = 0.80
                if len(prior_yields) >= 2:
                    stability = self._clamp(
                        1.0 - float(prior_yields.std(ddof=0)) / max(float(prior_yields.mean()), 1e-6),
                        0.58,
                        1.05,
                    )
                trend_score = 1.0
                if len(prior_yields) >= 2:
                    prior_years = prior.loc[prior_yields.index, "season_year"].astype(float).to_numpy()
                    prior_values = prior_yields.astype(float).to_numpy()
                    slope, _ = np.polyfit(prior_years, prior_values, 1)
                    trend_score = self._clamp(
                        1.0 + float(slope) / max(float(np.mean(prior_values)), 1.0),
                        0.82,
                        1.18,
                    )

                ml_rows.append(
                    {
                        "canonical_crop": crop_name,
                        "season_year": year,
                        "base_yield_t_ha": round(base_yield, 4),
                        "historical_mean_yield_t_ha": round(historical_mean, 4) if pd.notna(historical_mean) else np.nan,
                        "yield_stability": round(stability, 4),
                        "trend_score": round(trend_score, 4),
                        "base_surface_ha": round(float(row["base_surface_ha"]), 4),
                        "soil_preference": spec["soil_preference"],
                        "water_demand": spec["water_demand"],
                        "climate_preference": spec["climate_preference"],
                        "fertilizer_responsiveness": spec["fertilizer_responsiveness"],
                        **self._derive_context_values(sector, year),
                    }
                )

        ml_dataset = pd.DataFrame(ml_rows)
        if not ml_dataset.empty:
            self.load_log.append(
                f"Prepared supervised ML dataset with {len(ml_dataset)} crop-year observations for yield regression."
            )
        return ml_dataset

    def _fallback_bundle(self) -> Dict[str, object]:
        self.load_log.append("Official files were not usable. Fallback demo sample was generated.")
        self._record_derived(
            "fallback_demo_sample",
            "Fallback sample used only when the official Moroccan Excel files are missing or unreadable.",
        )
        self._record_derived(
            "resource_balance_index_derived",
            "Derived in fallback mode to preserve normalized resource scaling for the simulation.",
        )
        self._record_derived(
            "yield_reference_t_ha_derived",
            "Derived in fallback mode to preserve bounded yield scaling across agents.",
        )
        context = {
            "country": "Morocco",
            "region": "National",
            "year": 2020,
            "season_label": "2019/2020",
            "precipitation_mm": 255.6,
            "historical_precipitation_mm": 461.3,
            "cultivated_area_kha": 4412.0,
            "cultivated_intensity_pct": 81.01,
            "soil_quality_index": 0.98,
            "water_availability_index": 0.91,
            "climate_favorability_index": 0.85,
            "fertilizer_efficiency_index": 0.93,
            "precipitation_stress_index": 0.55,
            "resource_balance_index": 0.92,
            "decision_surface_ha": 1200.0,
            "yield_reference_t_ha": 11.15,
            "yield_scale_cap_t_ha": 42.32,
            "data_mode": "fallback_demo_sample",
        }
        crop_profiles = pd.DataFrame(
            [
                {
                    "canonical_crop": "Wheat",
                    "matched_products": "Ble tendre; Ble dur",
                    "matched_filiere": "Cereales",
                    "reference_year": 2020,
                    "base_yield_t_ha": 2.20,
                    "historical_mean_yield_t_ha": 2.55,
                    "base_surface_ha": 1852.0,
                    "base_production_tonnes": 1770000.0,
                    "yield_stability": 0.73,
                    "trend_score": 0.95,
                    "soil_preference": 0.90,
                    "water_demand": 0.82,
                    "climate_preference": 0.88,
                    "fertilizer_responsiveness": 0.94,
                    "official_dataset_flag": "fallback",
                    "normalized_base_yield_score": 0.0,
                    "normalized_surface_score": 0.0,
                    "normalized_production_score": 1.0,
                    "normalized_stability_score": 0.40,
                    "normalized_trend_score": 0.38,
                },
                {
                    "canonical_crop": "Oranges",
                    "matched_products": "Orange",
                    "matched_filiere": "Agrumes",
                    "reference_year": 2020,
                    "base_yield_t_ha": 18.40,
                    "historical_mean_yield_t_ha": 17.80,
                    "base_surface_ha": 4300.0,
                    "base_production_tonnes": 79000.0,
                    "yield_stability": 0.84,
                    "trend_score": 1.04,
                    "soil_preference": 0.96,
                    "water_demand": 1.02,
                    "climate_preference": 0.94,
                    "fertilizer_responsiveness": 0.91,
                    "official_dataset_flag": "fallback",
                    "normalized_base_yield_score": 0.33,
                    "normalized_surface_score": 0.45,
                    "normalized_production_score": 0.05,
                    "normalized_stability_score": 0.64,
                    "normalized_trend_score": 0.60,
                },
                {
                    "canonical_crop": "Tomatoes",
                    "matched_products": "Tomate",
                    "matched_filiere": "Maraichage",
                    "reference_year": 2020,
                    "base_yield_t_ha": 52.00,
                    "historical_mean_yield_t_ha": 49.50,
                    "base_surface_ha": 6100.0,
                    "base_production_tonnes": 122000.0,
                    "yield_stability": 0.88,
                    "trend_score": 1.05,
                    "soil_preference": 0.93,
                    "water_demand": 0.98,
                    "climate_preference": 0.92,
                    "fertilizer_responsiveness": 1.02,
                    "official_dataset_flag": "fallback",
                    "normalized_base_yield_score": 1.0,
                    "normalized_surface_score": 0.69,
                    "normalized_production_score": 0.07,
                    "normalized_stability_score": 0.73,
                    "normalized_trend_score": 0.63,
                },
                {
                    "canonical_crop": "Olives",
                    "matched_products": "Olivier",
                    "matched_filiere": "Oleiculture",
                    "reference_year": 2020,
                    "base_yield_t_ha": 3.90,
                    "historical_mean_yield_t_ha": 3.60,
                    "base_surface_ha": 7200.0,
                    "base_production_tonnes": 28100.0,
                    "yield_stability": 0.86,
                    "trend_score": 1.03,
                    "soil_preference": 0.95,
                    "water_demand": 0.74,
                    "climate_preference": 0.96,
                    "fertilizer_responsiveness": 0.84,
                    "official_dataset_flag": "fallback",
                    "normalized_base_yield_score": 0.03,
                    "normalized_surface_score": 1.0,
                    "normalized_production_score": 0.0,
                    "normalized_stability_score": 0.69,
                    "normalized_trend_score": 0.58,
                },
            ]
        )
        for column, key in (
            ("context_year", "year"),
            ("country", "country"),
            ("region", "region"),
            ("precipitation_mm", "precipitation_mm"),
            ("cultivated_area_kha", "cultivated_area_kha"),
            ("soil_quality_index_derived", "soil_quality_index"),
            ("water_availability_index_derived", "water_availability_index"),
            ("climate_favorability_index_derived", "climate_favorability_index"),
            ("fertilizer_efficiency_index_derived", "fertilizer_efficiency_index"),
            ("resource_balance_index_derived", "resource_balance_index"),
            ("decision_surface_ha_derived", "decision_surface_ha"),
        ):
            crop_profiles[column] = context[key]
        crop_profiles["real_columns_used"] = "occurrence,filiere,produit,indicateur,valeur,production_tonnes,surface_ha"
        crop_profiles["derived_columns_used"] = (
            "yield_t_per_ha,soil_quality_index_derived,water_availability_index_derived,"
            "climate_favorability_index_derived,fertilizer_efficiency_index_derived,"
            "resource_balance_index_derived,normalized_base_yield_score,normalized_surface_score,"
            "normalized_production_score,normalized_stability_score,normalized_trend_score,"
            "decision_surface_ha_derived,yield_reference_t_ha_derived"
        )

        ml_dataset = crop_profiles[
            [
                "canonical_crop",
                "reference_year",
                "base_yield_t_ha",
                "historical_mean_yield_t_ha",
                "yield_stability",
                "trend_score",
                "base_surface_ha",
                "soil_preference",
                "water_demand",
                "climate_preference",
                "fertilizer_responsiveness",
            ]
        ].copy()
        ml_dataset = ml_dataset.rename(columns={"reference_year": "season_year"})
        ml_dataset["historical_precipitation_mm"] = context["historical_precipitation_mm"]
        ml_dataset["cultivated_intensity_pct"] = context["cultivated_intensity_pct"]
        ml_dataset["soil_quality_index"] = context["soil_quality_index"]
        ml_dataset["water_availability_index"] = context["water_availability_index"]
        ml_dataset["climate_favorability_index"] = context["climate_favorability_index"]
        ml_dataset["fertilizer_efficiency_index"] = context["fertilizer_efficiency_index"]
        ml_dataset["precipitation_stress_index"] = context["precipitation_stress_index"]
        ml_dataset["resource_balance_index"] = context["resource_balance_index"]
        ml_dataset["precipitation_mm"] = context["precipitation_mm"]
        ml_dataset["cultivated_area_kha"] = context["cultivated_area_kha"]

        crop_profiles.to_csv(self.merged_csv_path, index=False)
        return {
            "merged_dataset": crop_profiles,
            "ml_dataset": ml_dataset,
            "context": context,
            "load_log": list(self.load_log),
            "derived_variables": list(self.derived_variables),
            "used_files": list(self.used_files),
            "real_columns": [
                "occurrence",
                "filiere",
                "produit",
                "indicateur",
                "valeur",
                "production_tonnes",
                "surface_ha",
            ],
            "derived_columns": [
                "yield_t_per_ha",
                "soil_quality_index_derived",
                "water_availability_index_derived",
                "climate_favorability_index_derived",
                "fertilizer_efficiency_index_derived",
                "resource_balance_index_derived",
                "normalized_base_yield_score",
                "normalized_surface_score",
                "normalized_production_score",
                "normalized_stability_score",
                "normalized_trend_score",
                "decision_surface_ha_derived",
            ],
        }

    def load_agricultural_data(self) -> Dict[str, object]:
        try:
            production = self._load_production_dataset()
            sector = self._load_sector_dataset()
            context = self._build_context(production, sector)
            crop_profiles = self._build_crop_profiles(production, context)
            ml_dataset = self._build_ml_training_dataset(production, sector)
            crop_profiles.to_csv(self.merged_csv_path, index=False)
            self.load_log.append(f"Saved merged agricultural dataset to '{self.merged_csv_path.name}'.")
            return {
                "merged_dataset": crop_profiles,
                "ml_dataset": ml_dataset,
                "context": context,
                "load_log": list(self.load_log),
                "derived_variables": list(self.derived_variables),
                "used_files": list(dict.fromkeys(self.used_files)),
                "real_columns": [
                    "occurrence",
                    "filiere",
                    "produit",
                    "indicateur",
                    "valeur",
                    "production_tonnes",
                    "surface_ha",
                ],
                "derived_columns": [
                    "yield_t_per_ha",
                    "soil_quality_index_derived",
                    "water_availability_index_derived",
                    "climate_favorability_index_derived",
                    "fertilizer_efficiency_index_derived",
                    "resource_balance_index_derived",
                    "normalized_base_yield_score",
                    "normalized_surface_score",
                    "normalized_production_score",
                    "normalized_stability_score",
                    "normalized_trend_score",
                    "decision_surface_ha_derived",
                ],
            }
        except Exception as exc:
            self.load_log.append(f"DataAgent fallback triggered: {exc}")
            return self._fallback_bundle()
