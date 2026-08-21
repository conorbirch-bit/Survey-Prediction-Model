from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = {
    "building_height": "Building Height",
    "ground_floor_area": "Internal Ground Floor Area (m2)",
    "flats": "Sovereign Flat",
}

TARGET_COLUMN = "Primary Service Appointment: Actual Duration (Minutes)"
REFERENCE_COLUMN = "Customer Reference Code  ↑"
BUILDING_COLUMN = "Building Name"

SMALL_SEGMENT = "1–6 flats"
STANDARD_SEGMENT = "7+ flats"
GARAGE_SEGMENT = "Garage"
FALLBACK_SEGMENT = "Residential fallback (flats missing)"
MODEL_VERSION = "20.2-segmented-1-6"


@dataclass
class ModelStats:
    features: Tuple[str, ...]
    rows: int
    mae_minutes: Optional[float]
    rmse_minutes: Optional[float]
    confidence: str


@dataclass
class PredictionResult:
    predicted_minutes: float
    planning_minutes: float
    model_features: Tuple[str, ...]
    model_label: str
    confidence: str
    validation_mae_minutes: Optional[float]
    validation_rmse_minutes: Optional[float]
    training_rows: int
    feature_count: int
    missing_inputs: Tuple[str, ...]

    def to_dict(self):
        return asdict(self)


class DurationPredictor:
    model_version = MODEL_VERSION
    """
    Version 20.2 segmented duration model.

    Known flat counts select one of three independent model families:
      - 0 flats       -> Garage model
      - 1 to 6 flats  -> Small-building model
      - 7+ flats      -> Standard/larger-building model

    The 1–6 and 7+ residential families each train their own Ridge regressions
    for all viable combinations of Building Height, Ground Floor Area and
    Sovereign Flats. The existing Version 17 fallback rule is preserved:
    whenever Ground Floor Area is missing, a residential prediction uses
    Sovereign Flats only.

    If the flat count itself is missing, the model cannot determine the size
    family, so it falls back to the original all-residential Height/Area model.

    Garages are trained separately from residential buildings. Because flats is
    always zero for a garage it is not used as a garage predictor. Height and/or
    Ground Floor Area are used when enough completed garages exist; otherwise
    the historical mean garage duration is used as the garage-only baseline.
    """

    def __init__(
        self,
        min_completed_duration: float = 6.0,
        ridge_alpha: float = 10.0,
        planning_buffer_pct: float = 0.15,
        planning_round_to: int = 5,
    ):
        self.min_completed_duration = min_completed_duration
        self.ridge_alpha = ridge_alpha
        # Kept for backwards compatibility with older app calls.
        # Planning duration intentionally equals raw prediction.
        self.planning_buffer_pct = 0.0
        self.planning_round_to = planning_round_to

        # Global residential models are retained only for buildings whose flat
        # count is missing, because those buildings cannot be assigned to the
        # 1–6 or 7+ family.
        self.models: Dict[Tuple[str, ...], Pipeline] = {}
        self.stats: Dict[Tuple[str, ...], ModelStats] = {}

        self.segment_models: Dict[str, Dict[Tuple[str, ...], Pipeline]] = {
            SMALL_SEGMENT: {},
            STANDARD_SEGMENT: {},
            GARAGE_SEGMENT: {},
        }
        self.segment_stats: Dict[str, Dict[Tuple[str, ...], ModelStats]] = {
            SMALL_SEGMENT: {},
            STANDARD_SEGMENT: {},
            GARAGE_SEGMENT: {},
        }

        self.garage_mean_minutes: Optional[float] = None
        self.garage_mean_stats: Optional[ModelStats] = None
        self.training_data: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            "Building Height": ["Building Height", "Building height"],
            "Internal Ground Floor Area (m2)": [
                "Internal Ground Floor Area (m2)",
                "Internal Ground Floor Area",
                "Internal Ground Floor Area (m²)",
                "Ground Floor Area (m2)",
                "Ground Floor Area (m²)",
                "Ground Floor Area",
            ],
            "Sovereign Flat": [
                "Sovereign Flat",
                "Sovereign Flats",
                "Sovereign Flat Count",
            ],
            TARGET_COLUMN: [
                TARGET_COLUMN,
                "Primary Service Appointment: Actual Duration",
                "Actual Duration (Minutes)",
            ],
        }

        rename = {}
        stripped = {str(c).strip(): c for c in df.columns}
        for canonical, possibilities in aliases.items():
            for option in possibilities:
                if option in stripped:
                    rename[stripped[option]] = canonical
                    break
        return df.rename(columns=rename)

    @classmethod
    def read_training_excel(cls, source, sheet_name=0) -> pd.DataFrame:
        """Read normal tables or Salesforce completed-survey report exports."""
        raw = pd.read_excel(source, sheet_name=sheet_name, header=None)

        header_aliases = {
            "Building Height",
            "Building height",
            "Internal Ground Floor Area (m2)",
            "Internal Ground Floor Area",
            "Internal Ground Floor Area (m²)",
            "Sovereign Flat",
            "Sovereign Flats",
            "Sovereign Flat Count",
            TARGET_COLUMN,
            "Primary Service Appointment: Actual Duration",
            "Actual Duration (Minutes)",
        }
        target_aliases = {
            TARGET_COLUMN,
            "Primary Service Appointment: Actual Duration",
            "Actual Duration (Minutes)",
        }

        header_row = None
        for idx, row in raw.iterrows():
            values = {
                str(value).strip()
                for value in row.tolist()
                if pd.notna(value)
            }
            if values.intersection(target_aliases):
                if len(values.intersection(header_aliases)) >= 3:
                    header_row = int(idx)
                    break

        if header_row is None or header_row == 0:
            return pd.read_excel(source, sheet_name=sheet_name)

        df = pd.read_excel(source, sheet_name=sheet_name, header=header_row)
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

        text_view = df.astype(str).apply(lambda col: col.str.strip().str.lower())
        footer_mask = text_view.apply(
            lambda row: row.isin({"total", "sum", "count"}).any(),
            axis=1,
        )
        df = df[~footer_mask].copy()

        identity_cols = [
            c for c in df.columns
            if str(c).strip() in {
                "Building Name",
                "Customer Reference Code  ↑",
                "Customer Reference Code",
                "Customer Reference",
                "Work Order Number",
            }
        ]
        if identity_cols:
            identity_present = pd.Series(False, index=df.index)
            for col in identity_cols:
                values = df[col]
                identity_present |= values.notna() & values.astype(str).str.strip().ne("")
            df = df[identity_present].copy()

        return df.reset_index(drop=True)

    def load_excel(self, path: str | Path, sheet_name=0) -> "DurationPredictor":
        return self.fit(self.read_training_excel(path, sheet_name=sheet_name))

    def _fit_models_for_subset(
        self,
        subset_df: pd.DataFrame,
        feature_keys,
        min_rows_mode: str = "residential",
    ) -> Tuple[Dict[Tuple[str, ...], Pipeline], Dict[Tuple[str, ...], ModelStats]]:
        models: Dict[Tuple[str, ...], Pipeline] = {}
        stats: Dict[Tuple[str, ...], ModelStats] = {}

        keys = list(feature_keys)
        for size in range(1, len(keys) + 1):
            for features in combinations(keys, size):
                cols = [FEATURE_COLUMNS[k] for k in features]
                train = subset_df.dropna(subset=cols + [TARGET_COLUMN]).copy()

                if min_rows_mode == "garage":
                    # Garages are currently a much smaller historical sample.
                    # Three rows are enough to create a one-feature garage line;
                    # two-feature garage models still require four rows.
                    min_rows = max(3, len(features) + 2)
                else:
                    min_rows = max(5, len(features) + 3)

                if len(train) < min_rows:
                    continue

                X = train[cols].to_numpy(dtype=float)
                y = train[TARGET_COLUMN].to_numpy(dtype=float)
                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=self.ridge_alpha)),
                ])
                model.fit(X, y)
                models[features] = model

                mae = rmse = None
                if len(train) >= 3:
                    loo = LeaveOneOut()
                    pred = cross_val_predict(model, X, y, cv=loo)
                    mae = float(mean_absolute_error(y, pred))
                    rmse = float(math.sqrt(mean_squared_error(y, pred)))

                stats[features] = ModelStats(
                    features=features,
                    rows=len(train),
                    mae_minutes=mae,
                    rmse_minutes=rmse,
                    confidence=self._confidence(mae, len(train), len(features)),
                )

        return models, stats

    def fit(self, df: pd.DataFrame) -> "DurationPredictor":
        df = self._normalise_columns(df.copy())

        required = list(FEATURE_COLUMNS.values()) + [TARGET_COLUMN]
        missing_columns = [c for c in required if c not in df.columns]
        if missing_columns:
            raise ValueError(
                "Training spreadsheet is missing required columns: "
                + ", ".join(missing_columns)
            )

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[
            df[TARGET_COLUMN].notna()
            & (df[TARGET_COLUMN] >= self.min_completed_duration)
        ].copy()

        # A few row-1 Excel exports contain a final numeric totals row. If the
        # file contains identity columns, require at least one identity value so
        # report totals can never be learned as if they were a giant building.
        identity_candidates = [
            "Customer Reference Code  ↑",
            "Customer Reference Code",
            "Customer Reference",
            "Building Name",
            "Work Order Number",
        ]
        identity_cols = [c for c in identity_candidates if c in df.columns]
        if identity_cols:
            identity_present = pd.Series(False, index=df.index)
            for col in identity_cols:
                values = df[col]
                identity_present |= (
                    values.notna()
                    & values.astype(str).str.strip().ne("")
                    & values.astype(str).str.strip().str.lower().ne("nan")
                )
            df = df[identity_present].copy()

        self.training_data = df

        flats_col = FEATURE_COLUMNS["flats"]
        all_keys = list(FEATURE_COLUMNS.keys())

        residential = df[df[flats_col].isna() | (df[flats_col] > 0)].copy()
        small = df[(df[flats_col] >= 1) & (df[flats_col] <= 6)].copy()
        standard = df[df[flats_col] >= 7].copy()
        garages = df[df[flats_col] == 0].copy()

        # Original all-residential family: used only when flat count is missing.
        self.models, self.stats = self._fit_models_for_subset(
            residential, all_keys, min_rows_mode="residential"
        )

        self.segment_models[SMALL_SEGMENT], self.segment_stats[SMALL_SEGMENT] = (
            self._fit_models_for_subset(
                small, all_keys, min_rows_mode="residential"
            )
        )
        self.segment_models[STANDARD_SEGMENT], self.segment_stats[STANDARD_SEGMENT] = (
            self._fit_models_for_subset(
                standard, all_keys, min_rows_mode="residential"
            )
        )

        # Garages have their own family. Flat count is deliberately excluded
        # because it is always zero and therefore contains no within-garage
        # information.
        garage_keys = ["building_height", "ground_floor_area"]
        self.segment_models[GARAGE_SEGMENT], self.segment_stats[GARAGE_SEGMENT] = (
            self._fit_models_for_subset(
                garages, garage_keys, min_rows_mode="garage"
            )
        )

        self.garage_mean_minutes = None
        self.garage_mean_stats = None
        garage_y = garages[TARGET_COLUMN].dropna().to_numpy(dtype=float)
        if len(garage_y) >= 1:
            self.garage_mean_minutes = float(np.mean(garage_y))

            garage_mae = garage_rmse = None
            if len(garage_y) >= 2:
                loo_predictions = []
                for i in range(len(garage_y)):
                    others = np.delete(garage_y, i)
                    loo_predictions.append(float(np.mean(others)))
                loo_predictions = np.asarray(loo_predictions, dtype=float)
                garage_mae = float(mean_absolute_error(garage_y, loo_predictions))
                garage_rmse = float(
                    math.sqrt(mean_squared_error(garage_y, loo_predictions))
                )

            self.garage_mean_stats = ModelStats(
                features=tuple(),
                rows=len(garage_y),
                mae_minutes=garage_mae,
                rmse_minutes=garage_rmse,
                confidence=self._confidence(garage_mae, len(garage_y), 0),
            )

        # At least one usable family must exist.
        if (
            not self.models
            and not self.segment_models[SMALL_SEGMENT]
            and not self.segment_models[STANDARD_SEGMENT]
            and self.garage_mean_minutes is None
        ):
            raise ValueError("Not enough completed historical data to train any model.")

        return self

    @staticmethod
    def _confidence(mae: Optional[float], rows: int, n_features: int) -> str:
        if mae is None or rows < 10:
            return "Low"
        if mae <= 15 and rows >= 25:
            return "High"
        if mae <= 22 and rows >= max(15, 5 * n_features):
            return "Medium"
        return "Low"

    @staticmethod
    def _available(values: Dict[str, Optional[float]]) -> set[str]:
        return {
            k for k, v in values.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

    def _ranked_compatible(
        self,
        models: Dict[Tuple[str, ...], Pipeline],
        stats: Dict[Tuple[str, ...], ModelStats],
        available: set[str],
    ) -> Tuple[str, ...]:
        compatible = [
            features for features in models
            if set(features).issubset(available)
        ]
        if not compatible:
            raise ValueError("No compatible trained duration equation is available.")

        def rank(features):
            stat = stats[features]
            mae = stat.mae_minutes if stat.mae_minutes is not None else float("inf")
            return (len(features), stat.rows, -mae)

        return max(compatible, key=rank)

    def _choose_residential_segment_model(
        self,
        segment: str,
        values: Dict[str, Optional[float]],
    ) -> Tuple[str, ...]:
        available = self._available(values)
        models = self.segment_models[segment]
        stats = self.segment_stats[segment]

        # Keep the Version 17 rule exactly: if area is absent, ignore height and
        # use the segment's own flats-only equation.
        if "ground_floor_area" not in available:
            if ("flats",) in models:
                return ("flats",)
            raise ValueError(
                f"The {segment} model does not yet have enough completed "
                "buildings to train its own flats-only equation."
            )

        return self._ranked_compatible(models, stats, available)

    def _choose_missing_flats_model(
        self,
        values: Dict[str, Optional[float]],
    ) -> Tuple[str, ...]:
        available = self._available(values)
        if "ground_floor_area" not in available:
            raise ValueError(
                "Ground-floor area and Sovereign flat count are both unavailable, "
                "so duration cannot be predicted."
            )
        return self._ranked_compatible(self.models, self.stats, available)

    def predict(
        self,
        building_height: Optional[float] = None,
        ground_floor_area: Optional[float] = None,
        flats: Optional[float] = None,
    ) -> PredictionResult:
        values = {
            "building_height": self._clean_value(building_height),
            "ground_floor_area": self._clean_value(ground_floor_area),
            "flats": self._clean_value(flats),
        }
        missing = tuple(k for k, v in values.items() if v is None)
        flats_value = values["flats"]

        if flats_value is not None and flats_value < 0:
            raise ValueError("Sovereign flat count cannot be negative.")

        # Garage: separate model family and never mixed into residential data.
        if flats_value == 0:
            segment = GARAGE_SEGMENT
            available = self._available(values) - {"flats"}
            garage_models = self.segment_models[GARAGE_SEGMENT]
            garage_stats = self.segment_stats[GARAGE_SEGMENT]

            compatible = [
                features for features in garage_models
                if set(features).issubset(available)
            ]
            if compatible:
                features = self._ranked_compatible(
                    garage_models, garage_stats, available
                )
                x = np.array([[values[k] for k in features]], dtype=float)
                raw = float(garage_models[features].predict(x)[0])
                stat = garage_stats[features]
                model_label = (
                    f"{GARAGE_SEGMENT} | "
                    + " + ".join(self.pretty_feature(k) for k in features)
                )
            elif self.garage_mean_minutes is not None and self.garage_mean_stats is not None:
                features = tuple()
                raw = float(self.garage_mean_minutes)
                stat = self.garage_mean_stats
                model_label = f"{GARAGE_SEGMENT} | historical average"
            else:
                raise ValueError(
                    "No completed garage surveys are available to train the garage model."
                )

        # Small residential: 1–6 flats inclusive.
        elif flats_value is not None and 1 <= flats_value < 7:
            segment = SMALL_SEGMENT
            features = self._choose_residential_segment_model(segment, values)
            model = self.segment_models[segment][features]
            x = np.array([[values[k] for k in features]], dtype=float)
            raw = float(model.predict(x)[0])
            stat = self.segment_stats[segment][features]
            model_label = (
                f"{segment} | "
                + " + ".join(self.pretty_feature(k) for k in features)
            )

        # Larger residential: 7 flats and above.
        elif flats_value is not None and flats_value >= 7:
            segment = STANDARD_SEGMENT
            features = self._choose_residential_segment_model(segment, values)
            model = self.segment_models[segment][features]
            x = np.array([[values[k] for k in features]], dtype=float)
            raw = float(model.predict(x)[0])
            stat = self.segment_stats[segment][features]
            model_label = (
                f"{segment} | "
                + " + ".join(self.pretty_feature(k) for k in features)
            )

        # Flat count missing: preserve Version 19's Height + Area fallback.
        else:
            segment = FALLBACK_SEGMENT
            features = self._choose_missing_flats_model(values)
            model = self.models[features]
            x = np.array([[values[k] for k in features]], dtype=float)
            raw = float(model.predict(x)[0])
            stat = self.stats[features]
            model_label = (
                f"{segment} | "
                + " + ".join(self.pretty_feature(k) for k in features)
            )

        predicted = round(max(self.min_completed_duration, raw), 1)
        planning = predicted

        return PredictionResult(
            predicted_minutes=predicted,
            planning_minutes=planning,
            model_features=features,
            model_label=model_label,
            confidence=stat.confidence,
            validation_mae_minutes=(
                round(stat.mae_minutes, 1)
                if stat.mae_minutes is not None else None
            ),
            validation_rmse_minutes=(
                round(stat.rmse_minutes, 1)
                if stat.rmse_minutes is not None else None
            ),
            training_rows=stat.rows,
            feature_count=len(features),
            missing_inputs=missing,
        )

    @staticmethod
    def _clean_value(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(value):
            return None
        return value

    @staticmethod
    def pretty_feature(key: str) -> str:
        return {
            "building_height": "Building height",
            "ground_floor_area": "Ground-floor area",
            "flats": "Sovereign flats",
        }[key]

    def _summary_rows_for_family(self, family, models, stats):
        rows = []
        for features, stat in sorted(
            stats.items(), key=lambda x: (len(x[0]), x[0])
        ):
            rows.append({
                "Model": (
                    f"{family} | "
                    + " + ".join(self.pretty_feature(k) for k in features)
                ),
                "Features": len(features),
                "Training rows": stat.rows,
                "LOOCV MAE (min)": (
                    round(stat.mae_minutes, 1)
                    if stat.mae_minutes is not None else None
                ),
                "LOOCV RMSE (min)": (
                    round(stat.rmse_minutes, 1)
                    if stat.rmse_minutes is not None else None
                ),
                "Confidence": stat.confidence,
            })
        return rows

    def model_summary(self) -> pd.DataFrame:
        rows = []
        rows.extend(
            self._summary_rows_for_family(
                FALLBACK_SEGMENT, self.models, self.stats
            )
        )
        for family in (SMALL_SEGMENT, STANDARD_SEGMENT, GARAGE_SEGMENT):
            rows.extend(
                self._summary_rows_for_family(
                    family,
                    self.segment_models[family],
                    self.segment_stats[family],
                )
            )

        if self.garage_mean_stats is not None:
            stat = self.garage_mean_stats
            rows.append({
                "Model": f"{GARAGE_SEGMENT} | historical average",
                "Features": 0,
                "Training rows": stat.rows,
                "LOOCV MAE (min)": (
                    round(stat.mae_minutes, 1)
                    if stat.mae_minutes is not None else None
                ),
                "LOOCV RMSE (min)": (
                    round(stat.rmse_minutes, 1)
                    if stat.rmse_minutes is not None else None
                ),
                "Confidence": stat.confidence,
            })

        return pd.DataFrame(rows)
