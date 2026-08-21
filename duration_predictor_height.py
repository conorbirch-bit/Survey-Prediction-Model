from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
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
    """
    Predicts site-survey duration from:
      - Building Height
      - Internal Ground Floor Area (m2)
      - Sovereign Flat

    A separate Ridge regression is trained for every available feature
    combination. At prediction time, the model with the greatest number
    of supplied features is used, so missing inputs do not prevent a prediction.
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
        # Version 17 intentionally applies no survey-duration uplift or rounding.
        self.planning_buffer_pct = 0.0
        self.planning_round_to = planning_round_to
        self.models: Dict[Tuple[str, ...], Pipeline] = {}
        self.stats: Dict[Tuple[str, ...], ModelStats] = {}
        self.training_data: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Be tolerant of small Salesforce/export naming differences.
        aliases = {
            "Building Height": ["Building Height", "Building height"],
            "Internal Ground Floor Area (m2)": [
                "Internal Ground Floor Area (m2)",
                "Internal Ground Floor Area",
                "Internal Ground Floor Area (m²)",
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
        """
        Read either a normal completed-surveys table or a Salesforce report
        export with title/filter rows above the real column headers.

        The completed-file layout can therefore change from:
            row 1 = headers
        to the Salesforce report format where the header row appears later.
        """
        raw = pd.read_excel(source, sheet_name=sheet_name, header=None)

        # Match the canonical names and the aliases accepted by
        # _normalise_columns().  The actual-duration column is the strongest
        # marker that we have found the completed-survey table rather than a
        # report title/filter row.
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
                # Require at least three recognised modelling/report columns so
                # an incidental bit of report metadata cannot be mistaken for
                # the table header.
                if len(values.intersection(header_aliases)) >= 3:
                    header_row = int(idx)
                    break

        if header_row is None or header_row == 0:
            # Preserve Version 17 behaviour exactly for the original row-1
            # table format.  Salesforce-only footer cleanup is applied only
            # when report metadata pushes the real header lower down.
            return pd.read_excel(source, sheet_name=sheet_name)

        df = pd.read_excel(source, sheet_name=sheet_name, header=header_row)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        # Salesforce report exports append Total / Count footer rows.  Those can
        # contain numeric sums in the duration and flat-count columns, so they
        # must be removed before training or they would look like giant sites.
        text_view = df.astype(str).apply(lambda col: col.str.strip().str.lower())
        footer_mask = text_view.apply(
            lambda row: row.isin({"total", "sum", "count"}).any(),
            axis=1,
        )
        df = df[~footer_mask].copy()

        # Detail rows in the completed report have a building and/or customer
        # reference.  This also safely removes any remaining summary footer row.
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
        df = self.read_training_excel(path, sheet_name=sheet_name)
        return self.fit(df)

    def fit(self, df: pd.DataFrame) -> "DurationPredictor":
        df = self._normalise_columns(df.copy())

        required = list(FEATURE_COLUMNS.values()) + [TARGET_COLUMN]
        missing_columns = [c for c in required if c not in df.columns]
        if missing_columns:
            raise ValueError(
                "Training spreadsheet is missing required columns: "
                + ", ".join(missing_columns)
            )

        # Convert model columns to numeric. Bad strings become missing values.
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Completed-survey rule: ignore blank/zero and obvious 1–5 minute
        # aborted/failed visits. Change min_completed_duration if needed.
        df = df[
            df[TARGET_COLUMN].notna()
            & (df[TARGET_COLUMN] >= self.min_completed_duration)
        ].copy()

        # Residential duration model only: an explicit 0-flat record is a
        # garage and is excluded. Missing flat counts are retained so the
        # height + ground-floor-area fallback can still be trained.
        flats_col = FEATURE_COLUMNS["flats"]
        df = df[df[flats_col].isna() | (df[flats_col] > 0)].copy()

        self.training_data = df
        self.models.clear()
        self.stats.clear()

        keys = list(FEATURE_COLUMNS.keys())

        # Train every non-empty combination so predictions survive missing data.
        for size in range(1, len(keys) + 1):
            for feature_keys in combinations(keys, size):
                cols = [FEATURE_COLUMNS[k] for k in feature_keys]
                subset = df.dropna(subset=cols + [TARGET_COLUMN]).copy()

                # Need enough observations for a meaningful model.
                min_rows = max(5, len(feature_keys) + 3)
                if len(subset) < min_rows:
                    continue

                X = subset[cols].to_numpy(dtype=float)
                y = subset[TARGET_COLUMN].to_numpy(dtype=float)

                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=self.ridge_alpha)),
                ])
                model.fit(X, y)
                self.models[feature_keys] = model

                mae = rmse = None
                if len(subset) >= 6:
                    loo = LeaveOneOut()
                    pred = cross_val_predict(model, X, y, cv=loo)
                    mae = float(mean_absolute_error(y, pred))
                    rmse = float(math.sqrt(mean_squared_error(y, pred)))

                confidence = self._confidence(mae, len(subset), len(feature_keys))
                self.stats[feature_keys] = ModelStats(
                    features=feature_keys,
                    rows=len(subset),
                    mae_minutes=mae,
                    rmse_minutes=rmse,
                    confidence=confidence,
                )

        if not self.models:
            raise ValueError("Not enough complete historical data to train any model.")

        return self

    @staticmethod
    def _confidence(mae: Optional[float], rows: int, n_features: int) -> str:
        if mae is None or rows < 10:
            return "Low"
        # Operational confidence bands, not statistical prediction intervals.
        if mae <= 15 and rows >= 25:
            return "High"
        if mae <= 22 and rows >= max(15, 5 * n_features):
            return "Medium"
        return "Low"

    def _choose_model(self, values: Dict[str, Optional[float]]) -> Tuple[str, ...]:
        available = {
            k for k, v in values.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        compatible = [
            features for features in self.models
            if set(features).issubset(available)
        ]
        if not compatible:
            raise ValueError(
                "No prediction can be made because none of building height, "
                "ground-floor area or Sovereign flats is available."
            )

        # Version 17 fallback rule:
        # If ground-floor area is missing, use flats ONLY. Building height is
        # deliberately ignored in that situation. This avoids switching to a
        # height + flats equation when the more physically informative area
        # input is absent.
        if "ground_floor_area" not in available:
            if "flats" in available and ("flats",) in self.models:
                return ("flats",)
            raise ValueError(
                "Ground-floor area is unavailable and no Sovereign flat count "
                "is available, so this residential model cannot predict duration."
            )

        # When area is available, use as much compatible information as possible.
        # This preserves height + area as the main fallback when flats are missing.
        def rank(features):
            stat = self.stats[features]
            mae = (
                stat.mae_minutes
                if stat.mae_minutes is not None
                else float("inf")
            )
            return (len(features), stat.rows, -mae)

        return max(compatible, key=rank)

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

        if values["flats"] == 0:
            raise ValueError(
                "0 Sovereign flats identifies a garage, which is excluded from "
                "the residential survey-duration model."
            )

        features = self._choose_model(values)
        cols = [FEATURE_COLUMNS[k] for k in features]
        x = np.array([[values[k] for k in features]], dtype=float)

        raw = float(self.models[features].predict(x)[0])
        predicted = max(self.min_completed_duration, raw)

        # Planning duration is intentionally identical to the raw model
        # prediction in Version 17: no percentage uplift and no 5-minute rounding.
        predicted = round(predicted, 1)
        planning = predicted

        stat = self.stats[features]
        missing = tuple(k for k, v in values.items() if v is None)

        return PredictionResult(
            predicted_minutes=predicted,
            planning_minutes=planning,
            model_features=features,
            model_label=" + ".join(self.pretty_feature(k) for k in features),
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

    def model_summary(self) -> pd.DataFrame:
        rows = []
        for features, stat in sorted(
            self.stats.items(), key=lambda x: (len(x[0]), x[0])
        ):
            rows.append({
                "Model": " + ".join(self.pretty_feature(k) for k in features),
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
        return pd.DataFrame(rows)
