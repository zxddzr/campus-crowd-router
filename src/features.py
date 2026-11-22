"""Feature engineering shared by training (`src/train_models.py`) and serving (`app.py`).

Single source of truth: whatever the app feeds a model at inference time goes
through exactly the same transformation that the training script used.  That is
the only way the evaluation numbers in `models/metrics.json` stay meaningful.

Leakage guard
-------------
The functions below only look at the seven raw input columns declared in
`campus_config.FEATURE_COLUMNS`.  `wait_time_mins` and `congestion_level` are
never touched, so no target-derived signal can sneak into the feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.campus_config import (
    FEATURE_COLUMNS,
    LUNCH_PEAK,
    quantize_slot,
    to_minutes,
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "zone_id",
    "service_type",
    "weekday",
    "slot_bucket",
    "zone_slot",
    "weekday_slot",
    "dismissal_spike",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "week",
    "zone_capacity",
    "nearby_class_dismissals",
    "dismissal_log",
    "dismissal_load",
    "dismissal_load_sq",
    "slot_minutes",
    "is_lunch_peak",
    "minutes_to_noon",
)


def _slot_minutes_series(values: pd.Series) -> np.ndarray:
    """Accept ``"11:45"`` strings (or already-numeric minute values)."""
    out = np.empty(len(values), dtype=float)
    for i, value in enumerate(values.to_numpy()):
        if isinstance(value, str):
            out[i] = float(to_minutes(value))
        else:
            out[i] = float(value)
    return out


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Build the engineered feature frame from the seven raw inputs."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")

    out = df.loc[:, list(FEATURE_COLUMNS)].copy()
    out["week"] = out["week"].astype(float)
    out["zone_capacity"] = out["zone_capacity"].astype(float)
    out["nearby_class_dismissals"] = out["nearby_class_dismissals"].astype(float)
    out["weekday"] = out["weekday"].astype(str)
    out["zone_id"] = out["zone_id"].astype(str)
    out["service_type"] = out["service_type"].astype(str)

    minutes = _slot_minutes_series(out["time_slot"])
    out["slot_minutes"] = minutes
    out["slot_bucket"] = [quantize_slot(m) for m in minutes]

    peak_start, peak_end = LUNCH_PEAK
    out["is_lunch_peak"] = ((minutes >= peak_start) & (minutes < peak_end)).astype(float)
    # Distance to the 12:30 demand crest -- gives the linear models a smooth
    # way to represent the hump shape of the lunch peak.
    out["minutes_to_noon"] = np.abs(minutes - (12 * 60 + 30))

    out["zone_slot"] = out["zone_id"].astype(str) + "|" + out["slot_bucket"].astype(str)
    out["weekday_slot"] = out["weekday"].astype(str) + "|" + out["slot_bucket"].astype(str)

    out["dismissal_log"] = np.log1p(out["nearby_class_dismissals"].to_numpy())
    # Headcount relative to how many seats the zone has: the bottleneck is the
    # ratio, not the raw headcount, and it saturates instead of growing linearly.
    out["dismissal_load"] = out["nearby_class_dismissals"].to_numpy() / out["zone_capacity"].to_numpy()
    out["dismissal_load_sq"] = out["dismissal_load"].to_numpy() ** 2
    out["dismissal_spike"] = np.where(
        out["nearby_class_dismissals"].to_numpy() >= 100.0, "spike", "normal"
    )

    return out


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Categorical one-hot encoding + numeric scaling, ready to be fitted.

    ``sparse_output=False`` keeps the matrix dense: with ~200 engineered
    columns that costs nothing and it lets the MLP consume the same block.
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", numeric_step, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# Plain feature block (used by the naive linear baseline only)
# --------------------------------------------------------------------------- #
BASIC_CATEGORICAL: tuple[str, ...] = ("zone_id", "service_type", "weekday", "slot_bucket")
BASIC_NUMERIC: tuple[str, ...] = ("week", "zone_capacity", "nearby_class_dismissals")


def build_basic_preprocessor() -> ColumnTransformer:
    """Raw one-hot block: no interaction terms, no non-linear transforms.

    This is what a "plain" LinearRegression / LogisticRegression baseline gets,
    so the comparison table isolates what feature engineering plus
    regularisation / a neural net actually buy us.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                list(BASIC_CATEGORICAL),
            ),
            ("numeric", StandardScaler(), list(BASIC_NUMERIC)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# Compact block for the neural network
# --------------------------------------------------------------------------- #
# A one-hidden-layer network learns its own interactions, so hand-crafted
# interaction one-hots only add variance on a 640-row training set.  The MLP
# therefore gets the raw one-hots plus the *physically meaningful* continuous
# features (occupancy load = headcount / capacity and its square, which together
# can represent the saturating queue curve).
MLP_CATEGORICAL: tuple[str, ...] = ("zone_id", "service_type", "weekday", "slot_bucket")
MLP_NUMERIC: tuple[str, ...] = (
    "week",
    "zone_capacity",
    "nearby_class_dismissals",
    "dismissal_load",
    "dismissal_load_sq",
    "slot_minutes",
    "is_lunch_peak",
)


def build_mlp_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                list(MLP_CATEGORICAL),
            ),
            ("numeric", StandardScaler(), list(MLP_NUMERIC)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator) -> Pipeline:
    """Preprocessor + estimator in one fit/predict unit (engineered features)."""
    return Pipeline(
        steps=[
            ("features", enrich_step()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def make_mlp_pipeline(estimator) -> Pipeline:
    """Same wrapper, compact block for the neural network."""
    return Pipeline(
        steps=[
            ("features", enrich_step()),
            ("preprocess", build_mlp_preprocessor()),
            ("model", estimator),
        ]
    )


def make_basic_pipeline(estimator) -> Pipeline:
    """Same wrapper, but with the naive raw one-hot feature block."""
    return Pipeline(
        steps=[
            ("features", enrich_step()),
            ("preprocess", build_basic_preprocessor()),
            ("model", estimator),
        ]
    )


class enrich_step:  # noqa: N801 - sklearn transformer naming style
    """Tiny stateless transformer so the whole stack can live in one Pipeline."""

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002
        return {}

    def set_params(self, **params):  # noqa: ARG002
        return self

    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return enrich(frame)

    def fit_transform(self, X, y=None):  # noqa: ARG002
        return self.transform(X)
