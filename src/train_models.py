"""Train, tune, compare and persist the CityUHK crowd models.

Run::

    python src/train_models.py

Pipeline alignment with the GE2340 course material
--------------------------------------------------
1. Time-ordered splitting (no shuffling, no future leakage):
   week 1-2 -> train (640 rows) | week 3 first half -> validation (160) |
   week 3 second half -> test (160, untouched until the very end).
   That is a 66.7 % / 16.7 % / 16.7 % chronological split (the requested
   70/15/15 split with a whole-number-of-days boundary).
2. Two baselines:
   * Historical Slot Average  (weekday x time-slot mean / majority class)
   * LinearRegression + LogisticRegression (softmax over cross-entropy)
3. Selected deployable models:
   * Ridge (L2)                -> wait time regression, alpha tuned on validation
   * MLPClassifier (32, ReLU)  -> congestion 3-class, Adam + L2 weight decay
4. Metrics: Wait-Time MAE (minutes), Congestion Macro F1, single-inference
   latency in milliseconds per row.
5. Artefacts: ``models/preprocessor.joblib``, ``models/ridge_model.joblib``,
   ``models/mlp_model.joblib``, ``models/metrics.json``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.campus_config import (  # noqa: E402
    CONGESTION_LEVELS,
    FEATURE_COLUMNS,
    PROJECT_NAME,
    RANDOM_SEED,
    TARGET_CONGESTION,
    TARGET_WAIT,
)
from src.features import enrich, make_basic_pipeline, make_mlp_pipeline, make_pipeline  # noqa: E402

DATA_CSV = PROJECT_ROOT / "data" / "cityuhk_campus_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"

RIDGE_ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
MLP_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
# Network-level randomness is part of the search: a 32-unit net on ~640 rows can
# land on a poor local minimum, so the seed is treated as one more
# hyper-parameter and picked on the validation split, never on the test split.
MLP_SEEDS = (7, 42, 2024)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def time_ordered_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: week 1-2 train, week 3 front validation, tail test."""
    week_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    ordered = df.assign(
        _week=df["week"].astype(int),
        _day=df["weekday"].map(week_order).astype(int),
        _slot=df["time_slot"].str.replace(":", "", regex=False).astype(int),
    ).sort_values(["_week", "_day", "_slot"], kind="mergesort")

    train = ordered[ordered["_week"].isin([1, 2])]
    week3 = ordered[ordered["_week"] == 3]
    n_val = len(week3) // 2          # first half of week 3 -> validation
    val = week3.iloc[:n_val]
    test = week3.iloc[n_val:]        # second half of week 3 -> held-out test

    return (
        train.drop(columns=["_week", "_day", "_slot"]),
        val.drop(columns=["_week", "_day", "_slot"]),
        test.drop(columns=["_week", "_day", "_slot"]),
    )


# --------------------------------------------------------------------------- #
# Baseline 1: historical slot average / majority vote
# --------------------------------------------------------------------------- #
class HistoricalSlotAverage:
    """Predict the historical mean (or majority class) of weekday x time-slot.

    This is the honest "what we would do without any ML" baseline: it only
    remembers the average of the same weekday/time-slot in the training weeks.
    Unseen keys fall back to the global mean / majority class.
    """

    def __init__(self, task: str = "regression") -> None:
        self.task = task
        self.table_: pd.Series | None = None
        self.fallback_ = None

    @staticmethod
    def _keys(X: pd.DataFrame) -> pd.Series:
        return X["weekday"].astype(str) + "|" + X["time_slot"].astype(str)

    def fit(self, X: pd.DataFrame, y) -> "HistoricalSlotAverage":
        keys = self._keys(X)
        if self.task == "regression":
            self.table_ = pd.Series(np.asarray(y, dtype=float), index=keys.to_numpy()).groupby(level=0).mean()
            self.fallback_ = float(np.mean(y))
        else:
            frame = pd.DataFrame({"key": keys.to_numpy(), "y": np.asarray(y)})
            self.table_ = frame.groupby("key")["y"].agg(lambda s: s.value_counts().idxmax())
            self.fallback_ = pd.Series(y).value_counts().idxmax()
        return self

    def predict(self, X: pd.DataFrame):
        keys = self._keys(X)
        return self.table_.reindex(keys.to_numpy()).fillna(self.fallback_).to_numpy()


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def measure_latency(predict_fn, X: pd.DataFrame, repeats: int = 25) -> float:
    """Median milliseconds per single row prediction on the test split."""
    predict_fn(X.iloc[: min(16, len(X))])
    per_row = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(X)
        per_row.append((time.perf_counter() - start) / len(X) * 1000.0)
    return float(np.median(per_row))


def score_regression(predict_fn, X, y) -> dict:
    pred = np.clip(np.asarray(predict_fn(X), dtype=float), 0.0, None)
    return {"mae_wait_mins": float(mean_absolute_error(y, pred))}


def score_classification(predict_fn, X, y) -> dict:
    pred = np.asarray(predict_fn(X))
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(CONGESTION_LEVELS), zero_division=0)),
        "accuracy": float(np.mean(pred == np.asarray(y))),
    }


def tune_ridge(train: pd.DataFrame, val: pd.DataFrame) -> tuple[float, Ridge]:
    best_alpha, best_mae, best_model = None, np.inf, None
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(Ridge(alpha=alpha, random_state=RANDOM_SEED))
        model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_WAIT])
        mae = mean_absolute_error(val[TARGET_WAIT], model.predict(val[list(FEATURE_COLUMNS)]))
        if mae < best_mae:
            best_alpha, best_mae, best_model = alpha, mae, model
    return float(best_alpha), best_model


def tune_mlp(train: pd.DataFrame, val: pd.DataFrame) -> tuple[float, int, MLPClassifier]:
    """Grid over L2 weight decay x init seed, selected on macro F1 (validation)."""
    best_alpha, best_seed, best_f1, best_model = None, None, -np.inf, None
    for alpha in MLP_ALPHAS:
        for seed in MLP_SEEDS:
            model = make_mlp_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=16,
                    learning_rate_init=0.003,
                    max_iter=6000,
                    tol=1e-6,
                    n_iter_no_change=60,
                    early_stopping=True,
                    validation_fraction=0.15,
                    shuffle=True,
                    random_state=seed,
                )
            )
            model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_CONGESTION])
            f1 = f1_score(
                val[TARGET_CONGESTION],
                model.predict(val[list(FEATURE_COLUMNS)]),
                average="macro",
                labels=list(CONGESTION_LEVELS),
                zero_division=0,
            )
            if f1 > best_f1:
                best_alpha, best_seed, best_f1, best_model = alpha, seed, f1, model
    return float(best_alpha), int(best_seed), best_model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if not DATA_CSV.exists():
        raise SystemExit(f"dataset not found: {DATA_CSV}\nrun `python src/generate_data.py` first")

    df = pd.read_csv(DATA_CSV)
    train, val, test = time_ordered_split(df)

    X_train, y_train_reg, y_train_cls = train[list(FEATURE_COLUMNS)], train[TARGET_WAIT], train[TARGET_CONGESTION]
    X_val, y_val_reg, y_val_cls = val[list(FEATURE_COLUMNS)], val[TARGET_WAIT], val[TARGET_CONGESTION]
    X_test, y_test_reg, y_test_cls = test[list(FEATURE_COLUMNS)], test[TARGET_WAIT], test[TARGET_CONGESTION]

    results: dict[str, dict] = {}

    # ---------------- Baseline 1: historical slot average ---------------- #
    slot_reg = HistoricalSlotAverage("regression").fit(X_train, y_train_reg)
    slot_cls = HistoricalSlotAverage("classification").fit(X_train, y_train_cls)
    results["Historical Slot Average (baseline)"] = {
        **score_regression(slot_reg.predict, X_test, y_test_reg),
        **score_classification(slot_cls.predict, X_test, y_test_cls),
        "latency_ms_per_row": measure_latency(slot_reg.predict, X_test),
        "kind": "baseline",
        "detail": "weekday x time-slot historical mean / majority class, global fallback",
    }

    # ---------------- Baseline 2: plain linear / logistic ---------------- #
    # "Plain" means: the raw one-hot block only (`build_basic_preprocessor`),
    # no interaction terms and no non-linear transforms.  This is the honest
    # textbook baseline that the selected models have to beat.
    lin_reg = make_basic_pipeline(LinearRegression())
    lin_reg.fit(X_train, y_train_reg)
    log_cls = make_basic_pipeline(LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_SEED))
    log_cls.fit(X_train, y_train_cls)
    results["Linear Regression + Logistic Regression (baseline)"] = {
        **score_regression(lin_reg.predict, X_test, y_test_reg),
        **score_classification(log_cls.predict, X_test, y_test_cls),
        "latency_ms_per_row": measure_latency(lin_reg.predict, X_test),
        "kind": "baseline",
        "detail": "plain one-hot block, softmax cross-entropy logistic classifier, no feature engineering",
    }

    # ---------------- Selected: Ridge + MLP ------------------------------ #
    ridge_alpha, ridge_model = tune_ridge(train, val)
    mlp_alpha, mlp_seed, mlp_model = tune_mlp(train, val)
    results["Ridge + MLPClassifier (selected)"] = {
        **score_regression(ridge_model.predict, X_test, y_test_reg),
        **score_classification(mlp_model.predict, X_test, y_test_cls),
        "latency_ms_per_row": measure_latency(ridge_model.predict, X_test),
        "kind": "selected",
        "detail": f"Ridge alpha={ridge_alpha:g} (L2, tuned on validation), "
                  f"MLP (32,) ReLU+Adam alpha={mlp_alpha:g} seed={mlp_seed}",
    }

    # ---------------- Persist artefacts ---------------------------------- #
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor = ridge_model.named_steps["preprocess"]
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(ridge_model, MODEL_DIR / "ridge_model.joblib")
    joblib.dump(mlp_model, MODEL_DIR / "mlp_model.joblib")

    ridge_mae = results["Ridge + MLPClassifier (selected)"]["mae_wait_mins"]
    best_baseline_mae = min(results[k]["mae_wait_mins"] for k in results if k != "Ridge + MLPClassifier (selected)")
    mlp_f1 = results["Ridge + MLPClassifier (selected)"]["macro_f1"]
    best_baseline_f1 = max(results[k]["macro_f1"] for k in results if k != "Ridge + MLPClassifier (selected)")

    metrics = {
        "project": PROJECT_NAME,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "file": "data/cityuhk_campus_data.csv",
            "rows": int(len(df)),
            "zones": int(df["zone_id"].nunique()),
            "features": list(FEATURE_COLUMNS),
            "targets": {"regression": TARGET_WAIT, "classification": TARGET_CONGESTION},
            "class_mix": {k: int(v) for k, v in df[TARGET_CONGESTION].value_counts().items()},
        },
        "splits": {
            "strategy": "time-ordered: week 1-2 train, week 3 first half validation, week 3 second half test",
            "train": {"rows": int(len(train)), "share": round(len(train) / len(df), 4),
                      "weeks": sorted(train["week"].unique().tolist())},
            "validation": {"rows": int(len(val)), "share": round(len(val) / len(df), 4),
                           "weeks": sorted(val["week"].unique().tolist())},
            "test": {"rows": int(len(test)), "share": round(len(test) / len(df), 4),
                     "weeks": sorted(test["week"].unique().tolist())},
        },
        "hyperparameters": {
            "ridge_alpha": ridge_alpha,
            "mlp_alpha": mlp_alpha,
            "mlp_random_state": mlp_seed,
            "mlp_hidden_layer_sizes": [32],
            "mlp_activation": "relu",
            "mlp_solver": "adam",
            "mlp_lr_init": 0.003,
            "mlp_batch_size": 16,
        },
        "metrics": results,
        "verdict": {
            "ridge_beats_baselines": bool(ridge_mae < best_baseline_mae),
            "mlp_beats_baselines": bool(mlp_f1 > best_baseline_f1),
            "best_baseline_mae": float(best_baseline_mae),
            "best_baseline_macro_f1": float(best_baseline_f1),
        },
        "limitations": [
            "Synthetic field-survey model: real queues react to weather, exams and unregistered events.",
            "Sudden rain pushes everyone indoors -> wait times can jump beyond the training range.",
            "Unlisted student activities (society fair, career talk) create demand spikes the model has never seen.",
            "Only 3 teaching weeks are covered; semester-start and exam-period behaviour differ.",
            "Per-phase bottlenecks shown in the dashboard are structure-weighted estimates, not separate predictions.",
        ],
    }

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    # ---------------- Report --------------------------------------------- #
    table = pd.DataFrame(results).T[["mae_wait_mins", "macro_f1", "accuracy", "latency_ms_per_row"]]
    table = table.astype(float).round(4)
    table.index.name = "model"
    print("\n=== Model comparison on the held-out week-3 tail (test split) ===")
    print(table.to_string())
    print(f"\nSelected Ridge alpha   : {ridge_alpha:g}")
    print(f"Selected MLP alpha/seed: {mlp_alpha:g} / {mlp_seed}")
    print(f"Split sizes            : train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Ridge MAE < baselines  : {metrics['verdict']['ridge_beats_baselines']} "
          f"({ridge_mae:.4f} vs {best_baseline_mae:.4f})")
    print(f"MLP macro F1 > baselines: {metrics['verdict']['mlp_beats_baselines']} "
          f"({mlp_f1:.4f} vs {best_baseline_f1:.4f})")
    print(f"\nSaved -> {MODEL_DIR}")

    if not (metrics["verdict"]["ridge_beats_baselines"] and metrics["verdict"]["mlp_beats_baselines"]):
        raise SystemExit(
            "selected models did not beat both baselines -- tune the feature set or the alpha grid"
        )


if __name__ == "__main__":
    main()
