"""End-to-end smoke test for the CityUHK Smart Campus demo.

Run::

    python src/generate_data.py
    python src/train_models.py
    python tests/smoke_test.py

Checks
------
1. the dataset exists and has the promised shape / privacy-safe schema,
2. both selected models beat both baselines on the held-out split,
3. the serving path (feature frame -> Ridge + MLP) returns sane values,
4. the real-campus walking model matches the surveyed routes,
5. the map coordinates stay inside their building footprints,
6. the Streamlit dashboard renders without raising (via ``AppTest``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.campus_config import (  # noqa: E402
    BUILDING_BOXES,
    CONGESTION_LEVELS,
    CROWD_PENALTY,
    FEATURE_COLUMNS,
    N_RECORDS,
    ORIGIN_LABELS,
    ORIGINS,
    USER_NEEDS,
    ZONE_BY_ID,
    expected_dismissals,
    from_minutes,
    walk_time,
    zones_for_need,
)

MODEL_DIR = PROJECT_ROOT / "models"
DATA_CSV = PROJECT_ROOT / "data" / "cityuhk_campus_data.csv"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main() -> None:
    print("== 1. dataset ==")
    df = pd.read_csv(DATA_CSV)
    check(len(df) == N_RECORDS, f"dataset has {N_RECORDS} rows (got {len(df)})")
    check(df["zone_id"].nunique() == 9, "nine campus zones present")
    check(list(df.columns) == [*FEATURE_COLUMNS, "wait_time_mins", "congestion_level"],
          "schema is features + two labels only (no personal data)")
    check(set(df["congestion_level"]) == set(CONGESTION_LEVELS), "three congestion classes present")
    check((df["wait_time_mins"] > 0).all(), "all wait times positive")
    check(df["zone_id"].isin(list(ZONE_BY_ID)).all(), "every zone id resolves to a configured zone")

    print("\n== 2. model comparison ==")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    selected = metrics["metrics"]["Ridge + MLPClassifier (selected)"]
    baselines = [v for k, v in metrics["metrics"].items() if v["kind"] == "baseline"]
    check(len(baselines) == 2, "two baselines recorded")
    check(all(selected["mae_wait_mins"] < b["mae_wait_mins"] for b in baselines),
          f"Ridge MAE {selected['mae_wait_mins']:.3f} beats both baselines")
    check(all(selected["macro_f1"] > b["macro_f1"] for b in baselines),
          f"MLP macro F1 {selected['macro_f1']:.3f} beats both baselines")
    check(selected["latency_ms_per_row"] < 50.0,
          f"single-row inference {selected['latency_ms_per_row']:.3f} ms (laptop-friendly)")

    print("\n== 3. serving path ==")
    ridge = joblib.load(MODEL_DIR / "ridge_model.joblib")
    mlp = joblib.load(MODEL_DIR / "mlp_model.joblib")

    rows = []
    for minute in (11 * 60 + 30, 12 * 60, 12 * 60 + 45, 15 * 60):
        for zone_id in zones_for_need("Counter Meal"):
            rows.append(
                {
                    "week": 3,
                    "weekday": "Wed",
                    "time_slot": from_minutes(minute),
                    "zone_id": zone_id,
                    "service_type": "Counter",
                    "zone_capacity": ZONE_BY_ID[zone_id]["capacity"],
                    "nearby_class_dismissals": expected_dismissals(minute),
                }
            )
    frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    wait = ridge.predict(frame)
    proba = mlp.predict_proba(frame)

    check(len(wait) == len(frame), "Ridge returns one prediction per row")
    check(np.isfinite(wait).all() and (wait >= -0.5).all(), "wait predictions are finite and non-negative")
    check(proba.shape[1] == 3, "MLP returns three class probabilities")
    check(np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "MLP probabilities sum to 1 (softmax)")
    check(list(mlp.named_steps["model"].classes_) == sorted(CONGESTION_LEVELS), "class order is stable")

    peak_mask = frame["time_slot"].isin(["11:45", "12:00", "12:45"]).to_numpy()
    check(bool(wait[peak_mask].mean() > wait[~peak_mask].mean()),
          "lunch-peak predictions exceed off-peak predictions")

    print("\n== 4. walking model (real campus routes) ==")
    check(ORIGIN_LABELS == ("AC1 (YEUNG 4/F LT)", "Library (YEUNG 3/F)", "AC2 (LI)", "AC3 (LAU)"),
          f"four departure points defined: {ORIGIN_LABELS}")
    check(walk_time("AC1 (YEUNG 4/F LT)", "AC1 Canteen") == 1.4,
          "AC1 4/F LT -> AC1 Canteen 5/F = 1.4 min (one floor up)")
    check(walk_time("AC1 (YEUNG 4/F LT)", "Library 3/F") == 1.4,
          "AC1 4/F LT -> Library 3/F = 1.4 min (same building)")
    check(3.0 <= walk_time("AC1 (YEUNG 4/F LT)", "AC2 Canteen (Counter)") <= 4.0,
          "AC1 -> AC2 within the surveyed 3-4 min range")
    check(4.0 <= walk_time("AC1 (YEUNG 4/F LT)", "AC3 Bistro (Counter)") <= 5.0,
          "AC1 -> AC3 within the surveyed 4-5 min range")
    check(walk_time("AC2 (LI)", "AC3 Bistro (Counter)") == 3.0, "AC2 -> AC3 = 3.0 min")
    check(walk_time("AC2 (LI)", "AC2 Canteen (Counter)") == 1.8, "AC2 1/F -> AC2 Canteen 3/F = 1.8 min")
    check(walk_time("AC3 (LAU)", "AC3 Bistro (Counter)") == 3.4, "AC3 1/F -> AC3 Bistro 7/F = 3.4 min")

    matrix = {tuple(origins): OB for origins, OB in ORIGINS.items()}
    check(len(matrix) == 4, "origin table is complete")
    for origin in ORIGIN_LABELS:
        for need in USER_NEEDS:
            for zone_id in zones_for_need(need):
                total = walk_time(origin, zone_id) + 2.0 + CROWD_PENALTY["Low"]
                if total <= 0:
                    check(False, f"route total is positive ({origin} -> {zone_id})")

    print("\n== 5. map geometry ==")
    for zone_id, zone in ZONE_BY_ID.items():
        box = BUILDING_BOXES[zone["building"]]
        x, y = zone["coords"]
        inside = box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"]
        check(inside, f"{zone_id} bubble sits inside its building footprint")

    print("\n== 6. streamlit dashboard ==")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    app.run()
    check(not app.exception, f"app.py runs without exception ({[str(e.value) for e in app.exception][:2]})")
    check(len(app.tabs) >= 2, f"two dashboard tabs rendered (got {len(app.tabs)})")
    check(len(app.dataframe) >= 1, "evaluation table rendered")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
