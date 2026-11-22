"""End-to-end smoke test for the CityUHK Smart Campus demo.

Run::

    python src/generate_data.py
    python src/train_models.py
    python tests/smoke_test.py

Checks
------
1. dataset shape / privacy-safe schema,
2. both selected models beat both baselines on the held-out split,
3. the serving path (feature frame -> Ridge + MLP) returns sane values,
4. the corridor walking network matches the surveyed routes,
5. the food-category filter matches the real canteen menus,
6. the map geometry stays inside the building footprints,
7. the Streamlit dashboard renders without raising (via ``AppTest``).
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
    CAT_ASIAN,
    CAT_NOODLE,
    CAT_RICE,
    CAT_WESTERN,
    CONGESTION_LEVELS,
    CROWD_PENALTY,
    FEATURE_COLUMNS,
    N_RECORDS,
    NETWORK_COORDS,
    ORIGIN_LABELS,
    STUDY_GOAL,
    ZONE_BY_ID,
    canteen_dishes,
    expected_dismissals,
    from_minutes,
    walk_time,
    zones_for_category,
    zones_for_goal,
)

MODEL_DIR = PROJECT_ROOT / "models"
DATA_CSV = PROJECT_ROOT / "data" / "cityuhk_campus_data.csv"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def buildings_of(zone_ids: list[str]) -> set[str]:
    return {ZONE_BY_ID[z]["building"] for z in zone_ids}


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
        for zone_id in zones_for_category(CAT_RICE):
            rows.append(
                {
                    "week": 3,
                    "weekday": "Wed",
                    "time_slot": from_minutes(minute),
                    "zone_id": zone_id,
                    "service_type": ZONE_BY_ID[zone_id]["service_type"],
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

    print("\n== 4. corridor walking network ==")
    check(ORIGIN_LABELS == ("AC1 (YEUNG 4/F LT)", "AC2 (LI)", "AC3 (LAU)", "Halls (九龙塘宿舍)"),
          f"four departure points defined: {ORIGIN_LABELS}")
    check(walk_time("AC1 (YEUNG 4/F LT)", "AC1 Canteen") == 1.5,
          "AC1 4/F -> AC1 Canteen 5/F = 1.5 min (elevator)")
    check(walk_time("AC1 (YEUNG 4/F LT)", "Library 3/F") == 1.5,
          "AC1 4/F -> Library 3/F = 1.5 min (same tower)")
    check(walk_time("AC1 (YEUNG 4/F LT)", "AC2 Canteen (Counter)") == 6.5,
          "AC1 4/F -> AC2 3/F = 6.5 min (5 min corridor + 1.5 min lift from B2)")
    check(walk_time("AC1 (YEUNG 4/F LT)", "AC3 Canteen (Counter)") == 8.0,
          "AC1 4/F -> AC3 7/F = 8.0 min (shortest path via AC1 5/F - AC3 3/F)")
    check(walk_time("AC2 (LI)", "AC3 Canteen (Counter)") == 2.5,
          "AC2 G/F -> AC3 7/F = 2.5 min (1 min corridor + 1.5 min lift)")
    check(walk_time("AC2 (LI)", "AC2 Canteen (Counter)") == 1.5, "AC2 G/F -> AC2 3/F = 1.5 min")
    check(walk_time("AC3 (LAU)", "AC3 Canteen (Counter)") == 1.5, "AC3 G/F -> AC3 7/F = 1.5 min")
    check(15.0 <= walk_time("Halls (九龙塘宿舍)", "AC1 Canteen") <= 20.0,
          f"Halls -> AC1 canteen = {walk_time('Halls (九龙塘宿舍)', 'AC1 Canteen'):.1f} min (surveyed 15-20)")
    check(10.0 <= walk_time("Halls (九龙塘宿舍)", "AC2 Canteen (Counter)") <= 15.0,
          f"Halls -> AC2 canteen = {walk_time('Halls (九龙塘宿舍)', 'AC2 Canteen (Counter)'):.1f} min")
    check(10.0 <= walk_time("Halls (九龙塘宿舍)", "AC3 Canteen (Counter)") <= 15.0,
          f"Halls -> AC3 canteen = {walk_time('Halls (九龙塘宿舍)', 'AC3 Canteen (Counter)'):.1f} min")

    for origin in ORIGIN_LABELS:
        for goal in zones_for_goal(STUDY_GOAL) + zones_for_category(CAT_RICE):
            total = walk_time(origin, zone_id=goal) + 2.0 + CROWD_PENALTY["Low"]
            if total <= 0:
                check(False, f"route total is positive ({origin} -> {goal})")

    print("\n== 5. food categories / menus ==")
    check(buildings_of(zones_for_category(CAT_RICE)) == {"YEUNG", "LI", "LAU"},
          "Rice & Quick Meals reachable from all three canteens")
    check(buildings_of(zones_for_category(CAT_NOODLE)) == {"YEUNG", "LI", "LAU"},
          "Noodles & Soup reachable from all three canteens")
    check(buildings_of(zones_for_category(CAT_WESTERN)) == {"YEUNG", "LI"},
          "Western & Fast Food only at AC1 and AC2")
    check(buildings_of(zones_for_category(CAT_ASIAN)) == {"YEUNG", "LAU"},
          "Asian & Curry only at AC1 and AC3")

    ac1 = canteen_dishes("YEUNG", CAT_RICE) + canteen_dishes("YEUNG", CAT_NOODLE) \
        + canteen_dishes("YEUNG", CAT_WESTERN) + canteen_dishes("YEUNG", CAT_ASIAN)
    check(len(set(ac1)) == 7, f"AC1 Canteen lists 7 dishes (got {len(set(ac1))})")
    check({"烧味", "车仔米线", "两餸饭", "拉面 & 牛肉饭", "Pizza", "炸鸡", "东南亚菜"} == set(ac1),
          f"AC1 dishes match the field menu: {sorted(ac1)}")

    ac2 = set(canteen_dishes("LI", CAT_RICE)) | set(canteen_dishes("LI", CAT_NOODLE)) \
        | set(canteen_dishes("LI", CAT_WESTERN))
    check(ac2 == {"两餸饭", "米线", "麻辣烫", "饺子", "意粉 & 焗饭"}, f"AC2 dishes match: {sorted(ac2)}")

    ac3 = set(canteen_dishes("LAU", CAT_RICE)) | set(canteen_dishes("LAU", CAT_NOODLE)) \
        | set(canteen_dishes("LAU", CAT_ASIAN))
    check(len(ac3) == 3, f"AC3 dishes match: {sorted(ac3)}")
    check(not canteen_dishes("LAU", CAT_WESTERN), "AC3 has no Western & Fast Food category")

    print("\n== 6. map geometry ==")
    for zone_id, zone in ZONE_BY_ID.items():
        box = BUILDING_BOXES[zone["building"]]
        x, y = zone["coords"]
        check(box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"],
              f"{zone_id} bubble sits inside its building footprint")
    check(set(NETWORK_COORDS) == {"YEUNG", "LI", "LAU", "HALLS"},
          "connection map has the three towers plus the Halls")

    print("\n== 7. streamlit dashboard ==")
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
