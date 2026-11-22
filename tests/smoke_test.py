"""End-to-end smoke test for the CityUHK Smart Campus demo.

Run::

    python src/generate_data.py
    python src/train_models.py
    python tests/smoke_test.py
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
    CAT_ASIAN,
    CAT_NOODLE,
    CAT_RICE,
    CAT_WESTERN,
    CONGESTION_LEVELS,
    DAY_END_MIN,
    DAY_START_MIN,
    FLOOR_CHOICES,
    FOOD_CATEGORIES,
    HALLS,
    OBSERVED_SLOTS,
    TARGET_CONGESTION,
    TARGET_WAIT,
    WEEKDAYS,
    ZONE_BY_ID,
    ZONE_IDS,
    canteen_dishes,
    is_service_open,
    opening_label,
    route_steps,
    walk_time,
    zones_for_category,
)
from src.i18n import LANGS, UI, day_name, dish_name, level_name, tr, zone_name

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
    print("== 1. dataset covers the whole week and the whole day ==")
    df = pd.read_csv(DATA_CSV)
    check(set(df["weekday"].unique()) == set(WEEKDAYS), "all seven weekdays present")
    check(df["time_slot"].nunique() == len(OBSERVED_SLOTS),
          f"{len(OBSERVED_SLOTS)} snapshots per day (07:30-21:30)")
    check(df["time_slot"].min() == "07:30" and df["time_slot"].max() == "21:30",
          f"day window {df['time_slot'].min()}–{df['time_slot'].max()}")
    check(list(df.columns) == [
        "week", "weekday", "time_slot", "zone_id", "service_type",
        "zone_capacity", "nearby_class_dismissals", TARGET_WAIT, TARGET_CONGESTION,
    ], "schema is features + two labels only")
    check(set(df[TARGET_CONGESTION]) == set(CONGESTION_LEVELS), "three classes present")
    check((df[TARGET_WAIT] > 0).all(), "all wait times positive")
    weekend = df["weekday"].isin(["Sat", "Sun"])
    check(df.loc[weekend, TARGET_WAIT].mean() < df.loc[~weekend, TARGET_WAIT].mean(),
          "weekend demand is lower than weekday demand")
    sun_ac3 = df[(df["weekday"] == "Sun") & df["zone_id"].str.startswith("AC3")]
    check(len(sun_ac3) == 0, "AC3 canteen has no Sunday observations (closed)")

    print("\n== 2. models still beat both baselines ==")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    selected = metrics["metrics"]["Ridge + MLPClassifier (selected)"]
    baselines = [v for k, v in metrics["metrics"].items() if v["kind"] == "baseline"]
    check(len(baselines) == 2, "two baselines recorded")
    check(all(selected["mae_wait_mins"] < b["mae_wait_mins"] for b in baselines),
          f"Ridge MAE {selected['mae_wait_mins']:.3f} beats both baselines")
    check(all(selected["macro_f1"] > b["macro_f1"] for b in baselines),
          f"MLP macro F1 {selected['macro_f1']:.3f} beats both baselines")
    check(selected["latency_ms_per_row"] < 50.0,
          f"single-row inference {selected['latency_ms_per_row']:.3f} ms")

    print("\n== 3. serving path ==")
    ridge = joblib.load(MODEL_DIR / "ridge_model.joblib")
    mlp = joblib.load(MODEL_DIR / "mlp_model.joblib")
    frame = pd.DataFrame(
        [
            {
                "week": 3,
                "weekday": "Wed",
                "time_slot": slot,
                "zone_id": zone_id,
                "service_type": ZONE_BY_ID[zone_id]["service_type"],
                "zone_capacity": ZONE_BY_ID[zone_id]["capacity"],
                "nearby_class_dismissals": 120,
            }
            for slot in ("11:45", "12:00", "18:30")
            for zone_id in zones_for_category(CAT_RICE)
        ]
    )
    wait = ridge.predict(frame)
    proba = mlp.predict_proba(frame)
    check(len(wait) == len(frame), "Ridge returns one prediction per row")
    check(np.isfinite(wait).all() and (wait >= -0.5).all(), "predictions finite and non-negative")
    check(proba.shape[1] == 3, "MLP returns three class probabilities")
    check(np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "probabilities sum to 1 (softmax)")

    print("\n== 4. floor-aware walking network ==")
    check(walk_time("YEUNG", "4F", "AC1 Canteen") == 1.5, "AC1 4/F -> AC1 canteen 5/F = 1.5 min")
    check(walk_time("YEUNG", "4F", "AC2 Canteen (Counter)") == 6.5,
          "AC1 4/F -> AC2 3/F = 6.5 min (corridor 5 + lift 1.5)")
    check(walk_time("YEUNG", "4F", "AC3 Canteen (Counter)") == 8.0,
          "AC1 4/F -> AC3 7/F = 8.0 min")
    check(walk_time("LI", "G", "AC3 Canteen (Counter)") == 2.5, "AC2 G/F -> AC3 7/F = 2.5 min")
    check(walk_time("HALLS", "G", "AC1 Canteen") >= 15.0, "Halls -> AC1 respects the 15-20 min range")
    # A high floor with no graph anchor still costs the extra lift ride.
    check(walk_time("LAU", "12F", "AC1 Canteen") == 8.0, "AC3 12/F -> AC1 5/F accounts for the lift")
    check(walk_time("LAU", "7F", "AC3 Canteen (Counter)") == 0.0, "standing on 7/F = 0 min walk")

    minutes, steps = route_steps("YEUNG", "4F", "AC3 Canteen (Counter)")
    kinds = [s["kind"] for s in steps]
    check(kinds == ["lift", "corridor", "lift"],
          f"route AC1 4/F -> AC3 7/F is lift -> corridor -> lift (got {kinds})")

    for building, floors in FLOOR_CHOICES.items():
        check(len(floors) >= 8, f"{building} exposes {len(floors)} selectable floors")
    check(len(FLOOR_CHOICES["LAU"]) == 21, "AC3 offers G/F + 1-20/F")

    print("\n== 5. opening hours ==")
    check(is_service_open("AC1 Canteen", "Sun", 12 * 60), "AC1 canteen open on Sunday noon")
    check(not is_service_open("AC3 Canteen (Counter)", "Sun", 12 * 60), "AC3 canteen closed on Sunday")
    check(is_service_open("AC3 Canteen (Counter)", "Sat", 12 * 60), "AC3 canteen open on Saturday")
    check(not is_service_open("AC1 Canteen", "Wed", 7 * 60 + 15), "canteen closed before 07:30")
    check(not is_service_open("AC1 Canteen", "Wed", 21 * 60), "canteen closed from 21:00")
    check(is_service_open("Library 3/F", "Sun", 21 * 60 + 30), "study areas stay open late")
    check(opening_label("AC1 Canteen") == "07:30–21:00", "opening label renders the window")

    print("\n== 6. food categories ==")
    check(buildings_of(zones_for_category(CAT_RICE)) == {"YEUNG", "LI", "LAU"}, "rice everywhere")
    check(buildings_of(zones_for_category(CAT_NOODLE)) == {"YEUNG", "LI", "LAU"}, "noodles everywhere")
    check(buildings_of(zones_for_category(CAT_WESTERN)) == {"YEUNG", "LI"}, "western only AC1/AC2")
    check(buildings_of(zones_for_category(CAT_ASIAN)) == {"YEUNG", "LAU"}, "asian only AC1/AC3")
    check(set(canteen_dishes("YEUNG", CAT_RICE)) == {"bbq", "two_dish"}, "AC1 rice dishes")
    check(not canteen_dishes("LAU", CAT_WESTERN), "AC3 has no western dishes")

    print("\n== 7. i18n completeness ==")
    check(len(LANGS) == 3, "three locales")
    base_keys = set(UI["en"])
    for lang in LANGS:
        missing = base_keys - set(UI[lang])
        extra = set(UI[lang]) - base_keys
        check(not missing and not extra, f"{lang} has exactly the English key set (missing {sorted(missing)[:3]})")
    for lang in LANGS:
        for key in ("app.title", "res.title", "st.closed", "m.data.src", "in.location"):
            check(bool(tr(lang, key)) and not tr(lang, key).startswith(key), f"{lang} resolves {key}")
    # No Chinese characters may leak into the English locale.
    def has_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    leaks = [k for k, v in UI["en"].items() if has_cjk(v)]
    check(not leaks, f"English UI has no CJK characters (leaks: {leaks[:3]})")
    check(day_name("zh_Hant", "Sun") == "週日", "weekday localises")
    check(level_name("zh_Hans", "High") == "爆满", "congestion level localises")
    check("食堂" in zone_name("zh_Hans", "AC1 Canteen"), "zone name localises")
    check(dish_name("en", "bbq") == "Siu Mei (BBQ)", "dish name localises")

    print("\n== 8. dashboard renders ==")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=240)
    app.run()
    check(not app.exception, f"app.py runs without exception ({[str(e.value) for e in app.exception][:2]})")
    check(len(app.tabs) >= 2, f"two tabs rendered (got {len(app.tabs)})")
    check(len(app.selectbox) >= 3, f"input widgets present ({len(app.selectbox)} selectboxes)")
    check(not any(s.label == "Teaching Week" for s in app.selectbox), "no teaching-week control")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
