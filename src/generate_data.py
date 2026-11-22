"""Synthetic-but-realistic CityUHK campus crowd dataset generator.

Run::

    python src/generate_data.py

Writes ``data/cityuhk_campus_data.csv`` — aggregate observations covering the
**whole service day and the whole week**:

    3 teaching weeks x 7 weekdays x 18 fifteen-minute snapshots (07:30-21:30)

Every snapshot only records zones that are actually **open** at that moment:

* canteens serve 07:30-21:00 (AC3 Bistro is closed on Sundays),
* study areas follow the library day (07:30-22:00),
* at most 8 zones are surveyed per snapshot (the survey team rotates).

Each row is one aggregate observation:

    week, weekday, time_slot, zone_id, service_type, zone_capacity,
    nearby_class_dismissals, wait_time_mins, congestion_level

The wait time comes from the stopwatch bottleneck decomposition of each canteen
(order screen / food pickup / find a seat / utensils) scaled by an occupancy
load factor.  The congestion label is derived independently from the
occupancy-to-capacity ratio (Low < 60 %, Medium 60-85 %, High > 85 %).

Weekend demand is deliberately much lower than weekdays, and there are no
class-dismissal waves on Saturday or Sunday.

Privacy: aggregated counts and durations only -- no identity, no device data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.campus_config import (  # noqa: E402
    FEATURE_COLUMNS,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MAX_ZONES_PER_SLOT,
    OBSERVED_SLOTS,
    RANDOM_SEED,
    TARGET_CONGESTION,
    TARGET_WAIT,
    WEEKDAYS,
    WEEKS,
    ZONES,
    demand_pressure,
    is_service_open,
    to_minutes,
)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "cityuhk_campus_data.csv"

# Day-level mood: Mondays busiest, Fridays lighter, weekends much quieter.
WEEKDAY_FACTOR = {
    "Mon": 1.05, "Tue": 1.03, "Wed": 1.00, "Thu": 0.98,
    "Fri": 0.94, "Sat": 0.62, "Sun": 0.55,
}
WEEK_FACTOR = {1: 1.00, 2: 1.03, 3: 0.97}
WEEKEND = ("Sat", "Sun")


def class_dismissals(minute_of_day: float, weekday: str, rng: np.random.Generator) -> int:
    """Students released by nearby lectures in the last 10 minutes.

    Weekdays have the two sharp non-linear spikes at 11:50 and 12:50; the
    weekend has no timetable, so only background footfall remains.
    """
    if weekday in WEEKEND:
        return int(max(0, rng.poisson(5)))
    hour = minute_of_day / 60.0
    amplitude = float(np.exp(rng.normal(0.0, 0.26)))
    spike_a = 190.0 * amplitude * np.exp(-(((hour - 11.833) / 0.25) ** 2))
    spike_b = 135.0 * amplitude * np.exp(-(((hour - 12.833) / 0.25) ** 2))
    background = float(rng.poisson(6))
    value = background + spike_a + spike_b + float(rng.normal(0.0, 5.0))
    return int(max(0, round(value)))


def load_factor(ratio: float) -> float:
    """Occupancy ratio -> 0..1 phase scaling (0.30 idle, 0.92 saturated)."""
    return float(np.clip((ratio - 0.30) / 0.62, 0.0, 1.0))


def occupancy_ratio(
    pressure: float,
    dismissals: int,
    capacity: int,
    popularity: float,
    mood: float,
    rng: np.random.Generator,
) -> float:
    """People-inside / seats.

    Two additive demand sources: a smooth demand curve and a **saturating
    surge** driven by the dismissal wave.  A zone absorbs the wave until about
    75 % of its seat capacity has been dumped in front of it; past that point
    occupancy climbs steeply.  That is a genuine non-linear interaction between
    two raw features (headcount x capacity), which a plain linear model cannot
    express while a one-hidden-layer network can.
    """
    critical = 0.75 * capacity
    steepness = 0.18 * capacity
    surge = 0.55 * popularity / (1.0 + np.exp(-(dismissals - critical) / steepness))
    ratio = 0.28 + 0.60 * pressure * popularity * mood + surge + float(rng.normal(0.0, 0.04))
    return float(np.clip(ratio, 0.10, 1.35))


def congestion_label(ratio: float) -> str:
    if ratio < LOW_THRESHOLD:
        return "Low"
    if ratio <= HIGH_THRESHOLD:
        return "Medium"
    return "High"


def wait_time(zone: dict, factor: float, rng: np.random.Generator) -> float:
    """Sum of the stopwatch-measured bottleneck phases at the current load."""
    total = 0.0
    for _label, low, peak in zone["phases"]:
        floor = low * 0.35
        value = floor + (peak - floor) * factor
        value *= float(np.exp(rng.normal(0.0, 0.13)))
        total += max(value, 0.05)
    total = max(total + float(rng.normal(0.0, 0.18)), 0.05)
    return round(total, 2)


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    zones = list(ZONES)
    rows: list[dict] = []
    day_index = 0

    for week in WEEKS:
        for weekday in WEEKDAYS:
            mood = WEEK_FACTOR[week] * WEEKDAY_FACTOR[weekday] * float(np.exp(rng.normal(0.0, 0.09)))

            for slot in OBSERVED_SLOTS:
                minute = to_minutes(slot)
                pressure = demand_pressure(minute)
                dismissals = class_dismissals(minute, weekday, rng)

                open_zones = [z for z in zones if is_service_open(z["zone_id"], weekday, minute)]
                if len(open_zones) > MAX_ZONES_PER_SLOT:
                    resting = day_index % len(open_zones)
                    open_zones = [z for i, z in enumerate(open_zones) if i != resting]

                for zone in open_zones:
                    ratio = occupancy_ratio(
                        pressure=pressure,
                        dismissals=dismissals,
                        capacity=zone["capacity"],
                        popularity=zone["popularity"],
                        mood=mood,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "week": week,
                            "weekday": weekday,
                            "time_slot": slot,
                            "zone_id": zone["zone_id"],
                            "service_type": zone["service_type"],
                            "zone_capacity": zone["capacity"],
                            "nearby_class_dismissals": dismissals,
                            TARGET_WAIT: wait_time(zone, load_factor(ratio), rng),
                            TARGET_CONGESTION: congestion_label(ratio),
                        }
                    )
            day_index += 1

    ordered = [
        "week", "weekday", "time_slot",
        *[c for c in FEATURE_COLUMNS if c not in ("week", "weekday", "time_slot")],
        TARGET_WAIT, TARGET_CONGESTION,
    ]
    return pd.DataFrame(rows, columns=ordered)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()

    assert df["zone_id"].nunique() == len(ZONES), "every zone must appear"
    assert df[TARGET_CONGESTION].isin(["Low", "Medium", "High"]).all()
    assert (df[TARGET_WAIT] > 0).all()
    assert set(df["weekday"].unique()) == set(WEEKDAYS), "all seven weekdays covered"

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[generate_data] wrote {len(df)} rows -> {OUTPUT_CSV}")
    print(f"[generate_data] zones            : {df['zone_id'].nunique()}")
    print(f"[generate_data] weeks             : {sorted(df['week'].unique())}")
    print(f"[generate_data] weekdays          : {list(WEEKDAYS)}")
    print(f"[generate_data] snapshots/day     : {len(OBSERVED_SLOTS)} -> {OBSERVED_SLOTS[0]} … {OBSERVED_SLOTS[-1]}")
    print("[generate_data] congestion mix   :")
    print(df[TARGET_CONGESTION].value_counts().to_string())
    print("[generate_data] rows per weekday :")
    print(df.groupby("weekday").size().to_string())
    print("[generate_data] mean wait by weekday (min):")
    print(df.groupby("weekday")[TARGET_WAIT].mean().round(2).to_string())
    print("[generate_data] weekend vs weekday mean wait:")
    weekend = df["weekday"].isin(["Sat", "Sun"])
    print(f"  weekday {df.loc[~weekend, TARGET_WAIT].mean():.2f} min | "
          f"weekend {df.loc[weekend, TARGET_WAIT].mean():.2f} min")
    print("[generate_data] AC3 rows on Sunday (should be 0 for the canteen):")
    sun_ac3 = df[(df["weekday"] == "Sun") & (df["zone_id"].str.startswith("AC3"))]
    print(f"  {len(sun_ac3)} rows")


if __name__ == "__main__":
    main()
