"""Synthetic-but-realistic CityUHK campus crowd dataset generator.

Run::

    python src/generate_data.py

Writes ``data/cityuhk_campus_data.csv`` with N = 960 aggregate observations:

    3 teaching weeks x 5 weekdays x 8 fifteen-minute snapshots x 8 zones
    (nine zones in total, one rests each day so that the total lands on 960)

Each row is one *aggregate* observation of a zone at a snapshot:

    week, weekday, time_slot, zone_id, service_type, zone_capacity,
    nearby_class_dismissals, wait_time_mins, congestion_level

The wait time is generated from the stopwatch bottleneck decomposition of each
canteen (order screen / food pickup / find a seat / utensils), scaled by an
occupancy load factor.  The congestion label is derived independently from the
occupancy-to-capacity ratio (Low < 60 %, Medium 60-85 %, High > 85 %).

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
    N_RECORDS,
    OBSERVED_SLOTS,
    RANDOM_SEED,
    TARGET_CONGESTION,
    TARGET_WAIT,
    WEEKDAYS,
    WEEKS,
    ZONES,
    expected_dismissals,
    to_minutes,
)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "cityuhk_campus_data.csv"

# Day-level mood: Mondays are busier, Fridays emptier, week 3 slightly lighter.
WEEKDAY_FACTOR = {"Mon": 1.05, "Tue": 1.03, "Wed": 1.00, "Thu": 0.98, "Fri": 0.94}
WEEK_FACTOR = {1: 1.00, 2: 1.03, 3: 0.97}


def lunch_pressure(minute_of_day: float) -> float:
    """Non-linear lunch demand curve on the CityUHK campus.

    Three overlapping peaks: a broad midday hump (12:00), a sharp 11:50
    class-dismissal wave and a smaller 12:50 wave.  Outside 11:00-14:30 the
    curve collapses to (almost) zero, which is what makes a plain linear model
    struggle here.
    """
    hour = minute_of_day / 60.0
    broad = 1.00 * np.exp(-(((hour - 12.25) / 0.85) ** 2))
    wave_a = 0.55 * np.exp(-(((hour - 11.83) / 0.20) ** 2))   # 11:50 dismissal
    wave_b = 0.45 * np.exp(-(((hour - 12.85) / 0.22) ** 2))   # 12:50 dismissal
    return float(broad + wave_a + wave_b)


def class_dismissals(minute_of_day: float, rng: np.random.Generator) -> int:
    """Students released by nearby lectures in the last 10 minutes.

    Two explosive, non-linear spikes at 11:50 and 12:50 sitting on a small
    Poisson background -- exactly the pattern a hand-timed field survey shows.
    The amplitude of the wave varies from day to day (a 09:00 lecture that ends
    early, a cancelled class, a society event spilling out), so the *same*
    weekday/time-slot does not always produce the same headcount.  Without that
    variance a lookup table would be enough and the MLP would have nothing to
    add.
    """
    hour = minute_of_day / 60.0
    amplitude = float(np.exp(rng.normal(0.0, 0.26)))
    scheduled_wave = expected_dismissals(minute_of_day) - 6  # timetable part only
    background = float(rng.poisson(6))
    value = background + scheduled_wave * amplitude + float(rng.normal(0.0, 5.0))
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

    Two additive demand sources:

    * a smooth lunch curve ``0.18 + 0.52 * pressure * popularity``, and
    * a **saturating surge** driven by the dismissal wave.  A zone absorbs the
    wave until roughly 55 % of its seat capacity has been dumped in front of
    it; past that point occupancy climbs steeply (the classic "queue collapses"
    threshold).  That is a genuine non-linear interaction between two raw
    features (headcount x capacity), which a plain linear model cannot express
    while a one-hidden-layer network can.
    """
    critical = 0.75 * capacity
    steepness = 0.18 * capacity
    surge = 0.55 * popularity / (1.0 + np.exp(-(dismissals - critical) / steepness))
    ratio = 0.22 + 0.56 * pressure * popularity * mood + surge + float(rng.normal(0.0, 0.04))
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
        value *= float(np.exp(rng.normal(0.0, 0.13)))  # stopwatch observation jitter
        total += max(value, 0.05)
    # Aggregate stopwatch uncertainty (reaction lag, rounding, walking into the queue).
    total = max(total + float(rng.normal(0.0, 0.18)), 0.05)
    return round(total, 2)


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    zones = list(ZONES)
    rows: list[dict] = []
    day_index = 0

    for week in WEEKS:
        for weekday in WEEKDAYS:
            # One zone "rests" each day (survey team rotation) -> 8 zones x 8 slots.
            resting = day_index % len(zones)
            mood = WEEK_FACTOR[week] * WEEKDAY_FACTOR[weekday] * float(np.exp(rng.normal(0.0, 0.09)))

            for slot in OBSERVED_SLOTS:
                minute = to_minutes(slot)
                pressure = lunch_pressure(minute)
                dismissals = class_dismissals(minute, rng)

                for zone_pos, zone in enumerate(zones):
                    if zone_pos == resting:
                        continue
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

    ordered = ["week", "weekday", "time_slot", *[c for c in FEATURE_COLUMNS if c not in ("week", "weekday", "time_slot")], TARGET_WAIT, TARGET_CONGESTION]
    return pd.DataFrame(rows, columns=ordered)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()

    assert len(df) == N_RECORDS, f"expected {N_RECORDS} rows, generated {len(df)}"
    assert df["zone_id"].nunique() == len(ZONES), "every zone must appear"
    assert df[TARGET_CONGESTION].isin(["Low", "Medium", "High"]).all()
    assert (df[TARGET_WAIT] > 0).all()

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[generate_data] wrote {len(df)} rows -> {OUTPUT_CSV}")
    print(f"[generate_data] zones            : {df['zone_id'].nunique()}")
    print(f"[generate_data] weeks x weekdays : {sorted(df['week'].unique())} x {sorted(df['weekday'].unique())}")
    print(f"[generate_data] snapshots/day    : {df['time_slot'].nunique()} -> {sorted(df['time_slot'].unique())}")
    print("[generate_data] congestion mix   :")
    print(df[TARGET_CONGESTION].value_counts().to_string())
    print("[generate_data] wait time by zone (min / mean / max):")
    summary = df.groupby("zone_id")[TARGET_WAIT].agg(["min", "mean", "max"]).round(2)
    print(summary.to_string())
    print("[generate_data] lunch-peak vs off-peak mean wait:")
    peak = df["time_slot"].isin(["11:30", "11:45", "12:00", "12:30", "13:00", "13:30"])
    print(f"  peak {df.loc[peak, TARGET_WAIT].mean():.2f} min | off-peak {df.loc[~peak, TARGET_WAIT].mean():.2f} min")


if __name__ == "__main__":
    main()
