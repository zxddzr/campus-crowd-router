"""Static configuration for the CityUHK Smart Campus Crowd-Balancing demo.

Everything in this module is intentionally human readable: the nine campus
zones (with their real buildings and floors), the stopwatch-derived bottleneck
structure of each canteen, the walking-time model and the crowd-penalty
constants.

Real campus geography used here
-------------------------------
* **AC1 / YEUNG** — Yeung Kin Man Academic Building, the main teaching building
  linked to Festival Walk.  4/F holds the LT lecture-theatre zone and the
  Library entrance; **5/F is the AC1 Canteen**.  The Library itself is addressed
  as ``Library (YEUNG 3/F)``.
* **AC2 / LI** — Li Dak Sum Yip Yio Chin Academic Building (classrooms and
  computer rooms); **3/F is the AC2 Canteen**.
* **AC3 / LAU** — Lau Ming Wai Academic Building; **7/F is the AC3 Bistro**
  (a.k.a. the AC3 canteen).

Walking model: inter-building doors follow the real pedestrian routes
(AC1 <-> AC2 ~3.5 min, AC1 <-> AC3 ~4.5 min, AC2 <-> AC3 ~3 min) and a change
of floor inside the same building costs 1.0 min + 0.4 min per floor.

The numbers below encode field observations taken on the CityUHK campus during
the 11:30-14:30 lunch peak.  Each canteen is modelled as a *chain of queues*
rather than a single blob, because the bottlenecks are sequential
(order screen -> food pickup -> find a seat -> utensils) and they scale
non-linearly with occupancy.

Privacy: only aggregate counters and wait times are modelled.  No student ID,
device identifier, Wi-Fi MAC address or any other personal data is used.
"""

from __future__ import annotations

import math
from typing import Final

# --------------------------------------------------------------------------- #
# Project meta
# --------------------------------------------------------------------------- #
PROJECT_NAME: Final[str] = "Smart Campus Crowd-Balancing and Space Routing Engine"
PROJECT_SUBTITLE: Final[str] = "GE2340 · City University of Hong Kong"
PROJECT_VERSION: Final[str] = "1.1.0"

# --------------------------------------------------------------------------- #
# Temporal grid
# --------------------------------------------------------------------------- #
SLOT_MINUTES: Final[int] = 15
DAY_START_MIN: Final[int] = 9 * 60      # 09:00
DAY_END_MIN: Final[int] = 17 * 60       # 17:00
LUNCH_PEAK: Final[tuple[int, int]] = (11 * 60 + 30, 14 * 60 + 30)

WEEKS: Final[tuple[int, ...]] = (1, 2, 3)
WEEKDAYS: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri")

# Eight observed 15-minute snapshots per teaching day.  Six of them sit inside
# the lunch peak (11:30-13:30) so that the 11:50 and 12:50 class-dismissal
# spikes are both captured, plus one morning and one afternoon baseline slot.
OBSERVED_SLOTS: Final[tuple[str, ...]] = (
    "09:30",  # morning baseline
    "11:30",
    "11:45",
    "12:00",  # 11:50 dismissal wave
    "12:30",
    "13:00",  # 12:50 dismissal wave
    "13:30",
    "15:30",  # afternoon baseline
)

# 3 weeks x 5 weekdays x 8 slots x 8 zones (of 9, one zone rests each day)
N_RECORDS: Final[int] = 960
RANDOM_SEED: Final[int] = 20241122

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "week",
    "weekday",
    "time_slot",
    "zone_id",
    "service_type",
    "zone_capacity",
    "nearby_class_dismissals",
)
TARGET_WAIT: Final[str] = "wait_time_mins"
TARGET_CONGESTION: Final[str] = "congestion_level"

CONGESTION_LEVELS: Final[tuple[str, str, str]] = ("Low", "Medium", "High")
CONGESTION_COLORS: Final[dict[str, str]] = {
    "Low": "#2E9E5B",
    "Medium": "#E8A317",
    "High": "#D5402B",
}

# Occupancy ratio (people inside / seats) thresholds.
LOW_THRESHOLD: Final[float] = 0.60
HIGH_THRESHOLD: Final[float] = 0.85

# Total-time penalty (minutes) attached to each predicted congestion level.
CROWD_PENALTY: Final[dict[str, float]] = {"Low": 0.0, "Medium": 1.5, "High": 4.0}

# --------------------------------------------------------------------------- #
# Buildings
# --------------------------------------------------------------------------- #
BUILDINGS: Final[tuple[str, ...]] = ("YEUNG", "LI", "LAU")

BUILDING_LABELS: Final[dict[str, str]] = {
    "YEUNG": "Yeung Kin Man Academic Building (AC1)",
    "LI": "Li Dak Sum Yip Yio Chin Academic Building (AC2)",
    "LAU": "Lau Ming Wai Academic Building (AC3)",
}
BUILDING_SHORT: Final[dict[str, str]] = {
    "YEUNG": "AC1 · YEUNG",
    "LI": "AC2 · LI",
    "LAU": "AC3 · LAU",
}

# Inter-building walking minutes between ground-level entrances (symmetric).
BASE_WALK: Final[dict[str, dict[str, float]]] = {
    "YEUNG": {"YEUNG": 0.0, "LI": 3.5, "LAU": 4.5},
    "LI": {"YEUNG": 3.5, "LI": 0.0, "LAU": 3.0},
    "LAU": {"YEUNG": 4.5, "LI": 3.0, "LAU": 0.0},
}

# Same-building floor change: fixed lift/stairs overhead + per-floor cost.
VERTICAL_BASE: Final[float] = 1.0
VERTICAL_PER_FLOOR: Final[float] = 0.4

# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
# coords = schematic 2-D position used by the dashboard map (not geo-accurate).
# phases = (label, low-value, peak-value); realised wait is
#          floor*0.35 + (peak - floor*0.35) * load_factor.
ZONES: Final[tuple[dict, ...]] = (
    {
        "zone_id": "AC1 Canteen",
        "service_type": "Counter",
        "building": "YEUNG",
        "floor": 5,
        "floor_label": "AC1 · 5/F",
        "capacity": 300,
        "popularity": 1.00,
        "coords": (0.45, 0.55),
        "phases": (
            ("Order screen queue", 2.0, 2.8),
            ("Food pickup", 3.0, 5.0),
            ("Find a seat", 1.5, 2.2),
            ("Utensils & condiments", 0.8, 1.0),
        ),
        "note": "Yeung 5/F counter: screen 2-3 min + pickup 3-5 min + seat 2 min + utensils 1 min.",
    },
    {
        "zone_id": "AC2 Canteen (Semi-Buffet)",
        "service_type": "Semi-Buffet",
        "building": "LI",
        "floor": 3,
        "floor_label": "AC2 · 3/F",
        "capacity": 260,
        "popularity": 0.88,
        "coords": (1.60, 0.30),
        "phases": (
            ("Find a seat", 1.0, 3.0),
            ("Semi-buffet serving", 1.0, 5.0),
        ),
        "note": "Li 3/F semi-buffet: seat 1-3 min + serving 1-5 min.",
    },
    {
        "zone_id": "AC2 Canteen (Counter)",
        "service_type": "Counter",
        "building": "LI",
        "floor": 3,
        "floor_label": "AC2 · 3/F",
        "capacity": 210,
        "popularity": 0.82,
        "coords": (1.60, -0.30),
        "phases": (
            ("Find a seat", 1.0, 3.0),
            ("Counter food pickup", 5.0, 10.0),
        ),
        "note": "Li 3/F counter: seat 1-3 min + pickup 5-10 min (slowest counter on campus).",
    },
    {
        "zone_id": "AC3 Bistro (Semi-Buffet)",
        "service_type": "Semi-Buffet",
        "building": "LAU",
        "floor": 7,
        "floor_label": "AC3 · 7/F",
        "capacity": 240,
        "popularity": 0.76,
        "coords": (3.00, 0.45),
        "phases": (
            ("Find a seat", 1.0, 5.0),
            ("Semi-buffet serving", 1.0, 3.0),
        ),
        "note": "Lau 7/F AC3 Bistro semi-buffet: seat 1-5 min + serving 1-3 min.",
    },
    {
        "zone_id": "AC3 Bistro (Counter)",
        "service_type": "Counter",
        "building": "LAU",
        "floor": 7,
        "floor_label": "AC3 · 7/F",
        "capacity": 200,
        "popularity": 0.70,
        "coords": (3.00, -0.10),
        "phases": (
            ("Find a seat", 1.0, 5.0),
            ("Counter food pickup", 4.0, 5.0),
        ),
        "note": "Lau 7/F AC3 Bistro counter: seat 1-5 min + pickup 4-5 min.",
    },
    {
        "zone_id": "Library 3/F",
        "service_type": "Study",
        "building": "YEUNG",
        "floor": 3,
        "floor_label": "Library · 3/F",
        "capacity": 180,
        "popularity": 1.05,
        "coords": (-0.55, -0.45),
        "phases": (("Seat search & settle in", 6.0, 12.0),),
        "note": "Library 3/F (entrance floor in Yeung): 6-12 min seat search at lunch peak.",
    },
    {
        "zone_id": "YEUNG 4/F Study Corner",
        "service_type": "Study",
        "building": "YEUNG",
        "floor": 4,
        "floor_label": "AC1 · 4/F",
        "capacity": 120,
        "popularity": 0.90,
        "coords": (-0.55, 0.55),
        "phases": (("Seat search & settle in", 6.0, 12.0),),
        "note": "Yeung 4/F LT corridor study corner: 6-12 min seat search at lunch peak.",
    },
    {
        "zone_id": "LAU 4/F Study Lounge",
        "service_type": "Study",
        "building": "LAU",
        "floor": 4,
        "floor_label": "AC3 · 4/F",
        "capacity": 140,
        "popularity": 0.58,
        "coords": (3.00, 1.05),
        "phases": (("Seat search & settle in", 0.2, 2.0),),
        "note": "Lau 4/F study lounge: 0-2 min seat search, usually the hidden gem.",
    },
    {
        "zone_id": "Library 4/F Quiet Zone",
        "service_type": "Study",
        "building": "YEUNG",
        "floor": 4,
        "floor_label": "Library · 4/F",
        "capacity": 150,
        "popularity": 0.55,
        "coords": (0.45, -0.45),
        "phases": (("Seat search & settle in", 0.2, 2.0),),
        "note": "Library 4/F quiet zone: 0-2 min seat search.",
    },
)

ZONE_IDS: Final[tuple[str, ...]] = tuple(z["zone_id"] for z in ZONES)
ZONE_BY_ID: Final[dict[str, dict]] = {z["zone_id"]: z for z in ZONES}

# Schematic footprints for the dashboard map (x0, x1, y0, y1).
BUILDING_BOXES: Final[dict[str, dict[str, float]]] = {
    "YEUNG": {"x0": -0.95, "x1": 0.95, "y0": -0.80, "y1": 0.95},
    "LI": {"x0": 1.05, "x1": 2.15, "y0": -0.70, "y1": 0.70},
    "LAU": {"x0": 2.45, "x1": 3.55, "y0": -0.70, "y1": 1.40},
}
BUILDING_ANCHORS: Final[dict[str, tuple[float, float]]] = {
    "YEUNG": (0.0, 0.90),
    "LI": (1.60, 0.65),
    "LAU": (3.00, 1.35),
}

# --------------------------------------------------------------------------- #
# Departure points
# --------------------------------------------------------------------------- #
# label -> (building, floor)
ORIGINS: Final[dict[str, tuple[str, int]]] = {
    "AC1 (YEUNG 4/F LT)": ("YEUNG", 4),
    "Library (YEUNG 3/F)": ("YEUNG", 3),
    "AC2 (LI)": ("LI", 1),
    "AC3 (LAU)": ("LAU", 1),
}
ORIGIN_LABELS: Final[tuple[str, ...]] = tuple(ORIGINS)

# --------------------------------------------------------------------------- #
# Demand goals
# --------------------------------------------------------------------------- #
USER_NEEDS: Final[dict[str, str]] = {
    "Quick Semi-Buffet": "Semi-Buffet",
    "Counter Meal": "Counter",
    "Study Seat": "Study",
}


# --------------------------------------------------------------------------- #
# Helpers (pure, dependency free -> reusable from the app and from training)
# --------------------------------------------------------------------------- #
def to_minutes(slot: str) -> int:
    """``"11:45"`` -> 705."""
    hour, minute = slot.split(":")
    return int(hour) * 60 + int(minute)


def from_minutes(total: int) -> str:
    """705 -> ``"11:45"`` (wraps inside a single day)."""
    total = int(total) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def quantize_slot(total_minutes: float) -> str:
    """Round any minute-of-day value onto the 15-minute observation grid."""
    quantised = int(round(float(total_minutes) / SLOT_MINUTES)) * SLOT_MINUTES
    return from_minutes(quantised)


def walk_time(origin_label: str, zone_id: str) -> float:
    """Door-to-door walking minutes from a departure point to a zone.

    Same building -> lift/stairs overhead plus a per-floor cost.
    Different building -> the measured ground-route distance between entrances.
    """
    origin_building, origin_floor = ORIGINS[origin_label]
    zone = ZONE_BY_ID[zone_id]
    if zone["building"] == origin_building:
        floors = abs(int(zone["floor"]) - int(origin_floor))
        return round(VERTICAL_BASE + VERTICAL_PER_FLOOR * floors, 1)
    return round(BASE_WALK[origin_building][zone["building"]], 1)


def expected_dismissals(minute_of_day: float) -> int:
    """Expected students released by nearby lectures in the last 10 minutes.

    This is the *scheduled* class-change signal (public timetable knowledge,
    no personal data): a small background flow with two sharp spikes at 11:50
    and 12:50.  The serving path uses it to build the
    ``nearby_class_dismissals`` feature for any arrival time the user picks,
    while the offline dataset adds day-to-day amplitude noise on top so the
    models cannot just memorise the timetable.
    """
    hour = minute_of_day / 60.0
    spike_a = 190.0 * math.exp(-((hour - 11.833) / 0.25) ** 2)
    spike_b = 135.0 * math.exp(-((hour - 12.833) / 0.25) ** 2)
    return int(max(0, round(6 + spike_a + spike_b)))


def phase_weights(zone_id: str) -> list[tuple[str, float]]:
    """Relative share of each bottleneck phase at full load.

    Used by the dashboard to break a *predicted total* wait into phases in an
    interpretable way without ever leaking per-phase labels into training.
    """
    zone = ZONE_BY_ID[zone_id]
    raw = [(label, (floor * 0.35 + peak) / 2.0) for label, floor, peak in zone["phases"]]
    total = sum(v for _, v in raw) or 1.0
    return [(label, v / total) for label, v in raw]


def phase_breakdown(zone_id: str, predicted_wait: float) -> list[tuple[str, float]]:
    """Split a predicted total wait into per-phase minutes."""
    return [(label, round(predicted_wait * share, 1)) for label, share in phase_weights(zone_id)]


def zones_for_need(need_label: str) -> list[str]:
    """Candidate zones for a demand goal."""
    service = USER_NEEDS[need_label]
    return [z["zone_id"] for z in ZONES if z["service_type"] == service]


def zone_hover_label(zone_id: str) -> str:
    """``"AC1 Canteen · AC1 · 5/F"`` style label for charts."""
    zone = ZONE_BY_ID[zone_id]
    return f"{zone_id} · {zone['floor_label']}"
