"""Static configuration for the CityUHK Smart Campus Crowd-Balancing demo.

Everything here is human readable: the nine campus zones (real buildings and
floors), the stopwatch-derived bottleneck structure of each canteen, the
walking-connection network, the canteen menus keyed by dish id, the service
hours, and the crowd-penalty constants.  Display strings live in ``i18n.py``.

Real campus geography
---------------------
* **AC1 / YEUNG** — Yeung Kin Man Academic Building, 8 floors.  5/F is the
  AC1 Canteen, 4/F holds the LT lecture-theatre zone and the Library entrance.
* **AC2 / LI** — Li Dak Sum Yip Yio Chin Academic Building, 9 floors.
  3/F is the AC2 Canteen.
* **AC3 / LAU** — Lau Ming Wai Academic Building, 20 floors.
  7/F is the AC3 Canteen (Bistro).
* **Halls** — Kowloon Tong student residence, a common departure point.

Walking model
-------------
``T_walk`` is a shortest-path search (Dijkstra) over a floor-level graph:
an elevator ride between any two floors of the same tower costs 1.5 min, and
the measured corridor links connect the towers and the Halls.

Privacy: only aggregate counters and wait times are modelled.  No student ID,
device identifier, Wi-Fi MAC address or any other personal data is used.
"""

from __future__ import annotations

import heapq
import math
from typing import Final

# --------------------------------------------------------------------------- #
# Project meta
# --------------------------------------------------------------------------- #
PROJECT_NAME: Final[str] = "Smart Campus Crowd-Balancing and Space Routing Engine"
PROJECT_SUBTITLE: Final[str] = "GE2340 · City University of Hong Kong"
PROJECT_VERSION: Final[str] = "2.0.0"

# --------------------------------------------------------------------------- #
# Temporal grid
# --------------------------------------------------------------------------- #
SLOT_MINUTES: Final[int] = 15
DAY_START_MIN: Final[int] = 7 * 60 + 30    # 07:30 — earliest canteen opening
DAY_END_MIN: Final[int] = 22 * 60          # 22:00 — study areas at night
LUNCH_PEAK: Final[tuple[int, int]] = (11 * 60 + 30, 14 * 60 + 30)

WEEKS: Final[tuple[int, ...]] = (1, 2, 3)
WEEKDAYS: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Eighteen observed 15-minute snapshots per day, covering the full 07:30-22:00
# service window: breakfast, the dense 11:30-13:30 lunch peak (so both the 11:50
# and 12:50 class-dismissal waves are captured), dinner and the late evening.
OBSERVED_SLOTS: Final[tuple[str, ...]] = (
    "07:30", "08:30", "09:30", "10:30",
    "11:30", "11:45", "12:00", "12:30", "13:00", "13:30",
    "14:30", "15:30", "16:30", "17:30", "18:30", "19:30",
    "20:30", "21:30",
)

MAX_ZONES_PER_SLOT: Final[int] = 8
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
CLOSED_COLOR: Final[str] = "#B9C2CC"

LOW_THRESHOLD: Final[float] = 0.60
HIGH_THRESHOLD: Final[float] = 0.85

CROWD_PENALTY: Final[dict[str, float]] = {"Low": 0.0, "Medium": 1.5, "High": 4.0}

# --------------------------------------------------------------------------- #
# Buildings
# --------------------------------------------------------------------------- #
BUILDINGS: Final[tuple[str, ...]] = ("YEUNG", "LI", "LAU")
HALLS: Final[str] = "HALLS"

BUILDING_SHORT: Final[dict[str, str]] = {
    "YEUNG": "AC1 · YEUNG",
    "LI": "AC2 · LI",
    "LAU": "AC3 · LAU",
    HALLS: "Halls",
}
BUILDING_FLOORS: Final[dict[str, int]] = {"YEUNG": 8, "LI": 9, "LAU": 20}

# Floors a student can actually be standing on, per building.
FLOOR_CHOICES: Final[dict[str, tuple[str, ...]]] = {
    "YEUNG": tuple(["G"] + [f"{i}F" for i in range(1, 9)]),
    "LI": ("B2", "B1", "G") + tuple([f"{i}F" for i in range(1, 10)]),
    "LAU": tuple(["G"] + [f"{i}F" for i in range(1, 21)]),
}
# Any floor is reachable from any other in the same tower via the lift, so the
# walking graph only needs the floors that anchor corridors or zones.
FLOOR_NODES: Final[dict[str, tuple[str, ...]]] = {
    "YEUNG": ("G", "3F", "4F", "5F"),
    "LI": ("B2", "B1", "G", "3F", "4F"),
    "LAU": ("G", "3F", "4F", "5F", "7F"),
    HALLS: ("G",),
}

VERTICAL_FLAT: Final[float] = 1.5

CORRIDOR_LINKS: Final[tuple[tuple[str, str, float], ...]] = (
    ("YEUNG-4F", "LI-B2", 5.0),
    ("YEUNG-4F", "LAU-G", 7.0),
    ("YEUNG-5F", "LI-B2", 5.0),
    ("YEUNG-5F", "LAU-3F", 5.0),
    ("LI-4F", "LAU-5F", 3.0),
    ("LI-G", "LAU-3F", 1.0),
    ("HALLS-G", "YEUNG-G", 17.0),
    ("HALLS-G", "LI-G", 12.0),
    ("HALLS-G", "LAU-G", 12.0),
)

CORRIDOR_TABLE: Final[tuple[tuple[str, str, float], ...]] = (
    ("AC1 4/F", "AC2 B2", 5.0),
    ("AC1 4/F", "AC3 G/F", 7.0),
    ("AC1 5/F", "AC2 B2", 5.0),
    ("AC1 5/F", "AC3 3/F", 5.0),
    ("AC2 4/F", "AC3 5/F", 3.0),
    ("AC2 G/F", "AC3 3/F", 1.0),
    ("Halls", "AC3", 12.0),
    ("Halls", "AC2", 12.0),
    ("Halls", "AC1", 17.0),
)

# --------------------------------------------------------------------------- #
# Service hours (minutes from midnight).  ``None`` means "always open".
# --------------------------------------------------------------------------- #
ALL_DAYS: Final[tuple[str, ...]] = WEEKDAYS
WEEKDAYS_ONLY: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat")

SERVICE_HOURS: Final[dict[str, dict | None]] = {
    "AC1 Canteen": {"open": 7 * 60 + 30, "close": 21 * 60, "days": ALL_DAYS},
    "AC2 Canteen (Semi-Buffet)": {"open": 7 * 60 + 30, "close": 21 * 60, "days": ALL_DAYS},
    "AC2 Canteen (Counter)": {"open": 7 * 60 + 30, "close": 21 * 60, "days": ALL_DAYS},
    "AC3 Canteen (Semi-Buffet)": {"open": 7 * 60 + 30, "close": 21 * 60, "days": WEEKDAYS_ONLY},
    "AC3 Canteen (Counter)": {"open": 7 * 60 + 30, "close": 21 * 60, "days": WEEKDAYS_ONLY},
    # Study areas follow the library day, not a canteen schedule.
    "Library 3/F": {"open": DAY_START_MIN, "close": DAY_END_MIN, "days": ALL_DAYS},
    "Library 4/F Quiet Zone": {"open": DAY_START_MIN, "close": DAY_END_MIN, "days": ALL_DAYS},
    "YEUNG 4/F Study Corner": {"open": DAY_START_MIN, "close": DAY_END_MIN, "days": ALL_DAYS},
    "LAU 4/F Study Lounge": {"open": DAY_START_MIN, "close": DAY_END_MIN, "days": ALL_DAYS},
}

# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
ZONES: Final[tuple[dict, ...]] = (
    {
        "zone_id": "AC1 Canteen",
        "service_type": "Counter",
        "building": "YEUNG",
        "floor": 5,
        "floor_label": "5/F",
        "node": "YEUNG-5F",
        "capacity": 300,
        "popularity": 1.00,
        "coords": (0.45, 0.55),
        "phases": (
            ("Order screen queue", 2.0, 2.8),
            ("Food pickup", 3.0, 5.0),
            ("Find a seat", 1.5, 2.2),
            ("Utensils & condiments", 0.8, 1.0),
        ),
        "note": "Yeung 5/F canteen: screen 2-3 min + pickup 3-5 min + seat 2 min + utensils 1 min.",
    },
    {
        "zone_id": "AC2 Canteen (Semi-Buffet)",
        "service_type": "Semi-Buffet",
        "building": "LI",
        "floor": 3,
        "floor_label": "3/F",
        "node": "LI-3F",
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
        "floor_label": "3/F",
        "node": "LI-3F",
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
        "zone_id": "AC3 Canteen (Semi-Buffet)",
        "service_type": "Semi-Buffet",
        "building": "LAU",
        "floor": 7,
        "floor_label": "7/F",
        "node": "LAU-7F",
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
        "zone_id": "AC3 Canteen (Counter)",
        "service_type": "Counter",
        "building": "LAU",
        "floor": 7,
        "floor_label": "7/F",
        "node": "LAU-7F",
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
        "floor_label": "3/F",
        "node": "YEUNG-3F",
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
        "floor_label": "4/F",
        "node": "YEUNG-4F",
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
        "floor_label": "4/F",
        "node": "LAU-4F",
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
        "floor_label": "4/F",
        "node": "YEUNG-4F",
        "capacity": 150,
        "popularity": 0.55,
        "coords": (0.45, -0.45),
        "phases": (("Seat search & settle in", 0.2, 2.0),),
        "note": "Library 4/F quiet zone: 0-2 min seat search.",
    },
)

ZONE_IDS: Final[tuple[str, ...]] = tuple(z["zone_id"] for z in ZONES)
ZONE_BY_ID: Final[dict[str, dict]] = {z["zone_id"]: z for z in ZONES}

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
NETWORK_COORDS: Final[dict[str, tuple[float, float]]] = {
    "YEUNG": (-1.30, 0.05),
    "LI": (0.55, 0.05),
    "LAU": (1.80, -0.35),
    HALLS: (-0.55, 1.35),
}

# --------------------------------------------------------------------------- #
# Food categories and canteen menus (dish ids resolve to names via i18n)
# --------------------------------------------------------------------------- #
CAT_ANY: Final[str] = "any"
CAT_RICE: Final[str] = "rice"
CAT_NOODLE: Final[str] = "noodle"
CAT_WESTERN: Final[str] = "western"
CAT_ASIAN: Final[str] = "asian"

FOOD_CATEGORIES: Final[tuple[str, ...]] = (CAT_ANY, CAT_RICE, CAT_NOODLE, CAT_WESTERN, CAT_ASIAN)

CANTEEN_MENU: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "YEUNG": {
        "bbq": (CAT_RICE,),
        "two_dish": (CAT_RICE,),
        "cart_noodle": (CAT_NOODLE,),
        "ramen_beef": (CAT_NOODLE,),
        "pizza": (CAT_WESTERN,),
        "fried_chicken": (CAT_WESTERN,),
        "sea_asian": (CAT_ASIAN,),
    },
    "LI": {
        "two_dish": (CAT_RICE,),
        "rice_noodle": (CAT_NOODLE,),
        "mala_tang": (CAT_NOODLE,),
        "dumpling": (CAT_NOODLE,),
        "pasta_bake": (CAT_WESTERN,),
    },
    "LAU": {
        "semi_buffet": (CAT_RICE,),
        "curry_rice": (CAT_ASIAN,),
        "signature_noodle": (CAT_NOODLE,),
    },
}

STUDY_GOAL: Final[str] = "study"


# --------------------------------------------------------------------------- #
# Walking network
# --------------------------------------------------------------------------- #
def _build_adjacency() -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = {}

    def add(a: str, b: str, weight: float) -> None:
        graph.setdefault(a, []).append((b, weight))
        graph.setdefault(b, []).append((a, weight))

    for building, floors in FLOOR_NODES.items():
        nodes = [f"{building}-{f}" for f in floors]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                add(a, b, VERTICAL_FLAT)
    for a, b, weight in CORRIDOR_LINKS:
        add(a, b, weight)
    return graph


_ADJACENCY: Final[dict[str, list[tuple[str, float]]]] = _build_adjacency()


def _floor_token(building: str, floor_label: str) -> str:
    """``YEUNG`` + ``"4/F"`` -> ``"YEUNG-4F"``; ``"G"`` stays ``"YEUNG-G"``."""
    token = floor_label.replace("/", "")
    return f"{building}-{token}"


def floor_node(building: str, floor_label: str) -> str:
    """Graph node for any selectable floor.

    Floors that do not anchor a corridor or a zone (e.g. AC3 12/F) route
    through the tower's top anchor; ``walk_route`` adds the lift ride from the
    real floor so the time stays correct.
    """
    candidate = _floor_token(building, floor_label)
    if candidate in _ADJACENCY:
        return candidate
    if building == HALLS:
        return "HALLS-G"
    return f"{building}-{FLOOR_NODES[building][-1]}"


def _origin_anchor(building: str, floor_label: str) -> str | None:
    """Anchor floor token when the chosen floor has no graph node."""
    candidate = _floor_token(building, floor_label)
    if candidate in _ADJACENCY or building == HALLS:
        return None
    return FLOOR_NODES[building][-1]


def shortest_path(origin_node: str, target_node: str) -> tuple[float, list[str]]:
    """Dijkstra returning (minutes, node path)."""
    if origin_node == target_node:
        return 0.0, [origin_node]
    dist: dict[str, float] = {origin_node: 0.0}
    prev: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, origin_node)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == target_node:
            break
        if cost > dist.get(node, math.inf):
            continue
        for neighbour, weight in _ADJACENCY.get(node, []):
            new_cost = cost + weight
            if new_cost < dist.get(neighbour, math.inf):
                dist[neighbour] = new_cost
                prev[neighbour] = node
                heapq.heappush(queue, (new_cost, neighbour))
    if target_node not in dist:
        raise ValueError(f"no walking path from {origin_node} to {target_node}")
    path = [target_node]
    while path[-1] != origin_node:
        path.append(prev[path[-1]])
    return round(dist[target_node], 2), list(reversed(path))


def walk_route(origin_building: str, origin_floor: str, zone_id: str) -> tuple[float, list[str]]:
    """Minutes + node path from a (building, floor) departure point to a zone."""
    start = floor_node(origin_building, origin_floor)
    minutes, path = shortest_path(start, ZONE_BY_ID[zone_id]["node"])
    if _origin_anchor(origin_building, origin_floor) is not None:
        minutes = round(minutes + VERTICAL_FLAT, 2)
    return minutes, path


def walk_time(origin_building: str, origin_floor: str, zone_id: str) -> float:
    return walk_route(origin_building, origin_floor, zone_id)[0]


def building_hop_minutes(a: str, b: str) -> float:
    if a == b:
        return 0.0
    best = math.inf
    for left, right, weight in CORRIDOR_LINKS:
        if {left.split("-")[0], right.split("-")[0]} == {a, b}:
            best = min(best, weight)
    return round(best, 1)


# --------------------------------------------------------------------------- #
# Service hours
# --------------------------------------------------------------------------- #
def is_service_open(zone_id: str, weekday: str, minute_of_day: float) -> bool:
    """Is this zone serving at that clock time on that weekday?"""
    hours = SERVICE_HOURS.get(zone_id)
    if hours is None:
        return True
    if weekday not in hours["days"]:
        return False
    return hours["open"] <= minute_of_day < hours["close"]


def opening_label(zone_id: str) -> str:
    """``"07:30–21:00"`` for the footer of a card."""
    hours = SERVICE_HOURS.get(zone_id)
    if hours is None:
        return "24h"
    return f"{from_minutes(hours['open'])}–{from_minutes(hours['close'])}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_minutes(slot: str) -> int:
    hour, minute = slot.split(":")
    return int(hour) * 60 + int(minute)


def from_minutes(total: int) -> str:
    total = int(total) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def quantize_slot(total_minutes: float) -> str:
    quantised = int(round(float(total_minutes) / SLOT_MINUTES)) * SLOT_MINUTES
    return from_minutes(quantised)


def expected_dismissals(minute_of_day: float) -> int:
    """Scheduled class-change flow: background plus the 11:50 and 12:50 waves."""
    hour = minute_of_day / 60.0
    spike_a = 190.0 * math.exp(-((hour - 11.833) / 0.25) ** 2)
    spike_b = 135.0 * math.exp(-((hour - 12.833) / 0.25) ** 2)
    return int(max(0, round(6 + spike_a + spike_b)))


def demand_pressure(minute_of_day: float) -> float:
    """Campus-wide demand curve across the whole 07:30-22:00 day."""
    hour = minute_of_day / 60.0
    breakfast = 0.28 * math.exp(-(((hour - 8.20) / 0.45) ** 2))
    lunch_main = 1.15 * math.exp(-(((hour - 12.25) / 0.85) ** 2))
    wave_a = 0.55 * math.exp(-(((hour - 11.83) / 0.20) ** 2))
    wave_b = 0.45 * math.exp(-(((hour - 12.85) / 0.22) ** 2))
    afternoon = 0.45 * math.exp(-(((hour - 15.80) / 0.90) ** 2))
    dinner = 0.85 * math.exp(-(((hour - 18.30) / 0.60) ** 2))
    return float(breakfast + lunch_main + wave_a + wave_b + afternoon + dinner)


def phase_weights(zone_id: str) -> list[tuple[str, float]]:
    zone = ZONE_BY_ID[zone_id]
    raw = [(label, (floor * 0.35 + peak) / 2.0) for label, floor, peak in zone["phases"]]
    total = sum(v for _, v in raw) or 1.0
    return [(label, v / total) for label, v in raw]


def phase_breakdown(zone_id: str, predicted_wait: float) -> list[tuple[str, float]]:
    return [(label, round(predicted_wait * share, 1)) for label, share in phase_weights(zone_id)]


def canteen_dishes(building: str, category: str) -> tuple[str, ...]:
    menu = CANTEEN_MENU.get(building, {})
    if category == CAT_ANY:
        return tuple(menu)
    return tuple(dish for dish, cats in menu.items() if category in cats)


def zones_for_category(category: str) -> list[str]:
    return [
        z["zone_id"]
        for z in ZONES
        if z["service_type"] != "Study" and canteen_dishes(z["building"], category)
    ]


def zones_for_study() -> list[str]:
    return [z["zone_id"] for z in ZONES if z["service_type"] == "Study"]


def zones_for_goal(goal_label: str) -> list[str]:
    if goal_label == STUDY_GOAL:
        return zones_for_study()
    return zones_for_category(goal_label)


def _floor_display(token: str) -> str:
    """``"5F"`` -> ``"5/F"``; ``"G"`` / ``"B2"`` stay as-is."""
    if len(token) > 1 and token.endswith("F") and token[:-1].isdigit():
        return f"{token[:-1]}/F"
    return token


def route_steps(origin_building: str, origin_floor: str, zone_id: str) -> tuple[float, list[dict]]:
    """Human-usable route: (minutes, [{kind: "lift"|"corridor", ...}]).

    Consecutive nodes inside the same tower become a single lift ride; a hop
    between two towers becomes a corridor step.
    """
    minutes, path = walk_route(origin_building, origin_floor, zone_id)
    steps: list[dict] = []
    for a, b in zip(path, path[1:]):
        a_building, _ = a.split("-", 1)
        b_building, b_floor = b.split("-", 1)
        if a_building == b_building:
            steps.append({"kind": "lift", "floor": _floor_display(b_floor)})
        else:
            steps.append({
                "kind": "corridor",
                "building": b_building,
                "floor": _floor_display(b_floor),
            })
    anchor = _origin_anchor(origin_building, origin_floor)
    if anchor is not None and not (steps and steps[0]["kind"] == "lift"):
        steps.insert(0, {"kind": "lift", "floor": _floor_display(anchor)})
    return minutes, steps
