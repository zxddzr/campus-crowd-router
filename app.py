"""Smart Campus Crowd-Balancing & Space Routing Engine -- local Streamlit dashboard.

Run::

    streamlit run app.py

Tab 1 -- Live Campus Router
    Pick where you are (AC1 / AC2 / AC3 / Halls), what you want to eat (or a
    study seat) and when you leave.  For every candidate zone the app walks you
    through the real corridor network, predicts wait time + congestion level at
    your *arrival* moment with the trained Ridge / MLP models, adds a crowd
    penalty and ranks the options ("Go Here Instead").
Tab 2 -- Model Evaluation & Course Alignment
    Reads ``models/metrics.json`` and shows how the two selected models compare
    against the two baselines.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.campus_config import (  # noqa: E402
    BUILDING_ANCHORS,
    BUILDING_BOXES,
    BUILDING_FLOORS,
    BUILDING_LABELS,
    BUILDING_SHORT,
    CANTEEN_MENU,
    CAT_ANY,
    CONGESTION_COLORS,
    CONGESTION_LEVELS,
    CORRIDOR_LINKS,
    CORRIDOR_TABLE,
    CROWD_PENALTY,
    DAY_END_MIN,
    DAY_START_MIN,
    FEATURE_COLUMNS,
    FOOD_CATEGORIES,
    FOOD_CATEGORY_HINT,
    HALLS,
    NETWORK_COORDS,
    ORIGIN_LABELS,
    PROJECT_NAME,
    PROJECT_SUBTITLE,
    PROJECT_VERSION,
    SLOT_MINUTES,
    STUDY_GOAL,
    WEEKDAYS,
    WEEKS,
    ZONES,
    ZONE_BY_ID,
    ZONE_IDS,
    canteen_dishes,
    expected_dismissals,
    from_minutes,
    phase_breakdown,
    walk_time,
    zones_for_goal,
)

MODEL_DIR = PROJECT_ROOT / "models"
CITY_TZ = ZoneInfo("Asia/Hong_Kong")

LEVEL_ORDER = list(CONGESTION_LEVELS)

# Compact map labels (the full zone id is too long for a bubble caption).
MAP_LABELS: dict[str, str] = {
    "AC1 Canteen": "AC1 Canteen",
    "AC2 Canteen (Semi-Buffet)": "AC2 Semi",
    "AC2 Canteen (Counter)": "AC2 Counter",
    "AC3 Canteen (Semi-Buffet)": "AC3 Semi",
    "AC3 Canteen (Counter)": "AC3 Counter",
    "Library 3/F": "Library 3/F",
    "YEUNG 4/F Study Corner": "Yeung 4/F",
    "LAU 4/F Study Lounge": "LAU 4/F",
    "Library 4/F Quiet Zone": "Library 4/F",
}

GOAL_FOOD = "🍜 Food (canteen)"
GOAL_STUDY = f"📚 {STUDY_GOAL}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_models():
    ridge = joblib.load(MODEL_DIR / "ridge_model.joblib")
    mlp = joblib.load(MODEL_DIR / "mlp_model.joblib")
    return ridge, mlp


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def feature_frame(zone_ids: list[str], week: int, weekday: str, minute_of_day: float) -> pd.DataFrame:
    """One row per zone at a given arrival minute."""
    rows = []
    for zone_id in zone_ids:
        zone = ZONE_BY_ID[zone_id]
        rows.append(
            {
                "week": int(week),
                "weekday": weekday,
                "time_slot": from_minutes(int(round(minute_of_day))),
                "zone_id": zone_id,
                "service_type": zone["service_type"],
                "zone_capacity": zone["capacity"],
                "nearby_class_dismissals": expected_dismissals(minute_of_day),
            }
        )
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


def predict_zone(ridge, mlp, zone_ids, week, weekday, minute_of_day) -> pd.DataFrame:
    frame = feature_frame(zone_ids, week, weekday, minute_of_day)
    wait = np.clip(np.asarray(ridge.predict(frame), dtype=float), 0.0, None)

    proba = np.asarray(mlp.predict_proba(frame), dtype=float)
    classes = list(mlp.named_steps["model"].classes_)
    out = pd.DataFrame({"zone_id": frame["zone_id"].to_numpy(), "wait_pred": wait})
    for level in LEVEL_ORDER:
        out[f"p_{level}"] = proba[:, classes.index(level)] if level in classes else 0.0
    out["level"] = [
        LEVEL_ORDER[int(np.argmax(row[[f"p_{lv}" for lv in LEVEL_ORDER]]))] for _, row in out.iterrows()
    ]
    return out


def route_table(ridge, mlp, origin: str, candidates: list[str], week: int, weekday: str,
                departure_min: float) -> pd.DataFrame:
    """Total-time model: walk + predicted wait at arrival + crowd penalty."""
    rows = []
    for zone_id in candidates:
        zone = ZONE_BY_ID[zone_id]
        walk = walk_time(origin, zone_id)
        arrival = departure_min + walk
        pred = predict_zone(ridge, mlp, [zone_id], week, weekday, arrival).iloc[0]
        penalty = CROWD_PENALTY[pred["level"]]
        rows.append(
            {
                "zone_id": zone_id,
                "service_type": zone["service_type"],
                "building": zone["building"],
                "floor_label": zone["floor_label"],
                "walk_mins": walk,
                "arrival": from_minutes(int(round(arrival))),
                "wait_pred": float(pred["wait_pred"]),
                "level": pred["level"],
                "p_Low": float(pred["p_Low"]),
                "p_Medium": float(pred["p_Medium"]),
                "p_High": float(pred["p_High"]),
                "crowd_penalty": penalty,
                "total_mins": walk + float(pred["wait_pred"]) + penalty,
            }
        )
    return pd.DataFrame(rows).sort_values("total_mins").reset_index(drop=True)


def all_zone_status(ridge, mlp, week: int, weekday: str, departure_min: float, origin: str) -> pd.DataFrame:
    """Every zone, for the campus-wide maps."""
    rows = []
    for zone_id in ZONE_IDS:
        zone = ZONE_BY_ID[zone_id]
        walk = walk_time(origin, zone_id)
        arrival = departure_min + walk
        pred = predict_zone(ridge, mlp, [zone_id], week, weekday, arrival).iloc[0]
        rows.append(
            {
                "zone_id": zone_id,
                "service_type": zone["service_type"],
                "building": zone["building"],
                "floor_label": zone["floor_label"],
                "walk_mins": walk,
                "arrival": from_minutes(int(round(arrival))),
                "wait_pred": float(pred["wait_pred"]),
                "level": pred["level"],
                "p_Low": float(pred["p_Low"]),
                "p_Medium": float(pred["p_Medium"]),
                "p_High": float(pred["p_High"]),
                "crowd_penalty": CROWD_PENALTY[pred["level"]],
            }
        )
    out = pd.DataFrame(rows)
    out["total_mins"] = out["walk_mins"] + out["wait_pred"] + out["crowd_penalty"]
    out.attrs["origin_label"] = origin
    return out


def zone_dishes(zone_id: str, goal: str, category: str) -> tuple[str, ...]:
    """Dishes of that canteen that match the selected food category."""
    if goal == GOAL_STUDY:
        return ()
    return canteen_dishes(ZONE_BY_ID[zone_id]["building"], category)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def corridor_minutes(a: str, b: str) -> list[float]:
    """All measured corridor links between two buildings (sorted, unique)."""
    if a == b:
        return []
    values = {
        weight
        for left, right, weight in CORRIDOR_LINKS
        if {left.split("-")[0], right.split("-")[0]} == {a, b}
    }
    return sorted(values)


# Manual nudges so the minute labels never sit under a node.
EDGE_LABEL_OFFSET: dict[frozenset, tuple[float, float]] = {
    frozenset(("YEUNG", "LI")): (0.0, -0.26),
    frozenset(("YEUNG", "LAU")): (0.0, -0.30),
    frozenset(("LI", "LAU")): (0.02, 0.26),
    frozenset(("HALLS", "YEUNG")): (-0.34, -0.08),
    frozenset(("HALLS", "LI")): (0.34, 0.02),
    frozenset(("HALLS", "LAU")): (0.12, 0.26),
}


def network_figure(status: pd.DataFrame, origin: str, category: str, goal: str,
                   mae: float) -> go.Figure:
    """2-D connection network: Halls + the three academic towers."""
    towers = ["YEUNG", "LI", "LAU"]
    origin_building = {
        "AC1 (YEUNG 4/F LT)": "YEUNG",
        "AC2 (LI)": "LI",
        "AC3 (LAU)": "LAU",
        "Halls (九龙塘宿舍)": HALLS,
    }[origin]

    # Per building: the worst canteen zone that is a candidate for this goal,
    # falling back to the worst zone of the building.
    candidates = set(zones_for_goal(category if goal == GOAL_FOOD else STUDY_GOAL))
    summary: dict[str, dict] = {}
    for building in towers + [HALLS]:
        subset = status[status["building"] == building]
        if subset.empty:
            summary[building] = {"level": None, "wait": 0.0, "rows": [], "best": None}
            continue
        eligible = subset[subset["zone_id"].isin(candidates)]
        pool = eligible if not eligible.empty else subset
        worst = pool.sort_values("wait_pred", ascending=False).iloc[0]
        summary[building] = {
            "level": worst["level"],
            "wait": float(worst["wait_pred"]),
            "rows": pool.sort_values("wait_pred", ascending=False).to_dict("records"),
            "best": pool.sort_values("total_mins").iloc[0],
        }

    fig = go.Figure()

    # Corridor edges first, so nodes draw on top.
    for a in towers + [HALLS]:
        for b in towers + [HALLS]:
            if a >= b:
                continue
            values = corridor_minutes(a, b)
            if not values:
                continue
            if len(values) == 1:
                label = f"{values[0]:.0f} min"
            else:
                label = f"{values[0]:.0f}–{values[-1]:.0f} min"
            x0, y0 = NETWORK_COORDS[a]
            x1, y1 = NETWORK_COORDS[b]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(color="#c3d0dd", width=3),
                    hoverinfo="skip", showlegend=False,
                )
            )
            dx, dy = EDGE_LABEL_OFFSET.get(frozenset((a, b)), (0.0, 0.0))
            fig.add_annotation(
                x=(x0 + x1) / 2 + dx, y=(y0 + y1) / 2 + dy,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color="#5b6f86"),
                bgcolor="rgba(255,255,255,0.92)",
                borderpad=2,
            )

    # Nodes.
    node_x, node_y, node_size, node_color, node_text, node_hover = [], [], [], [], [], []
    for building in towers + [HALLS]:
        info = summary[building]
        x, y = NETWORK_COORDS[building]
        label = BUILDING_SHORT[building]
        if building == HALLS:
            level = "Low"
            color = "#9aa9b8"
            size = 40
            hover = (
                f"<b>{BUILDING_LABELS[HALLS]}</b><br>"
                f"Your departure point · no canteen here<br>"
                f"AC1 ≈ 17 min · AC2 ≈ 12 min · AC3 ≈ 12 min walk"
            )
        else:
            floors = BUILDING_FLOORS[building]
            level = info["level"]
            color = CONGESTION_COLORS[level]
            size = 28 + 3.6 * min(info["wait"], 13.0)
            lines = [f"<b>{BUILDING_LABELS[building]}</b>", f"{floors} floors"]
            for row in info["rows"]:
                dishes = canteen_dishes(building, category) if goal == GOAL_FOOD else ()
                dish_txt = (" · " + " / ".join(dishes)) if dishes else ""
                lines.append(
                    f"<br>{row['zone_id']} · {row['floor_label']}<br>"
                    f"⏳ wait {row['wait_pred']:.1f} ± {mae:.2f} min · {row['level']} crowd<br>"
                    f"🚶 {row['walk_mins']:.1f} min from {origin}{dish_txt}"
                )
            if goal == GOAL_STUDY:
                lines.append("<br>(no canteen zone for a study goal)")
            hover = "".join(lines)
        node_x.append(x)
        node_y.append(y)
        node_size.append(size)
        node_color.append(color)
        node_text.append(label + (" 📍" if building == origin_building else ""))
        node_hover.append(hover)

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=node_size, color=node_color, line=dict(color="white", width=2.5), opacity=0.94),
            text=node_text,
            textposition="top center",
            textfont=dict(size=11, color="#26374b"),
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
            cliponaxis=False,
        )
    )

    for level in LEVEL_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=11, color=CONGESTION_COLORS[level]),
                name=f"{level} crowd", showlegend=True, hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=560,
        margin=dict(l=6, r=6, t=30, b=6),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        xaxis=dict(range=[-2.20, 2.35], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.95, 1.85], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def floor_map_figure(status: pd.DataFrame, routes: pd.DataFrame, need_label: str, mae: float) -> go.Figure:
    """Schematic 2-D map of the three towers with one bubble per zone."""
    ranks = {zone_id: idx + 1 for idx, zone_id in enumerate(routes["zone_id"])}

    fig = go.Figure()
    for building, box in BUILDING_BOXES.items():
        fig.add_shape(
            type="rect",
            x0=box["x0"], x1=box["x1"], y0=box["y0"], y1=box["y1"],
            line=dict(color="#c3d0dd", width=1.4), fillcolor="#f2f6fa", layer="below",
        )
        ax, ay = BUILDING_ANCHORS[building]
        fig.add_annotation(
            x=ax, y=ay, text=f"<b>{BUILDING_SHORT[building]}</b> · {BUILDING_FLOORS[building]}F",
            showarrow=False, font=dict(size=11, color="#41556d"), yshift=14,
        )

    hover = []
    for _, row in status.iterrows():
        rank = ranks.get(row["zone_id"])
        if rank == 1:
            reason = f"⭐ Best pick · total {row['total_mins']:.1f} min"
        elif rank is not None:
            reason = f"#{rank} pick · total {row['total_mins']:.1f} min"
        else:
            reason = f"Not eligible for “{need_label}” (would be {row['total_mins']:.1f} min)"
        hover.append(
            f"<b>{row['zone_id']}</b><br>"
            f"{BUILDING_LABELS[row['building']]}<br>"
            f"Floor: {row['floor_label']}<br>"
            f"🚶 {row['walk_mins']:.1f} min walk · arrive {row['arrival']}<br>"
            f"⏳ predicted wait {row['wait_pred']:.1f} ± {mae:.2f} min<br>"
            f"👥 {row['level']} crowd (Low {row['p_Low'] * 100:.0f}% / "
            f"Medium {row['p_Medium'] * 100:.0f}% / High {row['p_High'] * 100:.0f}%)<br>"
            f"{reason}"
        )

    fig.add_trace(
        go.Scatter(
            x=[ZONE_BY_ID[z]["coords"][0] for z in status["zone_id"]],
            y=[ZONE_BY_ID[z]["coords"][1] for z in status["zone_id"]],
            mode="markers+text",
            marker=dict(
                size=[15 + 3.6 * min(w, 13.0) for w in status["wait_pred"]],
                color=[CONGESTION_COLORS[lv] for lv in status["level"]],
                line=dict(color="white", width=2), opacity=0.92,
            ),
            text=[MAP_LABELS.get(z, z) for z in status["zone_id"]],
            textposition="bottom center",
            textfont=dict(size=9.5, color="#35485f"),
            hovertext=hover, hoverinfo="text", showlegend=False, cliponaxis=False,
        )
    )

    origin_label = status.attrs.get("origin_label")
    if origin_label:
        fig.add_annotation(
            x=-1.25, y=-1.10, text=f"📍 from {origin_label}", showarrow=False,
            font=dict(size=10, color="#7d5ba6"), xanchor="left",
        )

    fig.update_layout(
        height=560,
        margin=dict(l=6, r=6, t=26, b=6),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[-1.30, 3.90], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1.35, 1.80], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1280px;}
  .hero {background: linear-gradient(120deg, #10233f 0%, #1d4d7a 55%, #2a7f9e 100%);
         border-radius: 18px; padding: 22px 26px; color: #fff; margin-bottom: 18px;}
  .hero h1 {margin: 0; font-size: 1.55rem; letter-spacing: .2px;}
  .hero p {margin: 6px 0 0; opacity: .82; font-size: .92rem;}
  .route-card {background: #ffffff; border-radius: 14px; padding: 16px 18px;
               box-shadow: 0 2px 14px rgba(16,35,63,.10); height: 100%;}
  .route-card h3 {margin: 2px 0 6px; font-size: 1.02rem; color: #10233f;}
  .route-card .rank {font-size: .74rem; letter-spacing: .12em; text-transform: uppercase;
                     color: #6b7b8f; font-weight: 700;}
  .route-card .facts {font-size: .84rem; color: #35485f; margin-bottom: 8px; line-height: 1.5;}
  .route-card .saving {font-size: .78rem; color: #1f7a4d; font-weight: 600;}
  .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff;
         font-size: .72rem; font-weight: 700; letter-spacing: .04em;}
  .tag {display: inline-block; background: #eef4fa; color: #31465e; border-radius: 6px;
        padding: 2px 7px; font-size: .72rem; margin: 3px 3px 0 0; border: 1px solid #dbe6f0;}
  .menu {margin: 4px 0 8px;}
  .note {background: #f4f7fb; border-left: 4px solid #2a7f9e; border-radius: 8px;
         padding: 12px 16px; font-size: .86rem; color: #35485f;}
  div[data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
"""


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>🏫 {PROJECT_NAME}</h1>
          <p>{PROJECT_SUBTITLE} · v{PROJECT_VERSION} · Ridge + MLP routing over 9 zones in
          AC1/YEUNG, AC2/LI and AC3/LAU, reachable from the Kowloon Tong Halls<br/>
          11:30–14:30 lunch peak · 15-minute resolution · corridor walking network + predicted queue + crowd penalty</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Tab 1
# --------------------------------------------------------------------------- #
def tab_router(ridge, mlp, metrics: dict) -> None:
    mae = float(metrics["metrics"]["Ridge + MLPClassifier (selected)"]["mae_wait_mins"])
    now = datetime.now(CITY_TZ)
    now_min = now.hour * 60 + now.minute

    st.sidebar.header("1 · Your situation")
    origin = st.sidebar.selectbox("Where are you now?", ORIGIN_LABELS, index=0)
    st.sidebar.caption(
        "AC1 = Yeung Kin Man Academic Building (8F) · AC2 = Li Dak Sum Yip Yio Chin "
        "Academic Building (9F) · AC3 = Lau Ming Wai Academic Building (20F) · "
        "Halls = Kowloon Tong residence"
    )

    goal = st.sidebar.radio("What do you need?", [GOAL_FOOD, GOAL_STUDY], horizontal=True)
    if goal == GOAL_FOOD:
        category = st.sidebar.selectbox(
            "Food category", FOOD_CATEGORIES, index=0,
            help="Filter by dish family instead of a single dish, so the router can "
                 "balance load across canteens.",
        )
        st.sidebar.caption(FOOD_CATEGORY_HINT[category])
        candidates = zones_for_goal(category)
        need_label = category
    else:
        category = CAT_ANY
        candidates = zones_for_goal(STUDY_GOAL)
        need_label = STUDY_GOAL
        st.sidebar.caption("Library 3/F–4/F, Yeung 4/F corner, Lau 4/F lounge.")

    weekday = st.sidebar.selectbox("Day", WEEKDAYS, index=min(now.weekday(), len(WEEKDAYS) - 1))
    week = st.sidebar.selectbox("Teaching week", WEEKS, index=len(WEEKS) - 1)

    st.sidebar.header("2 · When do you leave?")
    options = list(range(DAY_START_MIN, DAY_END_MIN + 1, SLOT_MINUTES))
    target_min = now_min if DAY_START_MIN <= now_min <= DAY_END_MIN else 12 * 60
    default_index = min(range(len(options)), key=lambda i: abs(options[i] - target_min))
    departure_min = st.sidebar.select_slider(
        "Departure time (15-min grid)",
        options=options, value=options[default_index], format_func=from_minutes,
    )
    st.sidebar.caption(
        f"Current time is {now.strftime('%H:%M')} (Asia/Hong_Kong). "
        "Predictions are made for your **arrival** moment, i.e. departure + walking time. "
        "Outside 09:00–17:00 the slider opens on the lunch peak."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Routing formula**\n\n"
        r"$$T(z)=T_{walk}(u,z)+\hat W(z,\;t+T_{walk})+\;P_{crowd}(z)$$"
        "\n\n"
        "*T<sub>walk</sub>* = shortest path through the corridor network "
        "(elevator 1.5 min between floors, measured corridor links between towers)."
        "\n\n"
        "*P<sub>crowd</sub>* = 0 min (Low), 1.5 min (Medium), 4 min (High)."
    )

    routes = route_table(ridge, mlp, origin, candidates, week, weekday, departure_min)
    status = all_zone_status(ridge, mlp, week, weekday, departure_min, origin)
    worst_total = float(routes["total_mins"].max())

    st.markdown("### 🎯 Go Here Instead — top 3 picks")
    st.caption(
        f"Departure {from_minutes(departure_min)} · {need_label} · from {origin} · "
        f"wait-time error band ±{mae:.2f} min (Ridge test MAE)"
    )

    top3 = routes.head(3)
    columns = st.columns(3, gap="medium")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        color = CONGESTION_COLORS[row["level"]]
        saved = worst_total - row["total_mins"]
        badge = "⭐ BEST" if rank == 1 else f"#{rank}"
        footer = (
            f"saves {saved:.1f} min vs the busiest option"
            if saved > 0.05
            else "this is the busiest option right now"
        )
        dishes = zone_dishes(row["zone_id"], goal, category)
        menu_html = (
            f'<div class="menu">{"".join(f"<span class=\'tag\'>{d}</span>" for d in dishes)}</div>'
            if dishes
            else '<div class="menu"><span class="tag">quiet seat · seat search</span></div>'
        )
        columns[rank - 1].markdown(
            f"""
            <div class="route-card" style="border-top: 5px solid {color};">
              <div class="rank">{badge} · {row['service_type']}</div>
              <h3>{row['zone_id']}</h3>
              <div class="facts">
                🏢 <b>{row['floor_label']}</b><br/>
                🚶 walk <b>{row['walk_mins']:.1f} min</b> · arrive {row['arrival']}<br/>
                ⏳ predicted wait <b>{row['wait_pred']:.1f} ± {mae:.2f} min</b><br/>
                🏁 total <b>{row['total_mins']:.1f} min</b>
              </div>
              {menu_html}
              <span class="pill" style="background:{color}">{row['level']} crowd</span>
              <div class="saving" style="margin-top:10px;">{footer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🗺️ Campus connection network")
    st.caption(
        "Corridor network between the Kowloon Tong Halls and the three teaching towers, "
        "with the measured walking minutes on each link. Node colour and size = congestion level "
        "and predicted wait of that tower's canteens; hover a node for floors, wait time and dishes."
    )
    st.plotly_chart(network_figure(status, origin, category, goal, mae), width="stretch")

    st.markdown("### 🫧 Floor-level congestion")
    st.caption(
        "One bubble per zone inside its tower. Bubble size = predicted wait at your arrival moment, "
        "colour = congestion level. Hover for the floor, the wait and the routing reason."
    )
    st.plotly_chart(floor_map_figure(status, routes, need_label, mae), width="stretch")

    st.markdown("---")
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("#### 📊 Congestion board")
        board = status.sort_values("total_mins")
        fig = go.Figure(
            go.Bar(
                x=board["total_mins"],
                y=board["zone_id"],
                orientation="h",
                marker=dict(color=[CONGESTION_COLORS[lv] for lv in board["level"]]),
                text=[f"{lv} · {t:.1f} min" for lv, t in zip(board["level"], board["total_mins"])],
                textposition="inside", insidetextanchor="start",
                textfont=dict(color="white", size=11),
                customdata=board["wait_pred"],
                hovertemplate="<b>%{y}</b><br>total %{x:.1f} min · wait %{customdata:.1f} min<extra></extra>",
            )
        )
        fig.update_layout(
            height=420, margin=dict(l=6, r=10, t=10, b=10),
            xaxis_title="Total time from departure (minutes)",
            yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(gridcolor="#eef2f7")
        st.plotly_chart(fig, width="stretch")
        st.caption("Green = Low (<60 % seats), Yellow = Medium (60–85 %), Red = High (>85 %).")

    with right:
        st.markdown("#### 🧠 MLP softmax confidence")
        prob = status.sort_values("total_mins")
        prob_fig = go.Figure()
        for level in LEVEL_ORDER:
            prob_fig.add_trace(
                go.Bar(
                    x=prob[f"p_{level}"] * 100, y=prob["zone_id"], orientation="h",
                    name=level, marker_color=CONGESTION_COLORS[level],
                    hovertemplate=f"{level}: %{{x:.1f}} %<extra></extra>",
                )
            )
        prob_fig.update_layout(
            barmode="stack", height=420, margin=dict(l=6, r=10, t=10, b=10),
            xaxis_title="Predicted class probability (%)",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        prob_fig.update_xaxes(gridcolor="#eef2f7", range=[0, 100])
        st.plotly_chart(prob_fig, width="stretch")
        st.caption("One ReLU hidden layer (32 units), Adam + L2 weight decay, softmax output.")

    st.markdown("---")
    st.markdown("#### ⏱️ Multi-stage queue bottleneck breakdown")
    st.caption(
        "Canteen bottlenecks are sequential (order screen → food pickup → seat → utensils). "
        "The predicted total wait is split across those phases with the stopwatch structure "
        "weights measured on site, so you can see *where* the minutes actually go."
    )

    breakdown_rows = []
    for _, row in routes.iterrows():
        for phase, minutes in phase_breakdown(row["zone_id"], row["wait_pred"]):
            breakdown_rows.append(
                {
                    "zone": row["zone_id"],
                    "floor": row["floor_label"],
                    "phase": phase,
                    "est. minutes": minutes,
                    "total wait": round(row["wait_pred"], 2),
                    "congestion": row["level"],
                }
            )
    breakdown = pd.DataFrame(breakdown_rows)
    st.dataframe(
        breakdown, width="stretch", hide_index=True,
        column_config={
            "est. minutes": st.column_config.ProgressColumn(
                "est. minutes", min_value=0.0,
                max_value=float(max(5.0, breakdown["est. minutes"].max())),
                format="%.1f min",
            )
        },
    )

    st.markdown("#### 📈 Next 2 hours — where should you wait less?")
    horizon = list(range(int(departure_min), int(min(departure_min + 120, DAY_END_MIN)) + 1, SLOT_MINUTES))
    if len(horizon) < 2:
        horizon = [int(departure_min), int(min(departure_min + 120, 24 * 60 - 1))]
    line = go.Figure()
    palette = ["#1d4d7a", "#2a7f9e", "#4fa3a5", "#e8a317", "#d5402b", "#7d5ba6"]
    for idx, zone_id in enumerate(candidates):
        ys = [
            walk_time(origin, zone_id)
            + float(predict_zone(ridge, mlp, [zone_id], week, weekday, m)[["wait_pred"]].iloc[0, 0])
            for m in horizon
        ]
        line.add_trace(
            go.Scatter(
                x=[from_minutes(m) for m in horizon], y=ys, mode="lines+markers",
                name=zone_id, line=dict(width=2.4, color=palette[idx % len(palette)]),
                hovertemplate="%{x} → %{y:.1f} min total<extra>" + zone_id + "</extra>",
            )
        )
    line.update_layout(
        height=380, margin=dict(l=6, r=10, t=10, b=10),
        xaxis_title="Departure time", yaxis_title="Walk + predicted wait (minutes)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    line.update_xaxes(gridcolor="#eef2f7")
    line.update_yaxes(gridcolor="#eef2f7")
    st.plotly_chart(line, width="stretch")
    st.caption("Crowd penalty is excluded here so the raw queue dynamics stay visible.")

    with st.expander("🚶 Corridor network & canteen menus"):
        st.markdown("**Measured corridor links (minutes, both directions)**")
        corridor = pd.DataFrame(
            [{"from": a, "to": b, "minutes": m} for a, b, m in CORRIDOR_TABLE]
        )
        st.dataframe(corridor, width="stretch", hide_index=True)
        st.caption(
            "Elevator inside a tower: 1.5 min between any two floors. "
            "The app runs a shortest-path search over this network for every route."
        )

        st.markdown(f"**Door-to-door from {origin}**")
        walk_rows = [
            {
                "zone": ZONE_BY_ID[z]["zone_id"],
                "floor": ZONE_BY_ID[z]["floor_label"],
                "building": BUILDING_SHORT[ZONE_BY_ID[z]["building"]],
                "walk (min)": walk_time(origin, z),
            }
            for z in ZONE_IDS
        ]
        st.dataframe(pd.DataFrame(walk_rows), width="stretch", hide_index=True)

        st.markdown("**Canteen menus by food category**")
        menu_rows = []
        for building, dishes in CANTEEN_MENU.items():
            for dish, cats in dishes.items():
                menu_rows.append(
                    {
                        "canteen": BUILDING_SHORT[building],
                        "dish": dish,
                        "categories": " · ".join(c.split(" (")[0] for c in cats),
                    }
                )
        st.dataframe(pd.DataFrame(menu_rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 2
# --------------------------------------------------------------------------- #
def tab_evaluation(metrics: dict) -> None:
    st.markdown("### 📊 Model comparison (held-out week-3 second half)")

    splits = metrics["splits"]
    hyper = metrics["hyperparameters"]
    top = st.columns(4)
    top[0].metric("Train rows", f"{splits['train']['rows']}", f"week 1–2 · {splits['train']['share'] * 100:.1f} %",
                  delta_color="off")
    top[1].metric("Validation rows", f"{splits['validation']['rows']}",
                  f"week 3 first half · {splits['validation']['share'] * 100:.1f} %", delta_color="off")
    top[2].metric("Test rows", f"{splits['test']['rows']}",
                  f"week 3 second half · {splits['test']['share'] * 100:.1f} %", delta_color="off")
    top[3].metric("Ridge α / MLP α", f"{hyper['ridge_alpha']:g} / {hyper['mlp_alpha']:g}",
                  "tuned on validation", delta_color="off")

    rows = []
    for name, values in metrics["metrics"].items():
        rows.append(
            {
                "model": name,
                "type": values["kind"],
                "wait-time MAE (min)": round(values["mae_wait_mins"], 4),
                "congestion macro F1": round(values["macro_f1"], 4),
                "accuracy": round(values.get("accuracy", float("nan")), 4),
                "latency (ms / row)": round(values["latency_ms_per_row"], 4),
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True)

    fig_left, fig_right = st.columns(2, gap="large")
    with fig_left:
        mae_fig = go.Figure(
            go.Bar(
                x=table["model"], y=table["wait-time MAE (min)"],
                marker_color=["#8fa8c4", "#8fa8c4", "#1d4d7a"],
                text=[f"{v:.3f}" for v in table["wait-time MAE (min)"]],
                textposition="outside", hovertemplate="%{x}<br>MAE %{y:.4f} min<extra></extra>",
            )
        )
        mae_fig.update_layout(
            title="Wait-time MAE (minutes, lower is better)", height=380,
            margin=dict(l=6, r=10, t=50, b=10), plot_bgcolor="white",
            paper_bgcolor="white", showlegend=False,
        )
        mae_fig.update_yaxes(gridcolor="#eef2f7", rangemode="tozero")
        st.plotly_chart(mae_fig, width="stretch")

    with fig_right:
        f1_fig = go.Figure(
            go.Bar(
                x=table["model"], y=table["congestion macro F1"],
                marker_color=["#a9d3cd", "#a9d3cd", "#2a9e8f"],
                text=[f"{v:.3f}" for v in table["congestion macro F1"]],
                textposition="outside", hovertemplate="%{x}<br>Macro F1 %{y:.4f}<extra></extra>",
            )
        )
        f1_fig.update_layout(
            title="Congestion macro F1-score (higher is better)", height=380,
            margin=dict(l=6, r=10, t=50, b=10), plot_bgcolor="white",
            paper_bgcolor="white", showlegend=False,
        )
        f1_fig.update_yaxes(gridcolor="#eef2f7", rangemode="tozero", range=[0, 1.0])
        st.plotly_chart(f1_fig, width="stretch")

    verdict = metrics["verdict"]
    ok_mae = verdict["ridge_beats_baselines"]
    ok_f1 = verdict["mlp_beats_baselines"]
    col_a, col_b = st.columns(2)
    col_a.metric(
        "Ridge vs best baseline MAE",
        f"{metrics['metrics']['Ridge + MLPClassifier (selected)']['mae_wait_mins']:.3f} vs {verdict['best_baseline_mae']:.3f}",
        "beats both baselines ✅" if ok_mae else "does not beat the baseline ❌",
        delta_color="normal" if ok_mae else "inverse",
    )
    col_b.metric(
        "MLP vs best baseline macro F1",
        f"{metrics['metrics']['Ridge + MLPClassifier (selected)']['macro_f1']:.3f} vs {verdict['best_baseline_macro_f1']:.3f}",
        "beats both baselines ✅" if ok_f1 else "does not beat the baseline ❌",
        delta_color="normal" if ok_f1 else "inverse",
    )

    st.markdown("---")
    st.markdown("### 🎓 Course alignment")
    st.markdown(
        """
        | GE2340 concept | Where it lives in this project |
        | --- | --- |
        | Time-ordered / leakage-free splitting | `src/train_models.py::time_ordered_split` (week 1–2 train, week 3 split in half) |
        | Baseline thinking | Historical slot average, plain LinearRegression / LogisticRegression |
        | Regression with L2 regularisation | `Ridge`, α grid searched on the validation split |
        | Neural networks | `MLPClassifier((32,), relu, adam, alpha=weight decay)` |
        | Multi-class softmax + cross-entropy | MLP output layer; probabilities drive the confidence bars |
        | Feature engineering | one-hot zone / weekday / slot, zone×slot interaction, `dismissals / capacity` |
        | Evaluation metrics | Wait-time MAE (regression), Macro F1 + accuracy (classification), latency per row |
        | Deployment thinking | joblib pipelines, single-feature-frame serving path, latency measured on every model |
        | Graph algorithms (bonus) | Dijkstra over the corridor network drives every `T_walk` |
        """
    )

    st.markdown("### ⚠️ Known limitations")
    for item in metrics.get("limitations", []):
        st.markdown(f"- {item}")

    st.markdown(
        """
        <div class="note">
        <b>Why the error grows in some situations:</b> sudden thunderstorms push the whole
        campus indoors at once, and unregistered student activities (society fairs, career
        talks, exchange briefings) inject demand the model has never seen. Exam weeks also
        break the lunch-peak assumption entirely — the library becomes the bottleneck while
        the canteens empty out. The ±MAE band on the routing cards is the honest average
        error on the held-out week; treat a single prediction as an estimate, not a promise.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="CityUHK Smart Campus Router",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    hero()

    if not (MODEL_DIR / "metrics.json").exists():
        st.error(
            "No trained artefacts found. Run the two preparation steps first:\n\n"
            "```\npython src/generate_data.py\npython src/train_models.py\n```"
        )
        st.stop()

    ridge, mlp = load_models()
    metrics = load_metrics()

    tab1, tab2 = st.tabs(["🧭 Live Campus Router", "📊 Model Evaluation & Course Alignment"])
    with tab1:
        tab_router(ridge, mlp, metrics)
    with tab2:
        tab_evaluation(metrics)

    st.markdown("---")
    st.caption(
        f"{PROJECT_NAME} · {PROJECT_SUBTITLE} · demo data is an aggregate synthetic field survey "
        "of 960 zone-time observations (no personal data) · models trained on week 1–2, "
        "evaluated on week 3."
    )


if __name__ == "__main__":
    main()
