"""CityUHK Smart Campus Crowd-Balancing & Space Routing Engine — mobile-first dashboard.

Run::

    streamlit run app.py

Layout
------
* No sidebar.  Every input sits in a compact block at the top of the page so it
  stays reachable on a 375 px phone screen.
* **Tab 1 · Live Campus Router** — student view: three recommendation cards with
  the walking route in plain language, plus one congestion map.  No metrics.
* **Tab 2 · AI Model & Data** — defence view: model comparison, softmax
  distribution, data provenance and limitations.

Three locales (简体中文 / 繁體中文 / English) are resolved through ``src/i18n.py``;
no hard-coded UI string lives in this file.
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
    BUILDING_BOXES,
    BUILDING_FLOORS,
    CLOSED_COLOR,
    CONGESTION_COLORS,
    CONGESTION_LEVELS,
    CROWD_PENALTY,
    DAY_END_MIN,
    DAY_START_MIN,
    FEATURE_COLUMNS,
    FLOOR_CHOICES,
    FOOD_CATEGORIES,
    HALLS,
    SLOT_MINUTES,
    STUDY_GOAL,
    WEEKDAYS,
    ZONES,
    ZONE_BY_ID,
    ZONE_IDS,
    canteen_dishes,
    expected_dismissals,
    from_minutes,
    is_service_open,
    opening_label,
    phase_breakdown,
    route_steps,
    walk_time,
    zones_for_goal,
)
from src.i18n import (  # noqa: E402
    COURSE_TABLE,
    DEFAULT_LANG,
    LANG_LABELS,
    LANGS,
    building_label,
    building_short,
    category_hint,
    category_name,
    day_name,
    dish_name,
    level_name,
    phase_name,
    tr,
    zone_name,
    zone_short,
)

MODEL_DIR = PROJECT_ROOT / "models"
CITY_TZ = ZoneInfo("Asia/Hong_Kong")

# Plotly must never swallow a finger swipe on a phone: no mode bar, no scroll
# zoom, no drag-to-pan.  Hover tooltips still work on desktop.
CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "staticPlot": False,
}

TIME_OPTIONS = list(range(DAY_START_MIN, DAY_END_MIN + 1, SLOT_MINUTES))
GOAL_FOOD = "food"
GOAL_STUDY = STUDY_GOAL


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load(MODEL_DIR / "ridge_model.joblib"), joblib.load(MODEL_DIR / "mlp_model.joblib")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as handle:
        return json.load(handle)


def nearest_slot(minute: float) -> int:
    slot = int(round(minute / SLOT_MINUTES)) * SLOT_MINUTES
    return max(DAY_START_MIN, min(DAY_END_MIN, slot))


def any_canteen_open(weekday: str, minute: int) -> bool:
    return any(
        is_service_open(z["zone_id"], weekday, minute)
        for z in ZONES
        if z["service_type"] != "Study"
    )


def feature_frame(zone_ids: list[str], weekday: str, minute: float) -> pd.DataFrame:
    rows = []
    for zone_id in zone_ids:
        zone = ZONE_BY_ID[zone_id]
        rows.append(
            {
                "week": 3,
                "weekday": weekday,
                "time_slot": from_minutes(int(round(minute))),
                "zone_id": zone_id,
                "service_type": zone["service_type"],
                "zone_capacity": zone["capacity"],
                "nearby_class_dismissals": expected_dismissals(minute),
            }
        )
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


def predict(ridge, mlp, zone_ids: list[str], weekday: str, minute: float) -> pd.DataFrame:
    frame = feature_frame(zone_ids, weekday, minute)
    wait = np.clip(np.asarray(ridge.predict(frame), dtype=float), 0.0, None)
    proba = np.asarray(mlp.predict_proba(frame), dtype=float)
    classes = list(mlp.named_steps["model"].classes_)
    out = pd.DataFrame({"zone_id": frame["zone_id"].to_numpy(), "wait": wait})
    for level in CONGESTION_LEVELS:
        out[f"p_{level}"] = proba[:, classes.index(level)] if level in classes else 0.0
    out["level"] = [
        CONGESTION_LEVELS[int(np.argmax(row[[f"p_{lv}" for lv in CONGESTION_LEVELS]]))]
        for _, row in out.iterrows()
    ]
    return out


def build_routes(ridge, mlp, building: str, floor: str, goal: str, weekday: str,
                 depart: int) -> pd.DataFrame:
    """Walk + queue at arrival + crowd penalty, for every candidate zone."""
    candidates = zones_for_goal(goal)
    rows = []
    for zone_id in candidates:
        minutes, steps = route_steps(building, floor, zone_id)
        arrival = depart + minutes
        pred = predict(ridge, mlp, [zone_id], weekday, arrival).iloc[0]
        is_open = is_service_open(zone_id, weekday, arrival)
        penalty = CROWD_PENALTY[pred["level"]] if is_open else 0.0
        rows.append(
            {
                "zone_id": zone_id,
                "walk": minutes,
                "steps": steps,
                "arrival": from_minutes(int(round(arrival))),
                "arrival_min": arrival,
                "wait": float(pred["wait"]),
                "level": pred["level"],
                "p_Low": float(pred["p_Low"]),
                "p_Medium": float(pred["p_Medium"]),
                "p_High": float(pred["p_High"]),
                "open": is_open,
                "total": minutes + float(pred["wait"]) + penalty if is_open else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def route_text(lang: str, steps: list[dict]) -> str:
    parts = []
    for step in steps:
        if step["kind"] == "lift":
            parts.append(tr(lang, "route.lift", floor=step["floor"]))
        else:
            tag = {"YEUNG": "AC1", "LI": "AC2", "LAU": "AC3"}.get(step["building"], step["building"])
            parts.append(tr(lang, "route.corridor", place=f"{tag} {step['floor']}"))
    return " → ".join(parts)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
  .block-container {padding-top: 3.9rem; padding-bottom: 2.5rem; max-width: 1100px;}
  section[data-testid="stSidebar"] {display: none;}
  header[data-testid="stHeader"] {background: rgba(255,255,255,.85); backdrop-filter: blur(4px);}
  .hero {background: linear-gradient(120deg, #10233f 0%, #1d4d7a 60%, #2a7f9e 100%);
         border-radius: 16px; padding: 18px 20px; color: #fff; margin: 6px 0 14px;}
  .hero h1 {margin: 0; font-size: clamp(1.05rem, 3.4vw, 1.4rem); line-height: 1.3;}
  .hero p {margin: 6px 0 0; opacity: .85; font-size: clamp(.74rem, 2.4vw, .86rem);}
  .card {background:#fff; border-radius:14px; padding:14px 16px; margin-bottom:10px;
         box-shadow:0 2px 12px rgba(16,35,63,.10);}
  .card .top {display:flex; justify-content:space-between; align-items:center; gap:8px;}
  .card .rank {font-size:.7rem; letter-spacing:.1em; text-transform:uppercase;
               color:#6b7b8f; font-weight:700;}
  .card .name {font-size:clamp(.95rem,3vw,1.05rem); font-weight:700; color:#10233f; margin:4px 0 8px;}
  .card .badge {padding:2px 9px; border-radius:999px; color:#fff; font-size:.68rem; font-weight:700;}
  .card .row {font-size:.82rem; color:#35485f; line-height:1.6;}
  .card .total {font-size:.95rem; font-weight:700; color:#10233f; margin:6px 0;}
  .card .total small {font-weight:400; color:#6b7b8f; font-size:.76rem;}
  .card .tags {margin:6px 0 8px;}
  .card .tag {display:inline-block; background:#eef4fa; color:#31465e; border-radius:6px;
              padding:2px 7px; font-size:.7rem; margin:2px 3px 0 0; border:1px solid #dbe6f0;}
  .card .route {font-size:.78rem; color:#41556d; background:#f5f8fc; border-radius:8px;
                padding:6px 9px; margin-bottom:6px;}
  .card .save {font-size:.74rem; color:#1f7a4d; font-weight:600;}
  .card.closed {opacity:.55;}
  .warn {background:#fff6ec; border-left:4px solid #e8a317; border-radius:8px;
         padding:12px 14px; font-size:.85rem; color:#5a4a2a; margin:8px 0;}
  .note {background:#f4f7fb; border-left:4px solid #2a7f9e; border-radius:8px;
         padding:12px 14px; font-size:.84rem; color:#35485f;}
  div[data-testid="stMetricValue"] {font-size:1.15rem;}
  @media (max-width: 640px) {
    .block-container {padding-left: .8rem; padding-right: .8rem;}
    div[data-testid="stHorizontalBlock"] {flex-direction: column; gap: .4rem;}
    div[data-testid="stHorizontalBlock"] > div {width: 100% !important;}
  }
</style>
"""


def hero(lang: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>🏫 {tr(lang, "app.title")}</h1>
          <p>{tr(lang, "app.tagline")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Input block
# --------------------------------------------------------------------------- #
def _set_depart(value: int) -> None:
    st.session_state.depart = value


def input_block(lang: str) -> dict:
    if "depart" not in st.session_state:
        now = datetime.now(CITY_TZ)
        now_min = now.hour * 60 + now.minute
        weekday = WEEKDAYS[min(now.weekday(), 6)]
        slot = nearest_slot(now_min)
        st.session_state.depart = slot if any_canteen_open(weekday, slot) else nearest_slot(12 * 60)

    with st.container(border=True):
        row1 = st.columns([1, 1])
        with row1[0]:
            building = st.selectbox(
                tr(lang, "in.location"),
                list(BUILDING_FLOORS) + [HALLS],
                format_func=lambda b: building_label(lang, b),
            )
        with row1[1]:
            if building == HALLS:
                floor = "G"
                st.selectbox(tr(lang, "in.floor"), ["—"], disabled=True,
                             format_func=lambda _v: "—")
            else:
                floor = st.selectbox(tr(lang, "in.floor"), FLOOR_CHOICES[building])

        row2 = st.columns([1, 1])
        with row2[0]:
            goal_choice = st.radio(
                tr(lang, "in.goal"),
                [GOAL_FOOD, GOAL_STUDY],
                format_func=lambda g: tr(lang, f"goal.{g}"),
                horizontal=True,
            )
        with row2[1]:
            if goal_choice == GOAL_FOOD:
                category = st.selectbox(
                    tr(lang, "in.category"),
                    list(FOOD_CATEGORIES),
                    format_func=lambda c: category_name(lang, c),
                )
            else:
                category = None
                st.selectbox(tr(lang, "in.category"), ["—"], disabled=True,
                             format_func=lambda _v: "—")

        row3 = st.columns([1, 1])
        with row3[0]:
            weekday = st.selectbox(
                tr(lang, "in.day"),
                list(WEEKDAYS),
                format_func=lambda d: day_name(lang, d),
            )
        with row3[1]:
            depart = st.select_slider(
                tr(lang, "in.time"),
                options=TIME_OPTIONS,
                key="depart",
                format_func=from_minutes,
            )

        # Quick-pick buttons live *below* the slider but drive it through
        # callbacks, which run before the widget is rebuilt.
        now_dt = datetime.now(CITY_TZ)
        now_slot = nearest_slot(now_dt.hour * 60 + now_dt.minute)
        st.caption(tr(lang, "in.quick"))
        quick = st.columns(3)
        quick[0].button(tr(lang, "q.now"), width="stretch",
                        on_click=_set_depart, args=(now_slot,))
        quick[1].button(tr(lang, "q.p15"), width="stretch",
                        on_click=_set_depart, args=(nearest_slot(min(now_slot + 15, DAY_END_MIN)),))
        quick[2].button(tr(lang, "q.p30"), width="stretch",
                        on_click=_set_depart, args=(nearest_slot(min(now_slot + 30, DAY_END_MIN)),))

    hint = category_hint(lang, category) if category else ""
    if hint:
        st.caption(hint)

    return {
        "building": building,
        "floor": floor,
        "goal": category if goal_choice == GOAL_FOOD else GOAL_STUDY,
        "goal_choice": goal_choice,
        "weekday": weekday,
        "depart": int(depart),
    }


# --------------------------------------------------------------------------- #
# Tab 1 — student view
# --------------------------------------------------------------------------- #
def card_html(lang: str, row: pd.Series, rank: int, worst: float) -> str:
    color = CONGESTION_COLORS[row["level"]]
    zone = ZONE_BY_ID[row["zone_id"]]
    title = zone_name(lang, row["zone_id"])
    label = tr(lang, "res.best") if rank == 1 else tr(lang, "res.rank", n=rank)

    if row["open"]:
        dishes = canteen_dishes(zone["building"], row["category"]) if row.get("category") else ()
        tags = "".join(f'<span class="tag">{dish_name(lang, d)}</span>' for d in dishes)
        if not tags:
            tags = f'<span class="tag">{tr(lang, "res.study")}</span>'
        saved = worst - row["total"]
        footer = (
            tr(lang, "res.saves", n=f"{saved:.1f}")
            if np.isfinite(worst) and saved > 0.05
            else tr(lang, "res.busiest")
        )
        hours = opening_label(row["zone_id"])
        queue_line = (
            f'{tr(lang, "res.queue")} <b>{row["wait"]:.1f}</b> {tr(lang, "res.min")} · '
            f'{tr(lang, "res.total")} <b>{row["total"]:.1f}</b> {tr(lang, "res.min")}'
        )
        footer = f"{footer} · {hours}"
    else:
        tags = ""
        footer = f'{tr(lang, "st.closed")} · {opening_label(row["zone_id"])}'
        queue_line = tr(lang, "st.closed")

    return f"""
    <div class="card{' closed' if not row['open'] else ''}" style="border-top:4px solid {color};">
      <div class="top">
        <span class="rank">{label}</span>
        <span class="badge" style="background:{color}">{level_name(lang, row["level"])}</span>
      </div>
      <div class="name">{title}</div>
      <div class="row">
        🚶 {tr(lang, "res.walk")} <b>{row["walk"]:.1f}</b> {tr(lang, "res.min")} ·
        {tr(lang, "res.arrive")} {row["arrival"]}
      </div>
      <div class="total">{queue_line}</div>
      <div class="tags">{tags}</div>
      <div class="route">🧭 {route_text(lang, row["steps"])}</div>
      <div class="save">{footer}</div>
    </div>
    """


def campus_figure(lang: str, status: pd.DataFrame, top_ids: list[str]) -> go.Figure:
    fig = go.Figure()
    for building, box in BUILDING_BOXES.items():
        fig.add_shape(
            type="rect", x0=box["x0"], x1=box["x1"], y0=box["y0"], y1=box["y1"],
            line=dict(color="#c3d0dd", width=1.2), fillcolor="#f4f7fa", layer="below",
        )
        fig.add_annotation(
            x=(box["x0"] + box["x1"]) / 2, y=box["y1"],
            text=building_short(lang, building), showarrow=False,
            font=dict(size=10, color="#5b6f86"), yshift=12,
        )

    sizes, colors, texts, hovers = [], [], [], []
    for _, row in status.iterrows():
        if not row["open"]:
            sizes.append(14)
            colors.append(CLOSED_COLOR)
            state = f"{tr(lang, 'st.closed')} · {opening_label(row['zone_id'])}"
        else:
            sizes.append(14 + 3.4 * min(row["wait"], 12.0))
            colors.append(CONGESTION_COLORS[row["level"]])
            state = f"⏳ {row['wait']:.1f} {tr(lang, 'res.min')} · {level_name(lang, row['level'])}"
        short = zone_short(lang, row["zone_id"])
        texts.append(short)
        badge = f"⭐ {tr(lang, 'res.best')}" if row["zone_id"] in top_ids[:1] else ""
        hovers.append(
            f"<b>{zone_name(lang, row['zone_id'])}</b><br>{state}" + (f"<br>{badge}" if badge else "")
        )

    fig.add_trace(
        go.Scatter(
            x=[ZONE_BY_ID[z]["coords"][0] for z in status["zone_id"]],
            y=[ZONE_BY_ID[z]["coords"][1] for z in status["zone_id"]],
            mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(color="white", width=1.6), opacity=0.94),
            text=texts, textposition="bottom center",
            textfont=dict(size=9, color="#41556d"),
            hovertext=hovers, hoverinfo="text", showlegend=False, cliponaxis=False,
        )
    )
    return fig.update_layout(
        height=340,
        margin=dict(l=4, r=4, t=8, b=4),
        plot_bgcolor="white", paper_bgcolor="white",
        dragmode=False,
        xaxis=dict(range=[-1.15, 3.75], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.95, 1.55], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
    )


def tab_router(lang: str, ctx: dict, ridge, mlp) -> None:
    routes = build_routes(ridge, mlp, ctx["building"], ctx["floor"], ctx["goal"],
                          ctx["weekday"], ctx["depart"])
    # Attach the food category so the cards can list the right dishes.
    routes = routes.assign(category=ctx["goal"] if ctx["goal"] != GOAL_STUDY else None)

    open_rows = routes[routes["open"]].sort_values("total").reset_index(drop=True)
    closed_rows = routes[~routes["open"]].reset_index(drop=True)

    st.markdown(f"#### {tr(lang, 'res.title')}")
    st.caption(
        tr(lang, "res.sub", time=from_minutes(ctx["depart"]), day=day_name(lang, ctx["weekday"]))
    )

    if open_rows.empty:
        st.markdown(
            f'<div class="warn"><b>{tr(lang, "res.none_open", time=from_minutes(ctx["depart"]), day=day_name(lang, ctx["weekday"]))}</b><br/>'
            f'{tr(lang, "res.none_open.hint")}</div>',
            unsafe_allow_html=True,
        )
        if st.button(tr(lang, "res.jump_lunch"), width="stretch"):
            st.session_state.depart = nearest_slot(12 * 60)
            st.rerun()
    else:
        worst = float(open_rows["total"].max())
        top = open_rows.head(3)
        columns = st.columns(min(3, len(top)))
        for idx, (_, row) in enumerate(top.iterrows()):
            with columns[idx]:
                st.markdown(card_html(lang, row, idx + 1, worst), unsafe_allow_html=True)
                with st.expander(tr(lang, "res.phases")):
                    phases = phase_breakdown(row["zone_id"], row["wait"])
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    tr(lang, "res.phases"): phase_name(lang, label),
                                    tr(lang, "res.min"): minutes,
                                }
                                for label, minutes in phases
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )

        if len(closed_rows):
            names = "、".join(zone_name(lang, z) for z in closed_rows["zone_id"]) if lang != "en" else ", ".join(zone_name(lang, z) for z in closed_rows["zone_id"])
            st.caption(f'{tr(lang, "st.closed")}: {names}')

    st.markdown(f"#### {tr(lang, 'campus.title')}")
    st.caption(tr(lang, "campus.sub"))
    status = routes.copy()
    st.plotly_chart(
        campus_figure(lang, status, list(open_rows["zone_id"])),
        width="stretch",
        config=CHART_CONFIG,
    )


# --------------------------------------------------------------------------- #
# Tab 2 — model & data
# --------------------------------------------------------------------------- #
MODEL_SHORT_KEY = {
    "Historical Slot Average (baseline)": "m.model.slot",
    "Linear Regression + Logistic Regression (baseline)": "m.model.linear",
    "Ridge + MLPClassifier (selected)": "m.model.selected",
}


def tab_model(lang: str, ctx: dict, ridge, mlp, metrics: dict) -> None:
    st.markdown(f"### {tr(lang, 'm.title')}")
    st.markdown(tr(lang, "m.intro"))
    st.caption(tr(lang, "m.split"))

    rows = []
    for name, values in metrics["metrics"].items():
        rows.append(
            {
                tr(lang, "m.col.model"): (
                    tr(lang, MODEL_SHORT_KEY[name]) if name in MODEL_SHORT_KEY else name
                ),
                tr(lang, "m.col.role"): tr(lang, f"m.role.{values['kind']}"),
                tr(lang, "m.col.mae"): round(values["mae_wait_mins"], 3),
                tr(lang, "m.col.f1"): round(values["macro_f1"], 3),
                tr(lang, "m.col.acc"): round(values.get("accuracy", float("nan")), 3),
                tr(lang, "m.col.lat"): round(values["latency_ms_per_row"], 4),
            }
        )
    table = pd.DataFrame(rows)
    # Phones cannot scroll a six-column frame comfortably: show the two headline
    # metrics inline (short headers) and tuck accuracy + latency behind one tap.
    headline = pd.DataFrame(
        {
            tr(lang, "m.col.model"): table[tr(lang, "m.col.model")],
            tr(lang, "m.col.mae.short"): table[tr(lang, "m.col.mae")],
            tr(lang, "m.col.f1.short"): table[tr(lang, "m.col.f1")],
        }
    )
    st.dataframe(headline, hide_index=True, width="stretch")
    with st.expander(tr(lang, "m.more")):
        st.dataframe(
            table[[tr(lang, "m.col.model"), tr(lang, "m.col.role"),
                   tr(lang, "m.col.acc"), tr(lang, "m.col.lat")]],
            hide_index=True, width="stretch",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        mae_fig = go.Figure(
            go.Bar(
                x=table[tr(lang, "m.col.model")], y=table[tr(lang, "m.col.mae")],
                marker_color=["#8fa8c4", "#8fa8c4", "#1d4d7a"],
                text=[f"{v:.2f}" for v in table[tr(lang, "m.col.mae")]], textposition="outside",
            )
        )
        mae_fig.update_layout(
            title=tr(lang, "m.chart.mae"), height=300, showlegend=False, dragmode=False,
            margin=dict(l=4, r=4, t=40, b=4), plot_bgcolor="white", paper_bgcolor="white",
        )
        mae_fig.update_xaxes(tickfont=dict(size=9))
        st.plotly_chart(mae_fig, width="stretch", config=CHART_CONFIG)

    with col_b:
        f1_fig = go.Figure(
            go.Bar(
                x=table[tr(lang, "m.col.model")], y=table[tr(lang, "m.col.f1")],
                marker_color=["#a9d3cd", "#a9d3cd", "#2a9e8f"],
                text=[f"{v:.2f}" for v in table[tr(lang, "m.col.f1")]], textposition="outside",
            )
        )
        f1_fig.update_layout(
            title=tr(lang, "m.chart.f1"), height=300, showlegend=False, dragmode=False,
            margin=dict(l=4, r=4, t=40, b=4), plot_bgcolor="white", paper_bgcolor="white",
        )
        f1_fig.update_yaxes(range=[0, 1.0])
        f1_fig.update_xaxes(tickfont=dict(size=9))
        st.plotly_chart(f1_fig, width="stretch", config=CHART_CONFIG)

    verdict = metrics["verdict"]
    if verdict["ridge_beats_baselines"] and verdict["mlp_beats_baselines"]:
        st.success(tr(lang, "m.verdict"))
    else:
        st.warning(tr(lang, "m.verdict.no"))

    st.markdown(f"#### {tr(lang, 'm.softmax')}")
    st.caption(tr(lang, "m.softmax.sub"))
    prob = predict(ridge, mlp, list(ZONE_IDS), ctx["weekday"], ctx["depart"])
    prob = prob.assign(open=[is_service_open(z, ctx["weekday"], ctx["depart"]) for z in prob["zone_id"]])
    prob = prob.sort_values("wait", ascending=False)
    soft = go.Figure()
    for level in CONGESTION_LEVELS:
        soft.add_trace(
            go.Bar(
                y=[zone_name(lang, z) for z in prob["zone_id"]],
                x=prob[f"p_{level}"] * 100,
                orientation="h", name=level_name(lang, level),
                marker_color=CONGESTION_COLORS[level],
            )
        )
    soft.update_layout(
        barmode="stack", height=360, dragmode=False,
        margin=dict(l=4, r=4, t=10, b=4), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    soft.update_xaxes(range=[0, 100], title="%", gridcolor="#eef2f7")
    soft.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(soft, width="stretch", config=CHART_CONFIG)

    st.markdown(f"#### {tr(lang, 'm.data')}")
    data_rows = metrics["dataset"]["rows"]
    st.markdown(
        f"""
        <div class="note">
          <b>{tr(lang, "m.data.src")}</b><br/>
          {tr(lang, "m.data.cov")}: {tr(lang, "m.data.cov.value")}<br/>
          {tr(lang, "m.data.rows")}: <b>{data_rows}</b><br/><br/>
          {tr(lang, "m.data.note")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"#### {tr(lang, 'm.course')}")
    st.dataframe(
        pd.DataFrame(
            [
                {tr(lang, "m.course.head"): k, tr(lang, "m.course.body"): v}
                for k, v in COURSE_TABLE.get(lang, COURSE_TABLE["en"]).items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown(f"#### {tr(lang, 'm.limit')}")
    for item in metrics.get("limitations", []):
        st.markdown(f"- {item}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="CityUHK Smart Campus Router",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.session_state.lang = DEFAULT_LANG
    picked = st.segmented_control(
        tr(DEFAULT_LANG, "lang.label"),
        options=list(LANGS),
        format_func=lambda key: LANG_LABELS[key],
        default=st.session_state.lang,
        label_visibility="collapsed",
    )
    lang = picked or DEFAULT_LANG
    st.session_state.lang = lang

    hero(lang)

    if not (MODEL_DIR / "metrics.json").exists():
        st.error("Run `python src/generate_data.py` and `python src/train_models.py` first.")
        st.stop()

    ridge, mlp = load_models()
    metrics = load_metrics()

    ctx = input_block(lang)

    tab1, tab2 = st.tabs([tr(lang, "tab.router"), tr(lang, "tab.model")])
    with tab1:
        tab_router(lang, ctx, ridge, mlp)
    with tab2:
        tab_model(lang, ctx, ridge, mlp, metrics)


if __name__ == "__main__":
    main()
