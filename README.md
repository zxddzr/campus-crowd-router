# Smart Campus Crowd-Balancing & Space Routing Engine

**GE2340 · City University of Hong Kong** · mobile-first, demo · 

**https://cityuhk-crowd-router.streamlit.app/**

Predict the **queue you are about to walk into** — not the queue you left behind.
The website predicts waiting time (regression) and congestion level
`Low / Medium / High` (3-class) for nine CityUHK canteens and study areas,
walks you there through the real corridor network, and ranks the options:

```
Total(z) = T_walk(origin, z) + Ŵ(z, t + T_walk) + P_crowd(z)
```

---

## What changed in v2.0

| Area | Before | Now |
| --- | --- | --- |
| Layout | left sidebar | **no sidebar** — all inputs in one compact top block |
| Mobile | 2-column tables, chart gestures | 375 px-first, charts do not hijack the swipe, phase tables collapsed into `expander`s |
| Views | one mixed page | **Tab 1 student view** (no metrics) / **Tab 2 AI Model & Data** (defence view) |
| Language | English only | **简体中文 / 繁體中文 / English**, no mixed strings |
| Location | 4 fixed buildings | **building + floor picker** (AC1 1–8/F, AC2 B2–9/F, AC3 G/F–20/F) with a real walking route per card |
| Time | 09:00–17:00, 5 weekdays | **07:30–22:00, Mon–Sun**, with `Now / +15 / +30` quick picks |
| Opening hours | none | canteens 07:30–21:00, **AC3 closed on Sundays** → marked *Closed* and removed from the top 3 |
| Data | 960 rows, weekdays only | **2 889 rows**, 7 days, 18 snapshots/day, weekend demand modelled lower |

---

## Two views, one model

**Tab 1 · 校园实时导航 / Live Campus Router** — student facing.
Input block → three recommendation cards (walking time, arrival clock, queue,
dish tags, one-line route, opening hours) → one campus congestion map.
No MAE, no F1, no probabilities.

**Tab 2 · AI 模型与数据 / AI Model & Data** — defence facing.
Model comparison table and charts, softmax probability distribution, data
provenance, coverage, course alignment and limitations.

---

## Campus geography

| Key | Building | Floors | Zones modelled |
| --- | --- | --- | --- |
| **AC1 / YEUNG** | Yeung Kin Man Academic Building | 8 | **AC1 Canteen (5/F)**, Library 3/F, Library 4/F Quiet Zone, YEUNG 4/F Study Corner |
| **AC2 / LI** | Li Dak Sum Yip Yio Chin Academic Building | 9 (B2–9/F) | **AC2 Canteen (3/F)** — semi-buffet + counter |
| **AC3 / LAU** | Lau Ming Wai Academic Building | 20 | **AC3 Canteen / Bistro (7/F)** — semi-buffet + counter, LAU 4/F Study Lounge |
| **Halls** | Student Residence, Kowloon Tong | — | departure point only |

### Walking network (Dijkstra over a floor graph)

| Link | Minutes |
| --- | --- |
| any two floors, same tower (lift) | 1.5 |
| AC1 4/F ↔ AC2 B2 | 5 |
| AC1 4/F ↔ AC3 G/F | 7 |
| AC1 5/F ↔ AC2 B2 | 5 |
| AC1 5/F ↔ AC3 3/F | 5 |
| AC2 4/F ↔ AC3 5/F | 3 |
| AC2 G/F ↔ AC3 3/F | 1 |
| Halls ↔ AC3 / AC2 / AC1 | 12 / 12 / 17 |

Door-to-door examples: AC1 4/F → AC1 Canteen **1.5**, AC1 4/F → AC2 3/F **6.5**,
AC1 4/F → AC3 7/F **8.0**, AC2 G/F → AC3 7/F **2.5**, Halls → AC1 Canteen **18.0**.
Each card prints the route, e.g. *乘電梯至 5/F → 經連廊至 AC3 3/F → 乘電梯至 7/F*.

### Opening hours

| Canteen | Hours | Days |
| --- | --- | --- |
| AC1 Canteen (5/F) | 07:30–21:00 | Mon–Sun |
| AC2 Canteen (3/F) | 07:30–21:00 | Mon–Sun |
| AC3 Canteen / Bistro (7/F) | 07:30–21:00 | **Mon–Sat only** |
| Study areas | 07:30–22:00 | Mon–Sun |

The check uses the **arrival** time, not the departure time: leave at 20:50 and
walk 8 minutes and the canteen is already closed.

### Food categories

| Category | AC1 | AC2 | AC3 |
| --- | :-: | :-: | :-: |
| Any (fastest) | ✅ | ✅ | ✅ |
| Rice & quick meals | ✅ | ✅ | ✅ |
| Noodles & soup | ✅ | ✅ | ✅ |
| Western & fast food | ✅ | ✅ | — |
| Asian & curry | ✅ | — | ✅ |

Dishes: AC1 — 烧味 / 两餸饭 / 车仔米线 / 拉面&牛肉饭 / Pizza / 炸鸡 / 东南亚菜 ·
AC2 — 两餸饭 / 米线 / 麻辣烫 / 饺子 / 意粉&焗饭 ·
AC3 — 半自助（称重+米饭）/ 咖喱饭 / 特色面食.

---

## Results

Held-out **week-3 second half** (482 rows the models never saw).

| Model | Role | Wait-time MAE (min) ↓ | Macro F1 ↑ | Accuracy | Latency (ms/row) |
| --- | --- | --- | --- | --- | --- |
| Historical Slot Average | baseline | 1.605 | 0.727 | 0.892 | 0.001 |
| Linear + Logistic Regression | baseline | 0.930 | 0.711 | 0.894 | 0.019 |
| **Ridge + MLPClassifier** | **selected** | **0.663** | **0.750** | **0.911** | 0.022 |

Both selected models beat both baselines on both metrics. Macro F1 is lower than
v1 because the dataset now spans the whole day: roughly three quarters of all
observations are *Quiet*, so the macro average is dominated by the two small
classes. Accuracy on the same split is 0.91.

| Component | Setting |
| --- | --- |
| Regression | `Ridge(alpha=0.01)`, α grid searched on validation |
| Classification | `MLPClassifier((32,), relu, adam, alpha=1e-05, batch_size=16, lr=0.003, early_stopping=True)` |
| Split | week 1–2 train (1 926) · week 3 first half validation (481) · week 3 second half test (482) |

---

## Data provenance — read this before quoting the numbers

> **Simulated timetable & field-calibrated stopwatch data.**

This is **not** a live sensor feed and **not** a real attendance record.

* The demand curve (breakfast 08:20, lunch 12:15 + the 11:50 and 12:50
  class-dismissal waves, afternoon 15:50, dinner 18:20) is a *shape* taken from
  a typical CityUHK timetable.
* Phase durations (order screen / food pickup / seat / utensils) are calibrated
  against stopwatch observations of the canteen bottlenecks.
* Opening hours, the Sunday AC3 closure and the weekend demand drop are explicit
  rules, not learned behaviour.
* No student, device, card or Wi-Fi data is used anywhere.  Every row is an
  aggregate: `week, weekday, time_slot, zone_id, service_type, zone_capacity,
  nearby_class_dismissals, wait_time_mins, congestion_level`.

Coverage: **3 teaching weeks × Mon–Sun × 07:30–21:30 snapshots (18 per day)**,
2 889 rows.  Weekends are generated with a ≈0.55–0.62 demand factor and no
class-dismissal waves; AC3 produces **zero** Sunday rows because it is closed.

Reproducibility: fixed numpy seed (`20241122`).

---

## Repository layout

```
.
├── app.py                     # mobile-first dashboard, 2 tabs, 3 locales
├── requirements.txt
├── README.md
├── .gitignore
├── data/cityuhk_campus_data.csv
├── models/{preprocessor,ridge_model,mlp_model}.joblib + metrics.json
├── src/
│   ├── campus_config.py       # zones, floors, corridor graph, menus, service hours
│   ├── i18n.py                # zh_Hans / zh_Hant / en dictionaries + accessors
│   ├── features.py            # shared feature engineering + sklearn preprocessors
│   ├── generate_data.py       # full-week, full-day synthetic survey generator
│   └── train_models.py        # chronological split, baselines, tuning, metrics
└── tests/smoke_test.py
```

---

## Known limitations

- **Synthetic data.** Calibrated, but a model of the campus, not a measurement.
- **Rain and events.** A thunderstorm or an unregistered society fair can push
  real waits outside the training range.
- **Macro F1 is class-imbalanced.** The `High` class is ~11 % of the data, so a
  single misclassification moves the macro average noticeably.
- **Three teaching weeks only.** Semester start and exam weeks behave differently.
- **Maps are schematic.** Node positions are a readable 2-D layout, not
  geo-referenced coordinates; every walking time comes from the corridor graph.
- **Menus rotate by term**, so the dish list is representative rather than live.
- **Phase breakdowns are structure-weighted estimates**, not separate predictions.

---

## Course alignment (GE2340)

Chronological leakage-free split · baseline thinking · L2 regularisation with
grid search · one-hidden-layer neural network · softmax + cross-entropy ·
feature engineering (interaction terms, log/density transforms, load =
headcount / capacity) · regression + classification metrics with latency ·
Dijkstra graph search for walking time · joblib deployment.
The full mapping is rendered inside Tab 2 in all three languages.

---

## License / attribution

Course project for GE2340 (City University of Hong Kong). Code and synthetic
dataset are released for academic demonstration purposes.
