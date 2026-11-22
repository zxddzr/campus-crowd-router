# Smart Campus Crowd-Balancing & Space Routing Engine

**GE2340 · City University of Hong Kong** · lightweight, fully local demo

Predict the **queue you are about to walk into** — not the queue you left behind.
The app predicts waiting time (regression) and congestion level
`Low / Medium / High` (3-class classification) for nine CityUHK canteens and
study areas during the 11:30–14:30 lunch peak, then combines that with walking
time and a crowd penalty to answer one question:

> *Given where I am, what I need, and when I leave — **where should I go instead**?*

---

## Campus geography used by the model

| Key | Building | Floors modelled | Zones used here |
| --- | --- | --- | --- |
| **AC1 / YEUNG** | Yeung Kin Man Academic Building (main teaching building, linked to Festival Walk) | 4/F = LT lecture-theatre zone + Library entrance · **5/F = AC1 Canteen** | AC1 Canteen (5/F), YEUNG 4/F Study Corner, Library 3/F, Library 4/F Quiet Zone |
| **AC2 / LI** | Li Dak Sum Yip Yio Chin Academic Building (classrooms, computer rooms) | **3/F = AC2 Canteen** | AC2 Canteen (Semi-Buffet), AC2 Canteen (Counter) |
| **AC3 / LAU** | Lau Ming Wai Academic Building (faculty teaching space) | **7/F = AC3 Bistro** (the AC3 canteen) | AC3 Bistro (Semi-Buffet), AC3 Bistro (Counter), LAU 4/F Study Lounge |

**Departure points** (sidebar): `AC1 (YEUNG 4/F LT)`, `Library (YEUNG 3/F)`,
`AC2 (LI)`, `AC3 (LAU)`.

**Walking model** — real pedestrian routes between building entrances, plus a
lift/stairs allowance inside a building:

| Route | Minutes | Source |
| --- | --- | --- |
| same building, one floor | 1.4 | 1.0 min lift/stairs + 0.4 min per floor |
| AC1/YEUNG ↔ AC2/LI | 3.5 | surveyed ground route |
| AC1/YEUNG ↔ AC3/LAU | 4.5 | surveyed ground route |
| AC2/LI ↔ AC3/LAU | 3.0 | surveyed ground route |
| AC1 4/F LT → AC1 Canteen 5/F | 1.4 | one floor up |
| AC3 1/F → AC3 Bistro 7/F | 3.4 | six floors |

---

## Why this is interesting

The canteen bottlenecks are **sequential, not monolithic**. A field stopwatch
survey on campus shows that the AC1 counter is not "one 10-minute queue":

| Zone (floor) | Order screen | Food pickup | Find a seat | Utensils | Peak total |
| --- | --- | --- | --- | --- | --- |
| AC1 Canteen (AC1 · 5/F) | 2–3 min | 3–5 min | ~2 min | 1 min | **8–11 min** |
| AC2 Canteen, Semi-Buffet (AC2 · 3/F) | — | 1–5 min | 1–3 min | — | 2–8 min |
| AC2 Canteen, Counter (AC2 · 3/F) | — | 5–10 min | 1–3 min | — | **6–13 min** |
| AC3 Bistro, Semi-Buffet (AC3 · 7/F) | — | 1–3 min | 1–5 min | — | 2–8 min |
| AC3 Bistro, Counter (AC3 · 7/F) | — | 4–5 min | 1–5 min | — | 5–10 min |
| Library 3/F (AC1 · 3/F), YEUNG 4/F Study Corner | — | — | 6–12 min | — | 6–12 min |
| LAU 4/F Study Lounge (AC3 · 4/F), Library 4/F Quiet Zone | — | — | 0–2 min | — | 0–2 min |

Two demand shocks dominate the day: the **11:50** and **12:50** class-dismissal
waves. They are non-linear, they interact with the *capacity* of each zone
(a 120-seat corner drowns where a 300-seat canteen absorbs the wave), and that
interaction is exactly what a neural network can capture and a plain linear
model cannot.

---

## Highlights

- **960 aggregate observations** — 3 teaching weeks × 5 weekdays × 8 fifteen-minute snapshots × 8 zones (of 9, one zone rests per survey day).
- **Leakage-free evaluation** — chronological split: week 1–2 train (640), week 3 first half validation (160), week 3 second half test (160). No shuffling, no future data.
- **Four models compared in one table** — historical slot average, plain linear/logistic, and the selected Ridge + MLP pair.
- **Two interlocking targets** — wait time is built from the phase chain, congestion from the independent occupancy/capacity ratio (Low < 60 %, Medium 60–85 %, High > 85 %).
- **Schematic 2-D campus map** — AC1/YEUNG, AC2/LI and AC3/LAU footprints with congestion bubbles sized by predicted wait; hover a bubble for the floor, predicted wait, softmax probabilities and the routing reason.
- **Routing, not just prediction** — `Total(z) = T_walk(u,z) + Ŵ(z, t + T_walk) + P_crowd(z)`; predictions are made for the *arrival* moment.
- **Privacy by construction** — aggregated counts and durations only. No identity, no device, no Wi-Fi MAC data.
- **Runs in seconds on a laptop** — single-row inference under 0.1 ms.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. build the dataset (960 rows -> data/cityuhk_campus_data.csv)
python src/generate_data.py

# 3. train + tune + evaluate (writes models/*.joblib and models/metrics.json)
python src/train_models.py

# 4. launch the dashboard
streamlit run app.py
```

Then open the printed local URL (default <http://localhost:8501>).

Optional end-to-end verification (dataset → metrics → serving path → walking
model → map geometry → dashboard render):

```bash
python tests/smoke_test.py
```

Requirements: Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `joblib`,
`streamlit`, `plotly`. Nothing else, no API keys, no network access at runtime.

---

## Results

Held-out **week-3 second half** (160 rows the models never saw).

| Model | Role | Wait-time MAE (min) ↓ | Congestion macro F1 ↑ | Accuracy | Latency (ms / row) |
| --- | --- | --- | --- | --- | --- |
| Historical Slot Average | baseline | 2.111 | 0.758 | 0.794 | 0.004 |
| Linear Regression + Logistic Regression | baseline | 0.952 | 0.799 | 0.844 | 0.053 |
| **Ridge + MLPClassifier** | **selected** | **0.585** | **0.870** | **0.894** | 0.058 |

Both selected models beat both baselines: Ridge cuts the wait-time error by
**~39 %** versus the plain linear regression, and the MLP lifts macro F1 by
**+0.07 absolute** over the logistic baseline (and +0.11 over the slot average).

**Selected configuration**

| Component | Setting |
| --- | --- |
| Regression | `Ridge(alpha=0.3)`, α grid `0.01 … 300` searched on the validation split |
| Classification | `MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", alpha=1e-05, batch_size=16, lr=0.003, early_stopping=True)` |
| MLP seed | 7 — the initialisation seed is treated as one more hyper-parameter and also chosen on validation, never on test |
| Features | one-hot `zone_id`, `service_type`, `weekday`, 15-min slot, `zone×slot` interaction, `weekday×slot`, `log1p(dismissals)`, `dismissals / capacity` (+ squared), `minutes_to_noon`, `is_lunch_peak` |

The two baselines are deliberately naive: the slot average only remembers
`weekday × time-slot`, and the linear/logistic baseline gets the raw one-hot
block with **no interaction terms and no engineered features** — i.e. the
textbook model that the selected pipeline has to beat.

---

## Repository layout

```
.
├── app.py                     # Streamlit dashboard (2 tabs, 2-D campus map)
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── cityuhk_campus_data.csv    # 960-row aggregate dataset (generated)
├── models/
│   ├── preprocessor.joblib        # fitted ColumnTransformer (shared preprocessing pipeline)
│   ├── ridge_model.joblib         # Pipeline(engineered features -> Ridge)
│   ├── mlp_model.joblib           # Pipeline(compact features -> MLP)
│   └── metrics.json               # comparison table + splits + limitations (read by the dashboard)
├── src/
│   ├── campus_config.py       # buildings/floors, zones, phase bottlenecks, walking model, map coords
│   ├── features.py            # shared feature engineering + sklearn preprocessors
│   ├── generate_data.py       # synthetic field-survey data generator
│   └── train_models.py        # time-ordered split, baselines, tuning, evaluation, persistence
└── tests/
    └── smoke_test.py          # dataset / metrics / serving / walking / map / dashboard assertions
```

---

## Dataset

`data/cityuhk_campus_data.csv` — one row = one aggregate observation of one zone
at one 15-minute snapshot.

| Column | Type | Meaning |
| --- | --- | --- |
| `week` | int | teaching week 1–3 |
| `weekday` | str | `Mon` … `Fri` |
| `time_slot` | str | `09:30, 11:30, 11:45, 12:00, 12:30, 13:00, 13:30, 15:30` |
| `zone_id` | str | one of the nine zones listed in the geography table above |
| `service_type` | str | `Study` / `Semi-Buffet` / `Counter` |
| `zone_capacity` | int | seats / standing capacity of the zone |
| `nearby_class_dismissals` | int | students released by nearby lectures in the last 10 minutes (spikes at 11:50 and 12:50) |
| `wait_time_mins` | float | **label 1** — sum of the stopwatch-measured phases |
| `congestion_level` | str | **label 2** — `Low` (<60 %), `Medium` (60–85 %), `High` (>85 % occupancy vs capacity) |

Building and floor are *configuration*, not features: the model does not need
them, the dashboard does (map position, card labels, hover text).

The two labels are generated independently from two different mechanisms
(phase chain vs occupancy ratio), so the classifier is not just reading the
regression target back out. Class mix: 562 Low / 201 High / 197 Medium.

**Privacy:** aggregate counts and durations only. Every row is a zone-time
aggregate — no student ID, no student-level record, no device or Wi-Fi
identifier, nothing that could re-identify an individual.

Reproducibility: fixed numpy seed (`20241122`) inside `src/generate_data.py`,
so regenerating the CSV gives byte-identical numbers.

---

## The routing formula

For every candidate zone `z` that satisfies the requested need:

```
Total(z) = T_walk(origin, z) + Ŵ(z, t + T_walk(origin, z)) + P_crowd(z)
```

- `T_walk` — from the departure point to the zone: same building = 1.0 min + 0.4 min per floor; different building = the surveyed entrance-to-entrance distance (3.0–4.5 min).
- `Ŵ` — Ridge wait-time prediction evaluated at the **arrival** minute, not the departure minute.
- `P_crowd` — 0 min (Low), 1.5 min (Medium), 4 min (High), because a crowded zone costs you beyond the measured queue.
- The MLP softmax probabilities for the arrival moment drive the congestion badge, the confidence bars and the penalty.

The dashboard shows:

1. **Go Here Instead** — top-3 cards with floor, walking time, predicted wait (±MAE) and total time.
2. **Campus congestion map** — schematic AC1/YEUNG, AC2/LI and AC3/LAU footprints; bubble size = predicted wait, colour = congestion level; hover for the floor, the wait, the softmax split and the routing reason.
3. **Congestion board + MLP softmax confidence** — every zone ranked by total time.
4. **Per-phase bottleneck table** — where the waiting minutes actually go.
5. **2-hour look-ahead** — walk + predicted wait for every candidate zone from your departure time.

---

## Course alignment (GE2340)

| Concept | Where it lives |
| --- | --- |
| Leakage-free time-ordered splitting | `time_ordered_split` in `src/train_models.py` |
| Baseline thinking | historical slot average; plain linear / logistic regression |
| L2 regularisation + hyper-parameter search | Ridge, α grid on the validation split |
| Neural networks | one ReLU hidden layer (32 units), Adam, L2 weight decay |
| Multi-class softmax + cross-entropy | MLP output layer; probabilities surfaced in the UI |
| Feature engineering | interaction one-hots, log/density transforms, capacity-normalised load |
| Regression + classification metrics | Wait-time MAE, macro F1, accuracy, per-row latency |
| Deployment | joblib pipelines, one shared serving path, latency measured for every model |

---

## Known limitations

- **Synthetic data.** The dataset reproduces observed stopwatch behaviour and the
  known dismissal waves, but it is a model of the campus, not a live feed.
- **Sudden rain.** A thunderstorm pushes the whole campus indoors at once; real
  waits can jump outside the training range and the models will under-predict.
- **Unregistered events** (society fairs, career talks, exchange briefings)
  inject demand the model has never seen.
- **Only three teaching weeks** are covered — semester start and exam periods
  behave differently (the library becomes the bottleneck while canteens empty).
- **The map is schematic.** Bubble positions are a hand-placed 2-D layout for
  readability, not geo-referenced coordinates; walking times come from the
  matrix above, never from the drawing.
- **Phase breakdowns are estimates.** The per-phase minutes in the dashboard are
  the predicted total split by the measured stopwatch structure, not five
  independently predicted quantities.
- **The ±MAE band is an average**, not a per-prediction guarantee.

---

## License / attribution

Course project for GE2340 (City University of Hong Kong). Code and synthetic
dataset are released for academic demonstration purposes.
