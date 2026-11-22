"""Bilingual / trilingual UI strings for the CityUHK Smart Campus demo.

Three complete locales -- Simplified Chinese, Traditional Chinese and English.
Every visible string in the dashboard comes from here, so switching language
never leaves a stray English label in a Chinese view (or the reverse).

Usage::

    from src.i18n import tr, zone_name, dish_name, category_name
    tr("zh_Hant", "res.title")            # -> "去邊度好"
    zone_name("en", "AC1 Canteen")        # -> "AC1 Canteen"
"""

from __future__ import annotations

from src.campus_config import (
    CAT_ANY,
    CAT_ASIAN,
    CAT_NOODLE,
    CAT_RICE,
    CAT_WESTERN,
    HALLS,
    ZONES,
)

LANGS: tuple[str, str, str] = ("zh_Hans", "zh_Hant", "en")
LANG_LABELS: dict[str, str] = {
    "zh_Hans": "简体中文",
    "zh_Hant": "繁體中文",
    "en": "English",
}
DEFAULT_LANG: str = "zh_Hans"


# --------------------------------------------------------------------------- #
# UI strings
# --------------------------------------------------------------------------- #
UI: dict[str, dict[str, str]] = {
    "en": {
        # shell
        "app.title": "Smart Campus Crowd-Balancing & Space Routing",
        "app.tagline": "Find the shortest queue, not the nearest one.",
        "tab.router": "Live Campus Router",
        "tab.model": "AI Model & Data",
        "lang.label": "Language",
        # inputs
        "in.location": "I am at",
        "in.floor": "Floor",
        "in.goal": "I want",
        "in.category": "Food category",
        "in.day": "Day",
        "in.time": "Leaving at",
        "in.quick": "Quick pick",
        "q.now": "Now",
        "q.p15": "+15 min",
        "q.p30": "+30 min",
        "goal.food": "Something to eat",
        "goal.study": "A study seat",
        # buildings
        "bld.YEUNG": "AC1 (YEUNG, 1–8/F)",
        "bld.LI": "AC2 (LI, B2–9/F)",
        "bld.LAU": "AC3 (LAU, G/F–20/F)",
        "bld.HALLS": "Halls (Kowloon Tong)",
        "bld.short.YEUNG": "AC1 · YEUNG",
        "bld.short.LI": "AC2 · LI",
        "bld.short.LAU": "AC3 · LAU",
        "bld.short.HALLS": "Halls",
        # food categories
        "cat.any": "Any (fastest)",
        "cat.rice": "Rice & quick meals",
        "cat.noodle": "Noodles & soup",
        "cat.western": "Western & fast food",
        "cat.asian": "Asian & curry",
        "cat.any.hint": "No constraint — just the shortest door-to-door time.",
        "cat.rice.hint": "Two-dish rice, siu mei, weighed buffet.",
        "cat.noodle.hint": "Rice noodles, noodle soups, mala tang, dumplings.",
        "cat.western.hint": "Pasta, baked rice, pizza, fried chicken.",
        "cat.asian.hint": "South-East Asian plates and curry rice.",
        # congestion
        "lvl.Low": "Quiet",
        "lvl.Medium": "Busy",
        "lvl.High": "Packed",
        "st.open": "Open",
        "st.closed": "Closed",
        # results
        "res.title": "Go here instead",
        "res.sub": "Top 3 · leaving {time} · {day}",
        "res.from": "from {place}",
        "res.best": "Best pick",
        "res.rank": "#{n}",
        "res.walk": "walk",
        "res.arrive": "arrive",
        "res.queue": "queue",
        "res.total": "total",
        "res.min": "min",
        "res.saves": "saves {n} min vs the busiest canteen",
        "res.busiest": "currently the busiest canteen",
        "res.route": "Route",
        "res.phases": "Queue breakdown",
        "res.study": "seat search",
        "res.none_open": "Every canteen is closed at {time} on {day}.",
        "res.none_open.hint": "Canteens serve 07:30–21:00; AC3 is closed on Sundays.",
        "res.jump_lunch": "Jump to lunch peak",
        "res.study_only": "Study seats are open all day — no closing time.",
        # campus board
        "campus.title": "Campus status right now",
        "campus.sub": "Bubble size = expected queue · colour = how busy",
        # phases
        "ph.order": "Order screen",
        "ph.pickup": "Food pickup",
        "ph.seat": "Find a seat",
        "ph.utensil": "Utensils",
        "ph.serve": "Semi-buffet serving",
        "ph.seatsearch": "Seat search",
        # days
        "day.Mon": "Mon", "day.Tue": "Tue", "day.Wed": "Wed", "day.Thu": "Thu",
        "day.Fri": "Fri", "day.Sat": "Sat", "day.Sun": "Sun",
        # route hints
        "route.lift": "lift to {floor}",
        "route.corridor": "corridor to {place}",
        "route.start": "start at {place}",
        # model tab
        "m.title": "AI Model & Data",
        "m.intro": (
            "Two deployed models: **Ridge** (L2 regression) predicts the queue in minutes, "
            "**MLPClassifier** (one 32-unit ReLU layer, Adam, L2 weight decay) predicts "
            "Quiet / Busy / Packed. Both are tuned on a validation split and scored once "
            "on a held-out test split."
        ),
        "m.split": "Chronological split · week 1–2 train · week 3 first half validation · week 3 second half test",
        "m.col.model": "Model",
        "m.col.role": "Role",
        "m.col.mae": "Wait-time MAE (min) ↓",
        "m.col.f1": "Macro F1 ↑",
        "m.col.acc": "Accuracy",
        "m.col.lat": "Latency (ms / row)",
        "m.model.slot": "Slot average",
        "m.model.linear": "Linear / logistic",
        "m.model.selected": "Ridge + MLP",
        "m.col.mae.short": "MAE (min) ↓",
        "m.col.f1.short": "F1 ↑",
        "m.role.baseline": "baseline",
        "m.role.selected": "selected",
        "m.chart.mae": "Wait-time MAE (minutes, lower is better)",
        "m.chart.f1": "Congestion macro F1 (higher is better)",
        "m.verdict": "Selected models beat both baselines",
        "m.verdict.no": "A baseline is still ahead — see the metrics above.",
        "m.softmax": "Softmax probability distribution",
        "m.softmax.sub": "How confident the neural network is about each zone at your arrival time.",
        "m.data": "Data source & coverage",
        "m.data.src": "Simulated timetable & field-calibrated stopwatch data",
        "m.data.cov": "Coverage",
        "m.data.cov.value": "3 teaching weeks × Mon–Sun × 07:30–22:00 · 15-minute snapshots",
        "m.data.rows": "Rows in the published dataset",
        "m.data.note": (
            "This is **not** a live sensor feed. The demand curve and the class-change waves "
            "are simulated from the public timetable shape and calibrated against stopwatch "
            "observations taken on campus; closing hours and weekend behaviour are explicit "
            "rules. No student, device or Wi-Fi data is used anywhere."
        ),
        "m.limit": "Known limitations",
        "m.course": "Course alignment",
        "m.more": "Accuracy & latency",
        "m.course.head": "GE2340 concept",
        "m.course.body": "Where it lives",
    },
    "zh_Hans": {
        "app.title": "校园人流平衡与路线推荐",
        "app.tagline": "别去最近的，去最不用排队的。",
        "tab.router": "校园实时导航",
        "tab.model": "AI 模型与数据",
        "lang.label": "语言",
        "in.location": "我现在在",
        "in.floor": "楼层",
        "in.goal": "我想要",
        "in.category": "食物类别",
        "in.day": "星期",
        "in.time": "出发时间",
        "in.quick": "快捷选择",
        "q.now": "现在",
        "q.p15": "+15 分钟",
        "q.p30": "+30 分钟",
        "goal.food": "吃饭",
        "goal.study": "找自习座位",
        "bld.YEUNG": "AC1（杨建文楼，1–8 楼）",
        "bld.LI": "AC2（李达三叶耀珍楼，B2–9 楼）",
        "bld.LAU": "AC3（刘鸣炜楼，G/F–20 楼）",
        "bld.HALLS": "宿舍（九龙塘）",
        "bld.short.YEUNG": "AC1 · 杨建文楼",
        "bld.short.LI": "AC2 · 李达三楼",
        "bld.short.LAU": "AC3 · 刘鸣炜楼",
        "bld.short.HALLS": "宿舍",
        "cat.any": "不限（最快）",
        "cat.rice": "米饭快食（两餸饭 / 烧味 / 称重自助）",
        "cat.noodle": "面食汤粉（米线 / 面食 / 麻辣烫）",
        "cat.western": "西式快餐（意粉焗饭 / Pizza / 炸鸡）",
        "cat.asian": "东南亚与咖喱（东南亚菜 / 咖喱饭）",
        "cat.any.hint": "不限制菜式，只按门到门最短时间推荐。",
        "cat.rice.hint": "两餸饭、烧味、称重自助。",
        "cat.noodle.hint": "米线、面食汤粉、麻辣烫、饺子。",
        "cat.western.hint": "意粉、焗饭、Pizza、炸鸡。",
        "cat.asian.hint": "东南亚菜与咖喱饭。",
        "lvl.Low": "空闲",
        "lvl.Medium": "人多",
        "lvl.High": "爆满",
        "st.open": "营业中",
        "st.closed": "已打烊",
        "res.title": "改去这里",
        "res.sub": "前三推荐 · {day} {time} 出发",
        "res.from": "从{place}出发",
        "res.best": "最佳选择",
        "res.rank": "第 {n} 选",
        "res.walk": "步行",
        "res.arrive": "到达",
        "res.queue": "排队",
        "res.total": "合计",
        "res.min": "分钟",
        "res.saves": "比最挤的食堂省 {n} 分钟",
        "res.busiest": "目前最挤的就是这里",
        "res.route": "路线",
        "res.phases": "排队拆解",
        "res.study": "找座时间",
        "res.none_open": "{day} {time} 所有食堂都已打烊。",
        "res.none_open.hint": "食堂营业时间 07:30–21:00；AC3 星期日休息。",
        "res.jump_lunch": "跳到午餐高峰",
        "res.study_only": "自习区全天开放，没有打烊时间。",
        "campus.title": "当前校园状态",
        "campus.sub": "气泡越大等待越久 · 颜色代表拥挤程度",
        "ph.order": "点餐屏",
        "ph.pickup": "取餐",
        "ph.seat": "找座",
        "ph.utensil": "拿餐具",
        "ph.serve": "半自助取餐",
        "ph.seatsearch": "找座",
        "day.Mon": "周一", "day.Tue": "周二", "day.Wed": "周三", "day.Thu": "周四",
        "day.Fri": "周五", "day.Sat": "周六", "day.Sun": "周日",
        "route.lift": "乘电梯至 {floor}",
        "route.corridor": "经连廊至 {place}",
        "route.start": "从{place}出发",
        "m.title": "AI 模型与数据",
        "m.intro": (
            "线上部署两个模型：**Ridge**（L2 回归）预测排队分钟数，"
            "**MLPClassifier**（单层 32 神经元 ReLU，Adam，L2 权重衰减）预测"
            "空闲 / 人多 / 爆满。两者均在验证集上调参，只在测试集上评分一次。"
        ),
        "m.split": "按时间切分 · 第 1–2 周训练 · 第 3 周前半验证 · 第 3 周后半测试",
        "m.col.model": "模型",
        "m.col.role": "角色",
        "m.col.mae": "等待时长 MAE（分钟）↓",
        "m.col.f1": "宏平均 F1 ↑",
        "m.col.acc": "准确率",
        "m.col.lat": "单行推理耗时（毫秒）",
        "m.model.slot": "时段均值",
        "m.model.linear": "线性 / 逻辑回归",
        "m.model.selected": "Ridge + MLP",
        "m.col.mae.short": "MAE（分钟）↓",
        "m.col.f1.short": "F1 ↑",
        "m.role.baseline": "基线",
        "m.role.selected": "最终模型",
        "m.chart.mae": "等待时长 MAE（分钟，越低越好）",
        "m.chart.f1": "拥挤度宏平均 F1（越高越好）",
        "m.verdict": "最终模型优于两个基线",
        "m.verdict.no": "仍有基线更优，请见上方指标。",
        "m.softmax": "Softmax 概率分布",
        "m.softmax.sub": "神经网络对到达时刻各区域拥挤等级的把握程度。",
        "m.data": "数据来源与覆盖范围",
        "m.data.src": "模拟课表 + 实地秒表标定数据",
        "m.data.cov": "覆盖范围",
        "m.data.cov.value": "3 个教学周 × 周一至周日 × 07:30–22:00 · 每 15 分钟一个快照",
        "m.data.rows": "公开数据集行数",
        "m.data.note": (
            "这**不是**实时传感器数据。需求曲线与下课潮由公开课表形态模拟，"
            "并用校园实地秒表观测标定；营业时间与周末规律为显式规则。"
            "全程不使用任何学生个人、设备或 Wi-Fi 数据。"
        ),
        "m.limit": "已知局限",
        "m.course": "课程知识点对应",
        "m.more": "准确率与耗时",
        "m.course.head": "GE2340 知识点",
        "m.course.body": "在项目中的位置",
    },
    "zh_Hant": {
        "app.title": "校園人流平衡與路線推薦",
        "app.tagline": "別去最近的，去最不用排隊的。",
        "tab.router": "校園即時導航",
        "tab.model": "AI 模型與數據",
        "lang.label": "語言",
        "in.location": "我現在在",
        "in.floor": "樓層",
        "in.goal": "我想要",
        "in.category": "食物類別",
        "in.day": "星期",
        "in.time": "出發時間",
        "in.quick": "快速選擇",
        "q.now": "現在",
        "q.p15": "+15 分鐘",
        "q.p30": "+30 分鐘",
        "goal.food": "吃飯",
        "goal.study": "找自修座位",
        "bld.YEUNG": "AC1（楊建文樓，1–8 樓）",
        "bld.LI": "AC2（李達三葉耀珍樓，B2–9 樓）",
        "bld.LAU": "AC3（劉鳴煒樓，G/F–20 樓）",
        "bld.HALLS": "宿舍（九龍塘）",
        "bld.short.YEUNG": "AC1 · 楊建文樓",
        "bld.short.LI": "AC2 · 李達三樓",
        "bld.short.LAU": "AC3 · 劉鳴煒樓",
        "bld.short.HALLS": "宿舍",
        "cat.any": "不限（最快）",
        "cat.rice": "米飯快食（兩餸飯 / 燒味 / 秤重自助）",
        "cat.noodle": "麵食湯粉（米線 / 麵食 / 麻辣燙）",
        "cat.western": "西式快餐（意粉焗飯 / Pizza / 炸雞）",
        "cat.asian": "東南亞與咖喱（東南亞菜 / 咖喱飯）",
        "cat.any.hint": "不限制菜式，只按門到門最短時間推薦。",
        "cat.rice.hint": "兩餸飯、燒味、秤重自助。",
        "cat.noodle.hint": "米線、麵食湯粉、麻辣燙、餃子。",
        "cat.western.hint": "意粉、焗飯、Pizza、炸雞。",
        "cat.asian.hint": "東南亞菜與咖喱飯。",
        "lvl.Low": "空閒",
        "lvl.Medium": "人多",
        "lvl.High": "爆滿",
        "st.open": "營業中",
        "st.closed": "已打烊",
        "res.title": "改去這裡",
        "res.sub": "前三推薦 · {day} {time} 出發",
        "res.from": "從{place}出發",
        "res.best": "最佳選擇",
        "res.rank": "第 {n} 選",
        "res.walk": "步行",
        "res.arrive": "到達",
        "res.queue": "排隊",
        "res.total": "合計",
        "res.min": "分鐘",
        "res.saves": "比最擠的食堂省 {n} 分鐘",
        "res.busiest": "目前最擠的就是這裡",
        "res.route": "路線",
        "res.phases": "排隊拆解",
        "res.study": "找座時間",
        "res.none_open": "{day} {time} 所有食堂都已打烊。",
        "res.none_open.hint": "食堂營業時間 07:30–21:00；AC3 星期日休息。",
        "res.jump_lunch": "跳到午餐高峰",
        "res.study_only": "自修區全天開放，沒有打烊時間。",
        "campus.title": "目前校園狀態",
        "campus.sub": "氣泡越大等候越久 · 顏色代表擁擠程度",
        "ph.order": "點餐螢幕",
        "ph.pickup": "取餐",
        "ph.seat": "找位",
        "ph.utensil": "拿餐具",
        "ph.serve": "半自助取餐",
        "ph.seatsearch": "找位",
        "day.Mon": "週一", "day.Tue": "週二", "day.Wed": "週三", "day.Thu": "週四",
        "day.Fri": "週五", "day.Sat": "週六", "day.Sun": "週日",
        "route.lift": "乘電梯至 {floor}",
        "route.corridor": "經連廊至 {place}",
        "route.start": "從{place}出發",
        "m.title": "AI 模型與數據",
        "m.intro": (
            "線上部署兩個模型：**Ridge**（L2 回歸）預測排隊分鐘數，"
            "**MLPClassifier**（單層 32 神經元 ReLU，Adam，L2 權重衰減）預測"
            "空閒 / 人多 / 爆滿。兩者均在驗證集上調參，只在測試集上評分一次。"
        ),
        "m.split": "按時間切分 · 第 1–2 週訓練 · 第 3 週前半驗證 · 第 3 週後半測試",
        "m.col.model": "模型",
        "m.col.role": "角色",
        "m.col.mae": "等候時間 MAE（分鐘）↓",
        "m.col.f1": "巨集平均 F1 ↑",
        "m.col.acc": "準確率",
        "m.col.lat": "單列推理耗時（毫秒）",
        "m.model.slot": "時段均值",
        "m.model.linear": "線性 / 邏輯回歸",
        "m.model.selected": "Ridge + MLP",
        "m.col.mae.short": "MAE（分鐘）↓",
        "m.col.f1.short": "F1 ↑",
        "m.role.baseline": "基線",
        "m.role.selected": "最終模型",
        "m.chart.mae": "等候時間 MAE（分鐘，越低越好）",
        "m.chart.f1": "擁擠度巨集平均 F1（越高越好）",
        "m.verdict": "最終模型優於兩個基線",
        "m.verdict.no": "仍有基線更優，請見上方指標。",
        "m.softmax": "Softmax 機率分佈",
        "m.softmax.sub": "神經網路對到達時刻各區域擁擠等級的把握程度。",
        "m.data": "數據來源與覆蓋範圍",
        "m.data.src": "模擬課表 + 實地秒錶標定數據",
        "m.data.cov": "覆蓋範圍",
        "m.data.cov.value": "3 個教學週 × 週一至週日 × 07:30–22:00 · 每 15 分鐘一個快照",
        "m.data.rows": "公開數據集列數",
        "m.data.note": (
            "這**不是**即時感測器數據。需求曲線與下課潮由公開課表形態模擬，"
            "並用校園實地秒錶觀測標定；營業時間與週末規律為顯式規則。"
            "全程不使用任何學生個人、裝置或 Wi-Fi 數據。"
        ),
        "m.limit": "已知局限",
        "m.course": "課程知識點對應",
        "m.more": "準確率與耗時",
        "m.course.head": "GE2340 知識點",
        "m.course.body": "在專案中的位置",
    },
}


# --------------------------------------------------------------------------- #
# Zone / dish / category / phase names
# --------------------------------------------------------------------------- #
ZONE_NAMES: dict[str, dict[str, str]] = {
    "AC1 Canteen": {"en": "AC1 Canteen · 5/F", "zh_Hans": "AC1 食堂 · 5 楼", "zh_Hant": "AC1 食堂 · 5 樓"},
    "AC2 Canteen (Semi-Buffet)": {"en": "AC2 Semi-Buffet · 3/F", "zh_Hans": "AC2 食堂（半自助）· 3 楼", "zh_Hant": "AC2 食堂（半自助）· 3 樓"},
    "AC2 Canteen (Counter)": {"en": "AC2 Counter · 3/F", "zh_Hans": "AC2 食堂（档口）· 3 楼", "zh_Hant": "AC2 食堂（檔口）· 3 樓"},
    "AC3 Canteen (Semi-Buffet)": {"en": "AC3 Bistro Semi-Buffet · 7/F", "zh_Hans": "AC3 Bistro（半自助）· 7 楼", "zh_Hant": "AC3 Bistro（半自助）· 7 樓"},
    "AC3 Canteen (Counter)": {"en": "AC3 Bistro Counter · 7/F", "zh_Hans": "AC3 Bistro（档口）· 7 楼", "zh_Hant": "AC3 Bistro（檔口）· 7 樓"},
    "Library 3/F": {"en": "Library · 3/F", "zh_Hans": "图书馆 · 3 楼", "zh_Hant": "圖書館 · 3 樓"},
    "YEUNG 4/F Study Corner": {"en": "Yeung Study Corner · 4/F", "zh_Hans": "杨建文楼自习角 · 4 楼", "zh_Hant": "楊建文樓自修角 · 4 樓"},
    "LAU 4/F Study Lounge": {"en": "Lau Study Lounge · 4/F", "zh_Hans": "刘鸣炜楼自修室 · 4 楼", "zh_Hant": "劉鳴煒樓自修室 · 4 樓"},
    "Library 4/F Quiet Zone": {"en": "Library Quiet Zone · 4/F", "zh_Hans": "图书馆静音区 · 4 楼", "zh_Hant": "圖書館靜音區 · 4 樓"},
}

# Short captions used as bubble labels on the small campus map.
ZONE_SHORT: dict[str, dict[str, str]] = {
    "AC1 Canteen": {"en": "AC1 Canteen", "zh_Hans": "AC1 食堂", "zh_Hant": "AC1 食堂"},
    "AC2 Canteen (Semi-Buffet)": {"en": "AC2 Semi", "zh_Hans": "AC2 半自助", "zh_Hant": "AC2 半自助"},
    "AC2 Canteen (Counter)": {"en": "AC2 Counter", "zh_Hans": "AC2 档口", "zh_Hant": "AC2 檔口"},
    "AC3 Canteen (Semi-Buffet)": {"en": "AC3 Semi", "zh_Hans": "AC3 半自助", "zh_Hant": "AC3 半自助"},
    "AC3 Canteen (Counter)": {"en": "AC3 Counter", "zh_Hans": "AC3 档口", "zh_Hant": "AC3 檔口"},
    "Library 3/F": {"en": "Library 3/F", "zh_Hans": "图书馆 3 楼", "zh_Hant": "圖書館 3 樓"},
    "YEUNG 4/F Study Corner": {"en": "Yeung 4/F", "zh_Hans": "杨楼 4 楼", "zh_Hant": "楊樓 4 樓"},
    "LAU 4/F Study Lounge": {"en": "Lau 4/F", "zh_Hans": "刘楼 4 楼", "zh_Hant": "劉樓 4 樓"},
    "Library 4/F Quiet Zone": {"en": "Library 4/F", "zh_Hans": "图书馆 4 楼", "zh_Hant": "圖書館 4 樓"},
}

DISH_NAMES: dict[str, dict[str, str]] = {
    "bbq": {"en": "Siu Mei (BBQ)", "zh_Hans": "烧味", "zh_Hant": "燒味"},
    "two_dish": {"en": "Two-dish rice", "zh_Hans": "两餸饭", "zh_Hant": "兩餸飯"},
    "cart_noodle": {"en": "Cart noodles", "zh_Hans": "车仔米线", "zh_Hant": "車仔米線"},
    "ramen_beef": {"en": "Ramen & beef rice", "zh_Hans": "拉面 & 牛肉饭", "zh_Hant": "拉麵 & 牛肉飯"},
    "pizza": {"en": "Pizza", "zh_Hans": "Pizza", "zh_Hant": "Pizza"},
    "fried_chicken": {"en": "Fried chicken", "zh_Hans": "炸鸡", "zh_Hant": "炸雞"},
    "sea_asian": {"en": "Southeast Asian", "zh_Hans": "东南亚菜", "zh_Hant": "東南亞菜"},
    "rice_noodle": {"en": "Rice noodles", "zh_Hans": "米线", "zh_Hant": "米線"},
    "mala_tang": {"en": "Mala tang", "zh_Hans": "麻辣烫", "zh_Hant": "麻辣燙"},
    "dumpling": {"en": "Dumplings", "zh_Hans": "饺子", "zh_Hant": "餃子"},
    "pasta_bake": {"en": "Pasta & baked rice", "zh_Hans": "意粉 & 焗饭", "zh_Hant": "意粉 & 焗飯"},
    "semi_buffet": {"en": "Semi-buffet (weighed + rice)", "zh_Hans": "半自助（称重 + 米饭）", "zh_Hant": "半自助（秤重 + 米飯）"},
    "curry_rice": {"en": "Curry rice", "zh_Hans": "咖喱饭", "zh_Hant": "咖喱飯"},
    "signature_noodle": {"en": "Signature noodles", "zh_Hans": "特色面食", "zh_Hant": "特色麵食"},
}

CATEGORY_NAMES: dict[str, dict[str, str]] = {
    CAT_ANY: {"en": "Any (fastest)", "zh_Hans": "不限（最快）", "zh_Hant": "不限（最快）"},
    CAT_RICE: {"en": "Rice & quick meals", "zh_Hans": "米饭快食", "zh_Hant": "米飯快食"},
    CAT_NOODLE: {"en": "Noodles & soup", "zh_Hans": "面食汤粉", "zh_Hant": "麵食湯粉"},
    CAT_WESTERN: {"en": "Western & fast food", "zh_Hans": "西式快餐", "zh_Hant": "西式快餐"},
    CAT_ASIAN: {"en": "Asian & curry", "zh_Hans": "东南亚与咖喱", "zh_Hant": "東南亞與咖喱"},
}

CATEGORY_HINTS: dict[str, dict[str, str]] = {
    CAT_ANY: {"en": "No constraint — just the shortest door-to-door time.",
              "zh_Hans": "不限制菜式，只按门到门最短时间推荐。",
              "zh_Hant": "不限制菜式，只按門到門最短時間推薦。"},
    CAT_RICE: {"en": "Two-dish rice, siu mei, weighed buffet.",
               "zh_Hans": "两餸饭、烧味、称重自助。",
               "zh_Hant": "兩餸飯、燒味、秤重自助。"},
    CAT_NOODLE: {"en": "Rice noodles, noodle soups, mala tang, dumplings.",
                 "zh_Hans": "米线、面食汤粉、麻辣烫、饺子。",
                 "zh_Hant": "米線、麵食湯粉、麻辣燙、餃子。"},
    CAT_WESTERN: {"en": "Pasta, baked rice, pizza, fried chicken.",
                  "zh_Hans": "意粉、焗饭、Pizza、炸鸡。",
                  "zh_Hant": "意粉、焗飯、Pizza、炸雞。"},
    CAT_ASIAN: {"en": "South-East Asian plates and curry rice.",
                "zh_Hans": "东南亚菜与咖喱饭。",
                "zh_Hant": "東南亞菜與咖喱飯。"},
}

PHASE_NAMES: dict[str, dict[str, str]] = {
    "Order screen queue": {"en": "Order screen", "zh_Hans": "点餐屏", "zh_Hant": "點餐螢幕"},
    "Food pickup": {"en": "Food pickup", "zh_Hans": "取餐", "zh_Hant": "取餐"},
    "Find a seat": {"en": "Find a seat", "zh_Hans": "找座", "zh_Hant": "找位"},
    "Utensils & condiments": {"en": "Utensils", "zh_Hans": "拿餐具", "zh_Hant": "拿餐具"},
    "Semi-buffet serving": {"en": "Semi-buffet serving", "zh_Hans": "半自助取餐", "zh_Hant": "半自助取餐"},
    "Counter food pickup": {"en": "Counter pickup", "zh_Hans": "档口取餐", "zh_Hant": "檔口取餐"},
    "Seat search & settle in": {"en": "Seat search", "zh_Hans": "找座", "zh_Hant": "找位"},
}

COURSE_TABLE: dict[str, dict[str, str]] = {
    "en": {
        "Time-ordered, leakage-free splitting": "`time_ordered_split` — week 1–2 train, week 3 split in half",
        "Baseline thinking": "Historical slot average · plain LinearRegression / LogisticRegression",
        "L2 regularisation": "Ridge, α grid searched on the validation split",
        "Neural networks": "`MLPClassifier((32,), relu, adam, alpha=weight decay)`",
        "Multi-class softmax + cross-entropy": "MLP output layer; probabilities shown above",
        "Feature engineering": "zone × slot interaction, log/density transforms, load = dismissals / capacity",
        "Evaluation metrics": "Wait-time MAE · macro F1 · accuracy · latency per row",
        "Graph algorithms": "Dijkstra over the corridor network drives every walking time",
    },
    "zh_Hans": {
        "时间序切分、无泄露": "`time_ordered_split` —— 第 1–2 周训练，第 3 周对半切开",
        "基线思维": "历史时段均值 · 普通 LinearRegression / LogisticRegression",
        "L2 正则化": "Ridge，α 网格在验证集上搜索",
        "神经网络": "`MLPClassifier((32,), relu, adam, alpha=权重衰减)`",
        "多分类 Softmax + 交叉熵": "MLP 输出层，概率见上方",
        "特征工程": "zone × 时段交互、对数与密度变换、负载 = 下课人数 / 容量",
        "评估指标": "等待 MAE · 宏平均 F1 · 准确率 · 单行耗时",
        "图算法": "连廊网络上的 Dijkstra 算法决定每一次步行时间",
    },
    "zh_Hant": {
        "時間序切分、無洩漏": "`time_ordered_split` —— 第 1–2 週訓練，第 3 週對半切開",
        "基線思維": "歷史時段均值 · 普通 LinearRegression / LogisticRegression",
        "L2 正則化": "Ridge，α 網格在驗證集上搜尋",
        "神經網路": "`MLPClassifier((32,), relu, adam, alpha=權重衰減)`",
        "多分類 Softmax + 交叉熵": "MLP 輸出層，機率見上方",
        "特徵工程": "zone × 時段交互、對數與密度變換、負載 = 下課人數 / 容量",
        "評估指標": "等候 MAE · 巨集平均 F1 · 準確率 · 單列耗時",
        "圖演算法": "連廊網路上的 Dijkstra 演算法決定每一次步行時間",
    },
}


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def tr(lang: str, key: str, **kwargs: object) -> str:
    """Translate a UI key, falling back to English if a locale misses it."""
    table = UI.get(lang) or UI["en"]
    template = table.get(key) or UI["en"].get(key) or key
    return template.format(**kwargs) if kwargs else template


def zone_name(lang: str, zone_id: str) -> str:
    entry = ZONE_NAMES.get(zone_id, {})
    return entry.get(lang) or entry.get("en") or zone_id


def zone_short(lang: str, zone_id: str) -> str:
    entry = ZONE_SHORT.get(zone_id, {})
    return entry.get(lang) or entry.get("en") or zone_id


def dish_name(lang: str, dish_id: str) -> str:
    entry = DISH_NAMES.get(dish_id, {})
    return entry.get(lang) or entry.get("en") or dish_id


def category_name(lang: str, category_id: str) -> str:
    entry = CATEGORY_NAMES.get(category_id, {})
    return entry.get(lang) or entry.get("en") or category_id


def category_hint(lang: str, category_id: str) -> str:
    entry = CATEGORY_HINTS.get(category_id, {})
    return entry.get(lang) or entry.get("en") or ""


def phase_name(lang: str, label: str) -> str:
    entry = PHASE_NAMES.get(label, {})
    return entry.get(lang) or entry.get("en") or label


def day_name(lang: str, weekday: str) -> str:
    return tr(lang, f"day.{weekday}")


def building_label(lang: str, building: str) -> str:
    return tr(lang, f"bld.{building}")


def building_short(lang: str, building: str) -> str:
    return tr(lang, f"bld.short.{building}")


def level_name(lang: str, level: str) -> str:
    return tr(lang, f"lvl.{level}")


def building_axis(lang: str) -> tuple[str, ...]:
    """The three towers, west to east, for the maps."""
    return ("YEUNG", "LI", "LAU") if lang else ("YEUNG", "LI", "LAU")


__all__ = [
    "LANGS",
    "LANG_LABELS",
    "DEFAULT_LANG",
    "UI",
    "ZONE_NAMES",
    "DISH_NAMES",
    "CATEGORY_NAMES",
    "COURSE_TABLE",
    "tr",
    "zone_name",
    "dish_name",
    "category_name",
    "category_hint",
    "phase_name",
    "day_name",
    "building_label",
    "building_short",
    "level_name",
    "building_axis",
    "CAT_ANY",
    "CAT_RICE",
    "CAT_NOODLE",
    "CAT_WESTERN",
    "CAT_ASIAN",
    "HALLS",
    "ZONES",
]
