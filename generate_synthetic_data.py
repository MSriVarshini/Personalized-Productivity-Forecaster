
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_DAYS = 150
START_DATE = datetime(2026, 1, 1)


exam_week_starts = sorted(np.random.choice(range(10, N_DAYS - 10), size=3, replace=False))
exam_days = set()
for s in exam_week_starts:
    exam_days.update(range(s, s + 7))

rows = []
prev_focus_hours = 3.0  

for day_idx in range(N_DAYS):
    date = START_DATE + timedelta(days=day_idx)
    weekday = date.weekday()  # 0=Mon .. 6=Sun
    is_weekend = weekday >= 5
    is_exam_week = day_idx in exam_days

    
    base_sleep = 7.0
    if is_exam_week:
        base_sleep -= 1.0
    if is_weekend:
        base_sleep += 0.5
    sleep_hours = np.clip(np.random.normal(base_sleep, 0.9), 3.5, 10.0)

    
    base_social = 2.5
    if is_weekend:
        base_social += 1.2
    if is_exam_week:
        base_social -= 0.8
    social_media_hours = np.clip(np.random.normal(base_social, 0.7), 0.2, 7.0)

    
    base_study_app = 2.0
    if is_exam_week:
        base_study_app += 2.0
    if is_weekend:
        base_study_app -= 0.5
    study_app_hours = np.clip(np.random.normal(base_study_app, 0.8), 0.0, 8.0)

    
    if is_weekend:
        class_hours = np.random.choice([0, 0, 0, 1], p=[0.7, 0.15, 0.1, 0.05])
    else:
        class_hours = np.clip(np.random.normal(4.5, 1.2), 0, 8)

    
    energy = np.clip(np.random.normal(3.2 + 0.3 * (sleep_hours - 7), 0.8), 1, 5)

    
    focus = (
        0.55 * study_app_hours
        + 0.35 * (sleep_hours - 6)
        - 0.30 * social_media_hours
        + (1.5 if is_exam_week else 0)
        - (0.8 if is_weekend else 0)
        + 0.25 * prev_focus_hours
        + 0.3 * (energy - 3)
        + np.random.normal(0, 0.6)
    )
    focus_hours = float(np.clip(focus, 0, 10))

    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "day_of_week": date.strftime("%A"),
        "is_weekend": int(is_weekend),
        "is_exam_week": int(is_exam_week),
        "sleep_hours": round(sleep_hours, 2),
        "social_media_hours": round(social_media_hours, 2),
        "study_app_hours": round(study_app_hours, 2),
        "class_hours": round(class_hours, 2),
        "energy_self_log": round(energy, 1),
        "focus_hours": round(focus_hours, 2),
    })

    prev_focus_hours = focus_hours

df = pd.DataFrame(rows)


df["prev_day_focus_hours"] = df["focus_hours"].shift(1).fillna(df["focus_hours"].mean())
df["rolling_3day_focus_avg"] = df["focus_hours"].rolling(window=3, min_periods=1).mean().shift(1).fillna(df["focus_hours"].mean())

threshold = df["focus_hours"].quantile(0.60)
df["high_productivity_day"] = (df["focus_hours"] >= threshold).astype(int)

df.to_csv("daily_log.csv", index=False)
print(f"Generated {len(df)} days of synthetic data.")
print(f"High-productivity threshold (focus_hours >= {threshold:.2f}): {df['high_productivity_day'].mean():.1%} of days")
print(df.head(10).to_string())
