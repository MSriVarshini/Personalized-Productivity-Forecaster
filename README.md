Focus Log — Personalized Productivity Forecaster

A "digital twin" of a student's study habits: predicts how focused a given day will be from screen-time and lifestyle signals, then turns that prediction into a plain-English recommendation.

Problem statement

Most productivity apps track time after the fact. This project instead forecasts before the day is over: given today's sleep, screen time, schedule load, and yesterday's momentum, how many focused study hours should I expect — and is today a high- or low-productivity day?

Data

data/generate_synthetic_data.py generates 150 days of synthetic daily logs (sleep hours, social media / study-app screen time, class load, self-rated energy, exam-week flags) with realistic built-in relationships (e.g. poor sleep and high social media use suppress focus; exam weeks boost it; yesterday's focus has momentum into today).

This is clearly-labeled synthetic data, built so the full pipeline runs end-to-end without needing weeks of personal logging first. The generator is written so real Screen Time / Digital Wellbeing exports and calendar data can be substituted in with the same schema.
