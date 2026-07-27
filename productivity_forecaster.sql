/*
===============================================================================
                Personalized Productivity Forecaster
                SQL Database Design & Feature Engineering
===============================================================================

Project Overview
----------------
The Personalized Productivity Forecaster is a machine learning project that
predicts a student's daily productivity by analyzing behavioral data such as
screen time, study sessions, and calendar events.

SQL was used as the data preparation layer to organize, transform, and
aggregate raw data into meaningful features that were later used to train
machine learning models and create dashboard visualizations.
===============================================================================

CREATE DATABASE IF NOT EXISTS ProductivityForecaster;
USE ProductivityForecaster;

CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(100),
    age INT,
    department VARCHAR(100)
);

CREATE TABLE Screen_Time_Logs (
    log_id INT PRIMARY KEY,
    student_id INT,
    log_date DATE,
    app_category VARCHAR(50),
    screen_minutes INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);

CREATE TABLE Calendar_Events (
    event_id INT PRIMARY KEY,
    student_id INT,
    event_date DATE,
    event_type VARCHAR(50),
    duration_minutes INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);

CREATE TABLE Study_Sessions (
    session_id INT PRIMARY KEY,
    student_id INT,
    study_date DATE,
    subject VARCHAR(100),
    study_minutes INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);

CREATE TABLE Daily_Productivity (
    productivity_id INT PRIMARY KEY,
    student_id INT,
    productivity_date DATE,
    focused_hours DECIMAL(4,2),
    productivity_level VARCHAR(20),
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);

SELECT
    s.student_id,
    st.study_date,
    SUM(CASE WHEN l.app_category='Social Media' THEN l.screen_minutes ELSE 0 END) AS social_media_minutes,
    SUM(CASE WHEN l.app_category='Entertainment' THEN l.screen_minutes ELSE 0 END) AS entertainment_minutes,
    SUM(CASE WHEN l.app_category='Education' THEN l.screen_minutes ELSE 0 END) AS education_minutes,
    ROUND(SUM(st.study_minutes)/60,2) AS study_hours,
    COUNT(c.event_id) AS academic_events
FROM Students s
LEFT JOIN Screen_Time_Logs l ON s.student_id=l.student_id
LEFT JOIN Study_Sessions st ON s.student_id=st.student_id AND st.study_date=l.log_date
LEFT JOIN Calendar_Events c ON s.student_id=c.student_id AND c.event_date=l.log_date
GROUP BY s.student_id, st.study_date;

CREATE VIEW vw_focus_trend AS
SELECT productivity_date, AVG(focused_hours) AS average_focus_hours
FROM Daily_Productivity
GROUP BY productivity_date;

CREATE VIEW vw_productivity_distribution AS
SELECT productivity_level, COUNT(*) AS total_students
FROM Daily_Productivity
GROUP BY productivity_level;

CREATE VIEW vw_screen_time_distribution AS
SELECT app_category, SUM(screen_minutes) AS total_screen_minutes
FROM Screen_Time_Logs
GROUP BY app_category;

CREATE VIEW vw_screen_vs_study AS
SELECT s.student_id,
       SUM(l.screen_minutes) AS total_screen_minutes,
       ROUND(SUM(st.study_minutes)/60,2) AS study_hours
FROM Students s
LEFT JOIN Screen_Time_Logs l ON s.student_id=l.student_id
LEFT JOIN Study_Sessions st ON s.student_id=st.student_id
GROUP BY s.student_id;

/* End of SQL Module */
