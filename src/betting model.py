# Import necessary libraries
import requests
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Function to fetch fixtures
def fetch_fixtures(api_key, league_id, season, date):
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"}
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    params = {"league": league_id, "season": season, "date": date}
    response = requests.get(url, headers=headers, params=params)
    df = pd.json_normalize(response.json()['response'])
    return df

# Function to add form feature
def add_form_feature(df, team_col, goals_col):
    df['form'] = df.groupby(team_col)[goals_col].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    return df

# Function to get base models
def get_base_models():
    return [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ]

# Function to build ensemble
def build_ensemble(X_train, y_train):
    base_models = get_base_models()
    ensemble = VotingClassifier(estimators=base_models, voting='hard')
    ensemble.fit(X_train, y_train)
    return ensemble

# Function to tune model
def tune_model(model, param_grid, X, y):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

# Function to create database connection
def create_connection(db_file="betting_data.db"):
    conn = sqlite3.connect(db_file)
    return conn

# Function to create table
def create_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        date TEXT,
        league TEXT,
        home_team TEXT,
        away_team TEXT,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        home_goals INTEGER,
        away_goals INTEGER,
        home_form REAL,
        away_form REAL,
        outcome INTEGER
    );
    """
    conn.execute(sql)
    conn.commit()
