# Convert the Python script into a Jupyter Notebook format for better interactivity and visualization.

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
import nbformat as nbf

# Define functions

def fetch_fixtures(api_key, league_id, season, date):
    """Fetch fixtures from the API."""
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"}
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    params = {"league": league_id, "season": season, "date": date}
    response = requests.get(url, headers=headers, params=params)
    df = pd.json_normalize(response.json()['response'])
    return df

def add_form_feature(df, team_col, goals_col):
    """Add form feature to the dataset."""
    df['form'] = df.groupby(team_col)[goals_col].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    return df

def get_base_models():
    """Get base models for ensemble."""
    return [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ]

def build_ensemble(X_train, y_train):
    """Build an ensemble model."""
    base_models = get_base_models()
    ensemble = VotingClassifier(estimators=base_models, voting='hard')
    ensemble.fit(X_train, y_train)
    return ensemble

def tune_model(model, param_grid, X, y):
    """Tune the model using GridSearchCV."""
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

def create_connection(db_file="betting_data.db"):
    """Create a database connection."""
    conn = sqlite3.connect(db_file)
    return conn

def create_table(conn):
    """Create a table in the database."""
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

def convert_to_notebook(script_path, notebook_path):
    """Convert a Python script to a Jupyter Notebook."""
    with open(script_path, 'r') as script_file:
        script_lines = script_file.readlines()

    notebook = nbf.v4.new_notebook()
    cells = []

    for line in script_lines:
        if line.startswith('#'):  # Treat comments as Markdown cells
            cells.append(nbf.v4.new_markdown_cell(line.strip('#').strip()))
        else:  # Treat code as Code cells
            cells.append(nbf.v4.new_code_cell(line))

    notebook['cells'] = cells

    with open(notebook_path, 'w') as notebook_file:
        nbf.write(notebook, notebook_file)



# 4. Convert script to notebook
# convert_to_notebook('Football_Jackpot_Prediction.py', 'Football_Jackpot_Prediction.ipynb')

# Save this notebook as a .ipynb file for GitHub upload.