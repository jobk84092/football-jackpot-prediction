import pandas as pd
import joblib
from datetime import datetime
import requests
from fuzzywuzzy import process

def get_vegas_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching odds: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return None

def get_team_rating(team_name, league):
    """Get team rating based on league position and form"""
    # League position ratings (1-20)
    position_ratings = {
        'England': {
            'Liverpool FC': 1,
            'Arsenal FC': 2,
            'Tottenham Hotspur': 5,
            'Crystal Palace': 14,
            'Wycombe Wanderers': 15,
            'Charlton Athletic': 16
        },
        'France': {
            'Stade Reims': 8,
            'AS Saint-Etienne': 12,
            'AJ Auxerre': 9,
            'FC Nantes': 11,
            'Toulouse FC': 13,
            'RC Lens': 7
        },
        'Spain': {
            'CD Leganes': 4,
            'Espanyol Barcelona': 3
        },
        'Switzerland': {
            'FC Luzern': 6,
            'Lausanne-Sport': 10,
            'Servette Geneva': 5,
            'Young Boys Bern': 2
        },
        'Netherlands': {
            'Feyenoord Rotterdam': 2,
            'PSV Eindhoven': 1,
            'FC Twente Enschede': 4,
            'FC Utrecht': 7
        },
        'Italy': {
            'Hellas Verona': 15,
            'US Lecce': 13
        },
        'Germany': {
            'Bayer Leverkusen': 1,
            'Borussia Dortmund': 4
        },
        'Greece': {
            'PAOK Thessaloniki': 2,
            'AEK Athens': 1,
            'Panathinaikos Athens': 3,
            'Olympiacos Piraeus': 4
        },
        'Portugal': {
            'Moreirense FC': 8,
            'Estoril Praia': 9
        },
        'Belgium': {
            'FCV Dender EH': 12,
            'Oud-Heverlee Leuven': 8
        }
    }
    
    # Convert position to rating (1st = 2000, 20th = 1000)
    if league in position_ratings and team_name in position_ratings[league]:
        position = position_ratings[league][team_name]
        rating = 2000 - (position - 1) * 50  # Linear scale from 2000 to 1000
        return rating
    
    # Default rating for unknown teams
    return 1500

def test_previous_jackpot():
    # Load the model
    model = joblib.load('football_model.pkl')
    
    # Load Vegas API key
    with open('vegas_api_key.txt', 'r') as f:
        api_key = f.read().strip()
    
    # Get Vegas odds
    odds_data = get_vegas_odds(api_key)
    
    # Last weekend's jackpot matches
    jackpot_matches = [
        {"HomeTeam": "Stade Reims", "AwayTeam": "AS Saint-Etienne", "HomeOdds": 2.13, "DrawOdds": 3.75, "AwayOdds": 3.30, "Date": "2024-05-10", "League": "France"},
        {"HomeTeam": "AJ Auxerre", "AwayTeam": "FC Nantes", "HomeOdds": 2.90, "DrawOdds": 3.50, "AwayOdds": 2.43, "Date": "2024-05-10", "League": "France"},
        {"HomeTeam": "Toulouse FC", "AwayTeam": "RC Lens", "HomeOdds": 2.28, "DrawOdds": 3.55, "AwayOdds": 3.15, "Date": "2024-05-10", "League": "France"},
        {"HomeTeam": "CD Leganes", "AwayTeam": "Espanyol Barcelona", "HomeOdds": 2.44, "DrawOdds": 3.10, "AwayOdds": 3.30, "Date": "2024-05-11", "League": "Spain"},
        {"HomeTeam": "FC Luzern", "AwayTeam": "Lausanne-Sport", "HomeOdds": 2.41, "DrawOdds": 3.45, "AwayOdds": 2.65, "Date": "2024-05-11", "League": "Switzerland"},
        {"HomeTeam": "Feyenoord Rotterdam", "AwayTeam": "PSV Eindhoven", "HomeOdds": 2.60, "DrawOdds": 3.75, "AwayOdds": 2.39, "Date": "2024-05-11", "League": "Netherlands"},
        {"HomeTeam": "FC Twente Enschede", "AwayTeam": "FC Utrecht", "HomeOdds": 2.36, "DrawOdds": 3.65, "AwayOdds": 2.70, "Date": "2024-05-11", "League": "Netherlands"},
        {"HomeTeam": "Hellas Verona", "AwayTeam": "US Lecce", "HomeOdds": 2.20, "DrawOdds": 3.10, "AwayOdds": 3.80, "Date": "2024-05-11", "League": "Italy"},
        {"HomeTeam": "Tottenham Hotspur", "AwayTeam": "Crystal Palace", "HomeOdds": 2.65, "DrawOdds": 3.70, "AwayOdds": 2.50, "Date": "2024-05-11", "League": "England"},
        {"HomeTeam": "Bayer Leverkusen", "AwayTeam": "Borussia Dortmund", "HomeOdds": 2.70, "DrawOdds": 3.95, "AwayOdds": 2.40, "Date": "2024-05-11", "League": "Germany"},
        {"HomeTeam": "Servette Geneva", "AwayTeam": "Young Boys Bern", "HomeOdds": 2.22, "DrawOdds": 3.60, "AwayOdds": 2.85, "Date": "2024-05-11", "League": "Switzerland"},
        {"HomeTeam": "Liverpool FC", "AwayTeam": "Arsenal FC", "HomeOdds": 2.06, "DrawOdds": 3.60, "AwayOdds": 3.60, "Date": "2024-05-11", "League": "England"},
        {"HomeTeam": "PAOK Thessaloniki", "AwayTeam": "AEK Athens", "HomeOdds": 2.15, "DrawOdds": 3.10, "AwayOdds": 3.00, "Date": "2024-05-11", "League": "Greece"},
        {"HomeTeam": "Moreirense FC", "AwayTeam": "Estoril Praia", "HomeOdds": 2.40, "DrawOdds": 3.30, "AwayOdds": 2.95, "Date": "2024-05-11", "League": "Portugal"},
        {"HomeTeam": "Panathinaikos Athens", "AwayTeam": "Olympiacos Piraeus", "HomeOdds": 2.28, "DrawOdds": 3.05, "AwayOdds": 2.85, "Date": "2024-05-11", "League": "Greece"},
        {"HomeTeam": "FCV Dender EH", "AwayTeam": "Oud-Heverlee Leuven", "HomeOdds": 2.37, "DrawOdds": 3.45, "AwayOdds": 2.75, "Date": "2024-05-11", "League": "Belgium"},
        {"HomeTeam": "Wycombe Wanderers", "AwayTeam": "Charlton Athletic", "HomeOdds": 2.33, "DrawOdds": 3.10, "AwayOdds": 3.05, "Date": "2024-05-11", "League": "England"}
    ]
    
    # Prepare features for each match
    features_list = []
    for match in jackpot_matches:
        home_rating = get_team_rating(match['HomeTeam'], match['League'])
        away_rating = get_team_rating(match['AwayTeam'], match['League'])
        
        features = {
            'HomeGoalsFor': 2.0,  # Example values, should be calculated from historical data
            'HomeGoalsAgainst': 1.0,
            'AwayGoalsFor': 1.5,
            'AwayGoalsAgainst': 1.2,
            'day_of_week': datetime.strptime(match['Date'], '%Y-%m-%d').weekday(),
            'month': datetime.strptime(match['Date'], '%Y-%m-%d').month,
            'HomeElo': home_rating,
            'AwayElo': away_rating,
            'EloDiff': home_rating - away_rating,
            'vegas_home_odds': match['HomeOdds'],
            'vegas_draw_odds': match['DrawOdds'],
            'vegas_away_odds': match['AwayOdds']
        }
        features_list.append(features)
    
    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)
    
    # Make predictions
    predictions = model.predict_proba(df_features)
    
    # Print results
    print("\nPredictions for Last Weekend's Jackpot:")
    for i, match in enumerate(jackpot_matches):
        probs = predictions[i]
        print(f"\n{match['HomeTeam']} vs {match['AwayTeam']}:")
        print(f"Home Win: {probs[0]:.2%}")
        print(f"Draw: {probs[1]:.2%}")
        print(f"Away Win: {probs[2]:.2%}")
        print(f"Predicted Outcome: {'Home Win' if probs[0] > max(probs[1], probs[2]) else 'Draw' if probs[1] > max(probs[0], probs[2]) else 'Away Win'}")
        print(f"Actual Odds - Home: {match['HomeOdds']}, Draw: {match['DrawOdds']}, Away: {match['AwayOdds']}")

if __name__ == "__main__":
    test_previous_jackpot() 