"""
Jackpot Neural Network Predictor
Specialized script for making jackpot predictions using the trained neural network model.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from datetime import datetime, timedelta
import requests
import json
from neural_network_jackpot_model import JackpotNeuralNetwork
import warnings
warnings.filterwarnings('ignore')

class JackpotPredictor:
    """
    Specialized class for making jackpot predictions using neural networks.
    """
    
    def __init__(self, model_path='jackpot_neural_model.h5', scaler_path='scaler.pkl'):
        """
        Initialize the jackpot predictor.
        
        Args:
            model_path: Path to the trained neural network model
            scaler_path: Path to the fitted scaler
        """
        self.model = JackpotNeuralNetwork()
        self.model.load_model(model_path, scaler_path)
        self.predictions_history = []
        
    def prepare_match_features(self, match_data):
        """
        Prepare features for a single match or batch of matches.
        
        Args:
            match_data: DataFrame with match information
            
        Returns:
            DataFrame with engineered features
        """
        # This function should match your existing feature engineering pipeline
        # For now, we'll assume the data is already in the correct format
        
        required_features = [
            'HomeGoalsFor', 'HomeGoalsAgainst', 'AwayGoalsFor', 'AwayGoalsAgainst',
            'HomeWinStreak', 'AwayWinStreak', 'HomeCleanSheets', 'AwayCleanSheets',
            'HomeHomeWinRate', 'AwayAwayWinRate', 'H2HHomeWinRate', 'H2HAwayWinRate',
            'H2HDrawRate', 'day_of_week', 'month', 'season',
            'HomeElo', 'AwayElo', 'EloDiff', 'OddHome', 'OddDraw', 'OddAway'
        ]
        
        # Ensure all required features are present
        missing_features = [col for col in required_features if col not in match_data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add default values for missing features
            for feature in missing_features:
                match_data[feature] = 0.0
        
        return match_data[required_features]
    
    def predict_single_match(self, match_features, confidence_threshold=0.6):
        """
        Predict outcome for a single match.
        
        Args:
            match_features: Dictionary or DataFrame with match features
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame if needed
        if isinstance(match_features, dict):
            match_features = pd.DataFrame([match_features])
        
        # Prepare features
        features_df = self.prepare_match_features(match_features)
        
        # Make prediction
        predictions, high_conf = self.model.predict_jackpot(features_df, confidence_threshold)
        
        if len(predictions) == 0:
            return None
        
        result = predictions.iloc[0]
        
        # Map prediction to outcome
        outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        predicted_outcome = outcome_map.get(result['predicted_outcome'], 'Unknown')
        
        return {
            'predicted_outcome': predicted_outcome,
            'confidence': result['confidence'],
            'home_probability': result['home_prob'],
            'draw_probability': result['draw_prob'],
            'away_probability': result['away_prob'],
            'high_confidence': result['confidence'] >= confidence_threshold
        }
    
    def predict_jackpot_matches(self, matches_data, confidence_threshold=0.7, min_matches=10):
        """
        Predict outcomes for jackpot matches with filtering.
        
        Args:
            matches_data: DataFrame with match information
            confidence_threshold: Minimum confidence for predictions
            min_matches: Minimum number of matches to include
            
        Returns:
            DataFrame with filtered predictions
        """
        print(f"Predicting outcomes for {len(matches_data)} matches...")
        
        # Prepare features for all matches
        features_df = self.prepare_match_features(matches_data)
        
        # Make predictions
        all_predictions, high_conf_predictions = self.model.predict_jackpot(
            features_df, confidence_threshold
        )
        
        # Add match information back to predictions
        result_df = matches_data.copy()
        for col in ['predicted_outcome', 'confidence', 'home_prob', 'draw_prob', 'away_prob']:
            result_df[col] = all_predictions[col]
        
        # Filter by confidence and minimum matches
        filtered_predictions = result_df[result_df['confidence'] >= confidence_threshold]
        
        if len(filtered_predictions) < min_matches:
            print(f"Warning: Only {len(filtered_predictions)} matches meet confidence threshold")
            print("Lowering confidence threshold to get minimum matches...")
            # Lower threshold to get minimum matches
            filtered_predictions = result_df.nlargest(min_matches, 'confidence')
        
        # Sort by confidence
        filtered_predictions = filtered_predictions.sort_values('confidence', ascending=False)
        
        print(f"Selected {len(filtered_predictions)} matches for jackpot")
        print(f"Average confidence: {filtered_predictions['confidence'].mean():.3f}")
        
        return filtered_predictions
    
    def generate_jackpot_bet(self, predictions_df, stake_per_match=100):
        """
        Generate a jackpot bet slip from predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            stake_per_match: Stake amount per match
            
        Returns:
            Dictionary with bet slip information
        """
        bet_slip = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_matches': len(predictions_df),
            'total_stake': len(predictions_df) * stake_per_match,
            'stake_per_match': stake_per_match,
            'matches': []
        }
        
        outcome_map = {0: '1', 1: 'X', 2: '2'}
        
        for idx, row in predictions_df.iterrows():
            match_info = {
                'match_id': getattr(row, 'match_id', f'Match_{idx}'),
                'home_team': getattr(row, 'home_team', f'Home_{idx}'),
                'away_team': getattr(row, 'away_team', f'Away_{idx}'),
                'prediction': outcome_map.get(row['predicted_outcome'], 'Unknown'),
                'confidence': row['confidence'],
                'home_prob': row['home_prob'],
                'draw_prob': row['draw_prob'],
                'away_prob': row['away_prob']
            }
            bet_slip['matches'].append(match_info)
        
        return bet_slip
    
    def save_predictions(self, predictions_df, filename=None):
        """
        Save predictions to CSV file.
        
        Args:
            predictions_df: DataFrame with predictions
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'jackpot_neural_predictions_{timestamp}.csv'
        
        predictions_df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")
        
        return filename
    
    def load_historical_data(self, features_path, labels_path):
        """
        Load historical data for analysis.
        
        Args:
            features_path: Path to features CSV
            labels_path: Path to labels CSV
            
        Returns:
            DataFrame with historical data
        """
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(labels_path)
        
        # Combine features and labels
        historical_data = features_df.copy()
        historical_data['actual_outcome'] = labels_df.iloc[:, 0]
        
        return historical_data
    
    def analyze_performance(self, predictions_df, actual_outcomes):
        """
        Analyze prediction performance.
        
        Args:
            predictions_df: DataFrame with predictions
            actual_outcomes: Series with actual outcomes
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate accuracy
        correct_predictions = (predictions_df['predicted_outcome'] == actual_outcomes).sum()
        total_predictions = len(predictions_df)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate average confidence for correct predictions
        correct_mask = predictions_df['predicted_outcome'] == actual_outcomes
        avg_confidence_correct = predictions_df.loc[correct_mask, 'confidence'].mean()
        avg_confidence_incorrect = predictions_df.loc[~correct_mask, 'confidence'].mean()
        
        performance = {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_incorrect': avg_confidence_incorrect,
            'confidence_correlation': predictions_df['confidence'].corr(
                (predictions_df['predicted_outcome'] == actual_outcomes).astype(int)
            )
        }
        
        return performance

def main():
    """
    Main function to demonstrate jackpot prediction.
    """
    print("=== Jackpot Neural Network Predictor ===")
    
    # Initialize predictor
    predictor = JackpotPredictor()
    
    # Load sample data for demonstration
    try:
        # Load your engineered features
        features_df = pd.read_csv('engineered_kaggle_features_filtered_optimized.csv')
        
        # Take a sample for demonstration (first 20 matches)
        sample_matches = features_df.head(20)
        
        # Make predictions
        predictions = predictor.predict_jackpot_matches(
            sample_matches, 
            confidence_threshold=0.6, 
            min_matches=10
        )
        
        # Generate bet slip
        bet_slip = predictor.generate_jackpot_bet(predictions, stake_per_match=50)
        
        # Display results
        print("\n=== Jackpot Predictions ===")
        print(f"Date: {bet_slip['date']}")
        print(f"Total Matches: {bet_slip['total_matches']}")
        print(f"Total Stake: ${bet_slip['total_stake']}")
        print(f"Average Confidence: {predictions['confidence'].mean():.3f}")
        
        print("\n=== Match Predictions ===")
        for i, match in enumerate(bet_slip['matches'][:5], 1):  # Show first 5
            print(f"{i}. {match['home_team']} vs {match['away_team']}")
            print(f"   Prediction: {match['prediction']} (Confidence: {match['confidence']:.3f})")
            print(f"   Probabilities - Home: {match['home_prob']:.3f}, Draw: {match['draw_prob']:.3f}, Away: {match['away_prob']:.3f}")
            print()
        
        # Save predictions
        filename = predictor.save_predictions(predictions)
        
        print(f"Predictions saved to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the neural network model first by running:")
        print("python neural_network_jackpot_model.py")

if __name__ == "__main__":
    main()
