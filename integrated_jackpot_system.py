"""
Integrated Jackpot Betting System
Combines traditional ML and neural network approaches for comprehensive jackpot predictions.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from neural_network_jackpot_model import JackpotNeuralNetwork
from jackpot_neural_predictor import JackpotPredictor
import warnings
warnings.filterwarnings('ignore')

class IntegratedJackpotSystem:
    """
    Integrated system combining traditional ML and neural network approaches.
    """
    
    def __init__(self):
        """
        Initialize the integrated system.
        """
        self.traditional_model = None
        self.neural_model = None
        self.neural_predictor = None
        self.ensemble_predictions = None
        
    def load_traditional_model(self, model_path='football_model_filtered_optimized.pkl'):
        """
        Load the traditional ML model.
        
        Args:
            model_path: Path to the traditional model
        """
        try:
            self.traditional_model = joblib.load(model_path)
            print(f"✓ Traditional model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Could not load traditional model: {e}")
    
    def load_neural_model(self, model_path='jackpot_neural_model.h5', scaler_path='scaler.pkl'):
        """
        Load the neural network model.
        
        Args:
            model_path: Path to the neural network model
            scaler_path: Path to the scaler
        """
        try:
            self.neural_predictor = JackpotPredictor(model_path, scaler_path)
            print(f"✓ Neural network model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Could not load neural network model: {e}")
    
    def compare_models(self, features_df, labels_df=None):
        """
        Compare predictions from both models.
        
        Args:
            features_df: DataFrame with features
            labels_df: DataFrame with actual outcomes (optional)
            
        Returns:
            DataFrame with comparison results
        """
        print("Comparing model predictions...")
        
        results = features_df.copy()
        
        # Traditional model predictions
        if self.traditional_model is not None:
            try:
                trad_predictions = self.traditional_model.predict(features_df)
                trad_proba = self.traditional_model.predict_proba(features_df)
                results['traditional_prediction'] = trad_predictions
                results['traditional_confidence'] = np.max(trad_proba, axis=1)
                results['traditional_home_prob'] = trad_proba[:, 0]
                results['traditional_draw_prob'] = trad_proba[:, 1]
                results['traditional_away_prob'] = trad_proba[:, 2]
                print("✓ Traditional model predictions added")
            except Exception as e:
                print(f"✗ Traditional model prediction failed: {e}")
        
        # Neural network predictions
        if self.neural_predictor is not None:
            try:
                neural_predictions, _ = self.neural_predictor.model.predict_jackpot(
                    features_df, confidence_threshold=0.0
                )
                results['neural_prediction'] = neural_predictions['predicted_outcome']
                results['neural_confidence'] = neural_predictions['confidence']
                results['neural_home_prob'] = neural_predictions['home_prob']
                results['neural_draw_prob'] = neural_predictions['draw_prob']
                results['neural_away_prob'] = neural_predictions['away_prob']
                print("✓ Neural network predictions added")
            except Exception as e:
                print(f"✗ Neural network prediction failed: {e}")
        
        # Add actual outcomes if available
        if labels_df is not None:
            results['actual_outcome'] = labels_df.iloc[:, 0]
        
        return results
    
    def create_ensemble_predictions(self, features_df, weights={'traditional': 0.4, 'neural': 0.6}):
        """
        Create ensemble predictions combining both models.
        
        Args:
            features_df: DataFrame with features
            weights: Dictionary with model weights
            
        Returns:
            DataFrame with ensemble predictions
        """
        print("Creating ensemble predictions...")
        
        # Get individual model predictions
        comparison_df = self.compare_models(features_df)
        
        # Calculate ensemble probabilities
        ensemble_df = comparison_df.copy()
        
        if 'traditional_home_prob' in ensemble_df.columns and 'neural_home_prob' in ensemble_df.columns:
            ensemble_df['ensemble_home_prob'] = (
                weights['traditional'] * ensemble_df['traditional_home_prob'] +
                weights['neural'] * ensemble_df['neural_home_prob']
            )
            ensemble_df['ensemble_draw_prob'] = (
                weights['traditional'] * ensemble_df['traditional_draw_prob'] +
                weights['neural'] * ensemble_df['neural_draw_prob']
            )
            ensemble_df['ensemble_away_prob'] = (
                weights['traditional'] * ensemble_df['traditional_away_prob'] +
                weights['neural'] * ensemble_df['neural_away_prob']
            )
            
            # Get ensemble prediction
            proba_cols = ['ensemble_home_prob', 'ensemble_draw_prob', 'ensemble_away_prob']
            ensemble_df['ensemble_prediction'] = np.argmax(ensemble_df[proba_cols].values, axis=1)
            ensemble_df['ensemble_confidence'] = np.max(ensemble_df[proba_cols].values, axis=1)
            
            print("✓ Ensemble predictions created")
        
        return ensemble_df
    
    def generate_optimal_jackpot(self, features_df, confidence_threshold=0.7, min_matches=10, 
                                model_choice='ensemble'):
        """
        Generate optimal jackpot predictions using the specified model.
        
        Args:
            features_df: DataFrame with features
            confidence_threshold: Minimum confidence threshold
            min_matches: Minimum number of matches
            model_choice: 'traditional', 'neural', or 'ensemble'
            
        Returns:
            DataFrame with filtered predictions
        """
        print(f"Generating optimal jackpot using {model_choice} model...")
        
        if model_choice == 'ensemble':
            predictions_df = self.create_ensemble_predictions(features_df)
            pred_col = 'ensemble_prediction'
            conf_col = 'ensemble_confidence'
        elif model_choice == 'neural':
            predictions_df, _ = self.neural_predictor.model.predict_jackpot(
                features_df, confidence_threshold=0.0
            )
            pred_col = 'predicted_outcome'
            conf_col = 'confidence'
        elif model_choice == 'traditional':
            predictions_df = features_df.copy()
            trad_predictions = self.traditional_model.predict(features_df)
            trad_proba = self.traditional_model.predict_proba(features_df)
            predictions_df['traditional_prediction'] = trad_predictions
            predictions_df['traditional_confidence'] = np.max(trad_proba, axis=1)
            pred_col = 'traditional_prediction'
            conf_col = 'traditional_confidence'
        else:
            raise ValueError("model_choice must be 'traditional', 'neural', or 'ensemble'")
        
        # Filter by confidence
        filtered_predictions = predictions_df[predictions_df[conf_col] >= confidence_threshold]
        
        if len(filtered_predictions) < min_matches:
            print(f"Warning: Only {len(filtered_predictions)} matches meet confidence threshold")
            print("Lowering confidence threshold to get minimum matches...")
            filtered_predictions = predictions_df.nlargest(min_matches, conf_col)
        
        # Sort by confidence
        filtered_predictions = filtered_predictions.sort_values(conf_col, ascending=False)
        
        print(f"Selected {len(filtered_predictions)} matches for jackpot")
        print(f"Average confidence: {filtered_predictions[conf_col].mean():.3f}")
        
        return filtered_predictions
    
    def analyze_model_performance(self, features_df, labels_df):
        """
        Analyze performance of all models.
        
        Args:
            features_df: DataFrame with features
            labels_df: DataFrame with actual outcomes
            
        Returns:
            Dictionary with performance metrics
        """
        print("Analyzing model performance...")
        
        # Get predictions from all models
        comparison_df = self.compare_models(features_df, labels_df)
        
        performance = {}
        
        # Calculate accuracy for each model
        if 'traditional_prediction' in comparison_df.columns:
            trad_accuracy = (comparison_df['traditional_prediction'] == comparison_df['actual_outcome']).mean()
            performance['traditional_accuracy'] = trad_accuracy
            print(f"Traditional Model Accuracy: {trad_accuracy:.4f}")
        
        if 'neural_prediction' in comparison_df.columns:
            neural_accuracy = (comparison_df['neural_prediction'] == comparison_df['actual_outcome']).mean()
            performance['neural_accuracy'] = neural_accuracy
            print(f"Neural Network Accuracy: {neural_accuracy:.4f}")
        
        if 'ensemble_prediction' in comparison_df.columns:
            ensemble_accuracy = (comparison_df['ensemble_prediction'] == comparison_df['actual_outcome']).mean()
            performance['ensemble_accuracy'] = ensemble_accuracy
            print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
        
        return performance
    
    def plot_model_comparison(self, features_df, labels_df):
        """
        Create visualization comparing model performances.
        
        Args:
            features_df: DataFrame with features
            labels_df: DataFrame with actual outcomes
        """
        print("Creating model comparison visualization...")
        
        # Get predictions
        comparison_df = self.compare_models(features_df, labels_df)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        accuracies = []
        model_names = []
        
        if 'traditional_prediction' in comparison_df.columns:
            trad_acc = (comparison_df['traditional_prediction'] == comparison_df['actual_outcome']).mean()
            accuracies.append(trad_acc)
            model_names.append('Traditional')
        
        if 'neural_prediction' in comparison_df.columns:
            neural_acc = (comparison_df['neural_prediction'] == comparison_df['actual_outcome']).mean()
            accuracies.append(neural_acc)
            model_names.append('Neural Network')
        
        if 'ensemble_prediction' in comparison_df.columns:
            ensemble_acc = (comparison_df['ensemble_prediction'] == comparison_df['actual_outcome']).mean()
            accuracies.append(ensemble_acc)
            model_names.append('Ensemble')
        
        axes[0, 0].bar(model_names, accuracies, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Confidence distribution
        if 'traditional_confidence' in comparison_df.columns:
            axes[0, 1].hist(comparison_df['traditional_confidence'], alpha=0.7, label='Traditional', bins=20)
        if 'neural_confidence' in comparison_df.columns:
            axes[0, 1].hist(comparison_df['neural_confidence'], alpha=0.7, label='Neural', bins=20)
        if 'ensemble_confidence' in comparison_df.columns:
            axes[0, 1].hist(comparison_df['ensemble_confidence'], alpha=0.7, label='Ensemble', bins=20)
        
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Prediction agreement
        if 'traditional_prediction' in comparison_df.columns and 'neural_prediction' in comparison_df.columns:
            agreement = (comparison_df['traditional_prediction'] == comparison_df['neural_prediction']).mean()
            axes[1, 0].pie([agreement, 1-agreement], labels=['Agree', 'Disagree'], autopct='%1.1f%%')
            axes[1, 0].set_title('Traditional vs Neural Agreement')
        
        # 4. Confidence vs Accuracy
        if 'neural_confidence' in comparison_df.columns:
            correct_predictions = (comparison_df['neural_prediction'] == comparison_df['actual_outcome'])
            axes[1, 1].scatter(comparison_df['neural_confidence'], correct_predictions, alpha=0.5)
            axes[1, 1].set_title('Neural Network: Confidence vs Accuracy')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Correct Prediction')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Model comparison visualization saved as 'model_comparison.png'")

def main():
    """
    Main function demonstrating the integrated system.
    """
    print("=== Integrated Jackpot Betting System ===")
    
    # Initialize system
    system = IntegratedJackpotSystem()
    
    # Load models
    system.load_traditional_model()
    system.load_neural_model()
    
    # Load data
    try:
        features_df = pd.read_csv('engineered_kaggle_features_filtered_optimized.csv')
        labels_df = pd.read_csv('engineered_kaggle_labels_filtered_optimized.csv')
        
        print(f"Loaded {len(features_df)} matches with {features_df.shape[1]} features")
        
        # Analyze performance
        performance = system.analyze_model_performance(features_df, labels_df)
        
        # Create visualization
        system.plot_model_comparison(features_df, labels_df)
        
        # Generate optimal jackpot using ensemble
        print("\n=== Generating Optimal Jackpot ===")
        optimal_predictions = system.generate_optimal_jackpot(
            features_df, 
            confidence_threshold=0.7, 
            min_matches=15,
            model_choice='ensemble'
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimal_predictions.to_csv(f'integrated_jackpot_predictions_{timestamp}.csv', index=False)
        
        print(f"\nResults saved to integrated_jackpot_predictions_{timestamp}.csv")
        print(f"Best model accuracy: {max(performance.values()):.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required files are present.")

if __name__ == "__main__":
    main()
