"""
Neural Network Jackpot Betting Model
A TensorFlow-based deep learning model for predicting football match outcomes for jackpot betting.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class JackpotNeuralNetwork:
    """
    Neural Network model for jackpot betting predictions.
    """
    
    def __init__(self, input_dim=None, num_classes=3, model_path=None):
        """
        Initialize the neural network model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (3 for Home/Draw/Away)
            model_path: Path to load pre-trained model
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
        if model_path:
            self.load_model(model_path)
    
    def create_model(self, input_dim, num_classes, learning_rate=0.001):
        """
        Create a deep neural network architecture optimized for football predictions.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # First dense layer with batch normalization and dropout
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second dense layer
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third dense layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth dense layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, features_path, labels_path, test_size=0.2, random_state=42):
        """
        Prepare and preprocess the data for training.
        
        Args:
            features_path: Path to features CSV file
            labels_path: Path to labels CSV file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test: Split and preprocessed data
        """
        print("Loading and preparing data...")
        
        # Load data
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(labels_path)
        
        # Remove any rows with missing values
        features_df = features_df.dropna()
        labels_df = labels_df.dropna()
        
        # Ensure features and labels have same length
        min_length = min(len(features_df), len(labels_df))
        features_df = features_df.iloc[:min_length]
        labels_df = labels_df.iloc[:min_length]
        
        # Prepare features
        X = features_df.values
        
        # Prepare labels (convert to numeric if needed)
        y = labels_df.iloc[:, 0].values  # Take first column as target
        
        # Encode labels if they're not numeric
        if not np.issubdtype(y.dtype, np.number):
            y = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Output classes: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split if no validation data provided
        """
        print("Training neural network model...")
        
        # Create model if not exists
        if self.model is None:
            self.model = self.create_model(X_train.shape[1], len(np.unique(y_train)))
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_jackpot_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print results
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def predict_jackpot(self, features_df, confidence_threshold=0.6):
        """
        Make predictions for jackpot betting with confidence filtering.
        
        Args:
            features_df: DataFrame with match features
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        print("Making jackpot predictions...")
        
        # Preprocess features
        X = self.scaler.transform(features_df.values)
        
        # Make predictions
        predictions_proba = self.model.predict(X)
        predictions = np.argmax(predictions_proba, axis=1)
        confidence_scores = np.max(predictions_proba, axis=1)
        
        # Create results DataFrame
        results_df = features_df.copy()
        results_df['predicted_outcome'] = predictions
        results_df['confidence'] = confidence_scores
        results_df['home_prob'] = predictions_proba[:, 0]
        results_df['draw_prob'] = predictions_proba[:, 1]
        results_df['away_prob'] = predictions_proba[:, 2]
        
        # Filter by confidence threshold
        high_confidence = results_df[results_df['confidence'] >= confidence_threshold]
        
        print(f"Total predictions: {len(results_df)}")
        print(f"High confidence predictions (≥{confidence_threshold}): {len(high_confidence)}")
        
        return results_df, high_confidence
    
    def save_model(self, model_path='jackpot_neural_model.h5', scaler_path='scaler.pkl'):
        """
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        if self.model is not None:
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='jackpot_neural_model.h5', scaler_path='scaler.pkl'):
        """
        Load a trained model and scaler.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
        """
        try:
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def plot_training_history(self):
        """
        Plot the training history.
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def optimize_hyperparameters(X_train, y_train, X_test, y_test, n_trials=50):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_trials: Number of optimization trials
        
    Returns:
        Best hyperparameters
    """
    def objective(trial):
        # Define hyperparameter search space
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        
        # Create model with trial hyperparameters
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        for i in range(num_layers):
            units = trial.suggest_int(f'units_layer_{i}', 32, 256)
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
        )
        
        return history.history['val_accuracy'][-1]
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)
    
    return study.best_params

def main():
    """
    Main function to run the complete neural network training pipeline.
    """
    print("=== Neural Network Jackpot Betting Model ===")
    print(f"Started at: {datetime.now()}")
    
    # Initialize the model
    model = JackpotNeuralNetwork()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(
        'engineered_kaggle_features_filtered_optimized.csv',
        'engineered_kaggle_labels_filtered_optimized.csv'
    )
    
    # Optional: Optimize hyperparameters (uncomment if needed)
    # best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test, n_trials=20)
    
    # Train the model
    model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    # Evaluate the model
    results = model.evaluate(X_test, y_test)
    
    # Plot training history
    model.plot_training_history()
    
    # Save the model
    model.save_model()
    
    # Example: Make predictions on new data
    print("\n=== Making Sample Predictions ===")
    sample_features = pd.read_csv('engineered_kaggle_features_filtered_optimized.csv').head(10)
    predictions, high_conf_predictions = model.predict_jackpot(sample_features, confidence_threshold=0.7)
    
    print("\nSample predictions:")
    print(predictions[['predicted_outcome', 'confidence', 'home_prob', 'draw_prob', 'away_prob']].head())
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
