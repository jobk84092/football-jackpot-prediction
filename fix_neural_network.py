#!/usr/bin/env python3
"""
Fix the neural network model with better training parameters.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_better_model(input_dim, num_classes):
    """
    Create a better neural network architecture.
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First dense layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second dense layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with better parameters
    optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    """
    Prepare data with better preprocessing.
    """
    print("Loading and preparing data...")
    
    # Load data
    features_df = pd.read_csv('engineered_kaggle_features_filtered_optimized.csv')
    labels_df = pd.read_csv('engineered_kaggle_labels_filtered_optimized.csv')
    
    # Remove any rows with missing values
    features_df = features_df.dropna()
    labels_df = labels_df.dropna()
    
    # Ensure features and labels have same length
    min_length = min(len(features_df), len(labels_df))
    features_df = features_df.iloc[:min_length]
    labels_df = labels_df.iloc[:min_length]
    
    # Prepare features
    X = features_df.values
    y = labels_df.iloc[:, 0].values
    
    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_better_model(X_train, y_train, X_test, y_test):
    """
    Train the model with better parameters.
    """
    print("Training improved neural network model...")
    
    # Create model
    model = create_better_model(X_train.shape[1], len(np.unique(y_train)))
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'better_jackpot_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
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
    plt.savefig('better_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Check prediction distribution
    pred_dist = np.bincount(y_pred)
    print(f"\nPrediction distribution: {pred_dist}")
    print(f"True distribution: {np.bincount(y_test.astype(int))}")
    
    return accuracy, y_pred, y_pred_proba

def test_predictions(model, scaler):
    """
    Test predictions on sample data to check for bias.
    """
    print("\nTesting predictions on sample data...")
    
    # Create diverse test data
    test_features = []
    
    # Home win scenario
    home_win = np.array([2.0, 0.5, 1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.8, 0.3, 0.7, 0.2, 0.1, 1, 1, 2024, 1800, 1600, 200, 1.5, 4.0, 6.0])
    test_features.append(home_win)
    
    # Away win scenario
    away_win = np.array([1.0, 2.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.3, 0.8, 0.2, 0.7, 0.1, 1, 1, 2024, 1600, 1800, -200, 6.0, 4.0, 1.5])
    test_features.append(away_win)
    
    # Draw scenario
    draw = np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.3, 0.3, 0.4, 1, 1, 2024, 1700, 1700, 0, 2.5, 3.2, 2.5])
    test_features.append(draw)
    
    test_features = np.array(test_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Make predictions
    predictions = model.predict(test_features_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    
    scenarios = ['Home Win', 'Away Win', 'Draw']
    for i, (scenario, pred_class, probs) in enumerate(zip(scenarios, predicted_classes, predictions)):
        print(f"{scenario} scenario: Predicted class {pred_class} with probabilities {probs}")
    
    return predicted_classes

def main():
    """
    Main function to fix and retrain the model.
    """
    print("=== Fixing Neural Network Model ===")
    print(f"Started at: {datetime.now()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # Train better model
    model, history = train_better_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Test predictions
    test_preds = test_predictions(model, scaler)
    
    # Save model and scaler
    model.save('better_jackpot_model.h5')
    import joblib
    joblib.dump(scaler, 'better_scaler.pkl')
    
    print(f"\nModel saved as 'better_jackpot_model.h5'")
    print(f"Scaler saved as 'better_scaler.pkl'")
    print(f"Final accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('better_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
