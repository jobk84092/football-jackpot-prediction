#!/usr/bin/env python3
"""
Simple, robust neural network for football predictions.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_simple_model(input_dim, num_classes):
    """
    Create a simple, robust neural network.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_robust():
    """
    Prepare data with robust preprocessing.
    """
    print("Loading data...")
    
    # Load data
    features_df = pd.read_csv('engineered_kaggle_features_filtered_optimized.csv')
    labels_df = pd.read_csv('engineered_kaggle_labels_filtered_optimized.csv')
    
    # Clean data
    features_df = features_df.dropna()
    labels_df = labels_df.dropna()
    
    # Ensure same length
    min_length = min(len(features_df), len(labels_df))
    features_df = features_df.iloc[:min_length]
    labels_df = labels_df.iloc[:min_length]
    
    X = features_df.values
    y = labels_df.iloc[:, 0].values.astype(int)
    
    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Check for class imbalance
    class_counts = np.bincount(y)
    print(f"Class distribution: Home={class_counts[0]}, Draw={class_counts[1]}, Away={class_counts[2]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_simple_model(X_train, y_train, X_test, y_test):
    """
    Train the simple model.
    """
    print("Training simple neural network...")
    
    model = create_simple_model(X_train.shape[1], len(np.unique(y_train)))
    
    # Train with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=1
    )
    
    return model, history

def evaluate_simple_model(model, X_test, y_test):
    """
    Evaluate the simple model.
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Check prediction distribution
    pred_dist = np.bincount(y_pred)
    true_dist = np.bincount(y_test)
    
    print(f"\nPrediction distribution: {pred_dist}")
    print(f"True distribution: {true_dist}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('simple_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, y_pred, y_pred_proba

def test_diverse_scenarios(model, scaler):
    """
    Test model on diverse scenarios to check for bias.
    """
    print("\nTesting diverse scenarios...")
    
    scenarios = [
        # Strong home win
        {
            'name': 'Strong Home Win',
            'features': [3.0, 0.5, 1.0, 2.5, 3.0, 0.0, 1.0, 0.0, 0.9, 0.2, 0.8, 0.1, 0.1, 1, 1, 2024, 1800, 1500, 300, 1.2, 5.0, 8.0]
        },
        # Strong away win
        {
            'name': 'Strong Away Win', 
            'features': [0.5, 2.5, 2.5, 0.5, 0.0, 3.0, 0.0, 1.0, 0.2, 0.9, 0.1, 0.8, 0.1, 1, 1, 2024, 1500, 1800, -300, 8.0, 5.0, 1.2]
        },
        # Clear draw
        {
            'name': 'Clear Draw',
            'features': [1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.3, 0.3, 0.4, 1, 1, 2024, 1700, 1700, 0, 2.5, 3.2, 2.5]
        },
        # Balanced match
        {
            'name': 'Balanced Match',
            'features': [1.8, 1.2, 1.8, 1.2, 1.0, 1.0, 0.0, 0.0, 0.6, 0.6, 0.4, 0.4, 0.2, 1, 1, 2024, 1750, 1750, 0, 2.0, 3.5, 3.0]
        }
    ]
    
    for scenario in scenarios:
        features = np.array(scenario['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        probabilities = prediction[0]
        
        outcome_map = {0: 'Home', 1: 'Draw', 2: 'Away'}
        print(f"{scenario['name']}: Predicted {outcome_map[predicted_class]} with probabilities {probabilities}")

def main():
    """
    Main function.
    """
    print("=== Simple Neural Network Model ===")
    print(f"Started at: {datetime.now()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data_robust()
    
    # Train model
    model, history = train_simple_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = evaluate_simple_model(model, X_test, y_test)
    
    # Test diverse scenarios
    test_diverse_scenarios(model, scaler)
    
    # Save model
    model.save('simple_jackpot_model.h5')
    import joblib
    joblib.dump(scaler, 'simple_scaler.pkl')
    
    print(f"\nModel saved as 'simple_jackpot_model.h5'")
    print(f"Scaler saved as 'simple_scaler.pkl'")
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
    plt.savefig('simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
