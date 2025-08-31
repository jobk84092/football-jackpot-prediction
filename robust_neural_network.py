#!/usr/bin/env python3
"""
Robust neural network with proper data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_data(features_df, labels_df):
    """
    Clean the data by removing outliers and extreme values.
    """
    print("Cleaning data...")
    
    # Remove rows with missing values
    features_df = features_df.dropna()
    labels_df = labels_df.dropna()
    
    # Ensure same length
    min_length = min(len(features_df), len(labels_df))
    features_df = features_df.iloc[:min_length]
    labels_df = labels_df.iloc[:min_length]
    
    # Remove extreme outliers from odds
    features_df = features_df[
        (features_df['OddHome'] > 0) & (features_df['OddHome'] < 20) &
        (features_df['OddDraw'] > 0) & (features_df['OddDraw'] < 15) &
        (features_df['OddAway'] > 0) & (features_df['OddAway'] < 25)
    ]
    
    # Get corresponding labels
    labels_df = labels_df.iloc[features_df.index]
    
    # Remove extreme Elo differences
    features_df = features_df[abs(features_df['EloDiff']) < 400]
    labels_df = labels_df.iloc[features_df.index]
    
    print(f"After cleaning: {len(features_df)} samples")
    
    return features_df, labels_df

def create_robust_model(input_dim, num_classes):
    """
    Create a robust neural network architecture.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First layer
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second layer
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third layer
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with class weights to handle imbalance
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_robust():
    """
    Prepare data with robust preprocessing.
    """
    print("Loading and preparing data...")
    
    # Load data
    features_df = pd.read_csv('engineered_kaggle_features_filtered_optimized.csv')
    labels_df = pd.read_csv('engineered_kaggle_labels_filtered_optimized.csv')
    
    # Clean data
    features_df, labels_df = clean_data(features_df, labels_df)
    
    # Prepare features and labels
    X = features_df.values
    y = labels_df.iloc[:, 0].values.astype(int)
    
    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Calculate class weights to handle imbalance
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {
        0: total_samples / (len(class_counts) * class_counts[0]),
        1: total_samples / (len(class_counts) * class_counts[1]),
        2: total_samples / (len(class_counts) * class_counts[2])
    }
    print(f"Class weights: {class_weights}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use RobustScaler to handle outliers
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, class_weights

def train_robust_model(X_train, y_train, X_test, y_test, class_weights):
    """
    Train the robust model with class weights.
    """
    print("Training robust neural network...")
    
    model = create_robust_model(X_train.shape[1], len(np.unique(y_train)))
    
    # Train with class weights and early stopping
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ],
        verbose=1
    )
    
    return model, history

def evaluate_robust_model(model, X_test, y_test):
    """
    Evaluate the robust model.
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
    
    # Calculate distribution ratios
    pred_ratios = pred_dist / len(y_pred)
    true_ratios = true_dist / len(y_test)
    
    print(f"\nPrediction ratios: Home={pred_ratios[0]:.3f}, Draw={pred_ratios[1]:.3f}, Away={pred_ratios[2]:.3f}")
    print(f"True ratios: Home={true_ratios[0]:.3f}, Draw={true_ratios[1]:.3f}, Away={true_ratios[2]:.3f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('robust_confusion_matrix.png', dpi=300, bbox_inches='tight')
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
    print("=== Robust Neural Network Model ===")
    print(f"Started at: {datetime.now()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, class_weights = prepare_data_robust()
    
    # Train model
    model, history = train_robust_model(X_train, y_train, X_test, y_test, class_weights)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = evaluate_robust_model(model, X_test, y_test)
    
    # Test diverse scenarios
    test_diverse_scenarios(model, scaler)
    
    # Save model
    model.save('robust_jackpot_model.h5')
    import joblib
    joblib.dump(scaler, 'robust_scaler.pkl')
    
    print(f"\nModel saved as 'robust_jackpot_model.h5'")
    print(f"Scaler saved as 'robust_scaler.pkl'")
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
    plt.savefig('robust_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()
