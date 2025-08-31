#!/usr/bin/env python3
"""
Quick Training Script for Neural Network Jackpot Model
Run this script to train your neural network model.
"""

import sys
import os
from neural_network_jackpot_model import JackpotNeuralNetwork, main

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} installed")
    except ImportError:
        print("✗ TensorFlow not installed. Installing...")
        os.system("pip install tensorflow")
    
    try:
        import optuna
        print(f"✓ Optuna {optuna.__version__} installed")
    except ImportError:
        print("✗ Optuna not installed. Installing...")
        os.system("pip install optuna")

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'engineered_kaggle_features_filtered_optimized.csv',
        'engineered_kaggle_labels_filtered_optimized.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✓ {file} found")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    return True

def main_training():
    """Main training function."""
    print("=== Neural Network Jackpot Model Training ===")
    print("Checking dependencies and data files...")
    
    # Check dependencies
    check_dependencies()
    
    # Check data files
    if not check_data_files():
        print("Please ensure all required data files are present.")
        return
    
    print("\nStarting training...")
    print("This may take several minutes depending on your data size and hardware.")
    
    try:
        # Run the main training function
        main()
        print("\n=== Training Completed Successfully! ===")
        print("You can now use the trained model for predictions.")
        print("Run: python jackpot_neural_predictor.py")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main_training()
