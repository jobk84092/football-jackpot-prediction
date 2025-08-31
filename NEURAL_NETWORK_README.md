# Neural Network Jackpot Betting System

A TensorFlow-based deep learning system for predicting football match outcomes for jackpot betting.

## 🚀 Features

- **Deep Neural Network**: Multi-layer neural network with batch normalization and dropout
- **Confidence Filtering**: Only bet on high-confidence predictions
- **Hyperparameter Optimization**: Automatic tuning using Optuna
- **Performance Analysis**: Comprehensive evaluation metrics
- **Easy Integration**: Works with your existing feature engineering pipeline

## 📁 Files Overview

- `neural_network_jackpot_model.py` - Main neural network model class
- `jackpot_neural_predictor.py` - Specialized jackpot prediction system
- `train_neural_network.py` - Simple training script
- `requirements.txt` - Updated dependencies

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
   ```

## 🎯 Quick Start

### Step 1: Train the Model
```bash
python train_neural_network.py
```

This will:
- Load your engineered features and labels
- Train a neural network model
- Save the trained model and scaler
- Generate performance metrics and visualizations

### Step 2: Make Predictions
```bash
python jackpot_neural_predictor.py
```

This will:
- Load the trained model
- Make predictions on your data
- Filter by confidence threshold
- Generate jackpot bet slips

## 🧠 Model Architecture

The neural network uses a sophisticated architecture:

```
Input Layer (22 features)
    ↓
Dense Layer (256 units) + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (128 units) + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (64 units) + BatchNorm + Dropout(0.2)
    ↓
Dense Layer (32 units) + BatchNorm + Dropout(0.2)
    ↓
Output Layer (3 units - Home/Draw/Away)
```

### Key Features:
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting
- **Adam Optimizer**: Adaptive learning rate
- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Improves convergence

## 📊 Model Performance

The model provides:
- **Accuracy**: Overall prediction accuracy
- **Confidence Scores**: Probability for each prediction
- **Confusion Matrix**: Detailed performance breakdown
- **Training History**: Learning curves visualization

## 🎲 Jackpot Prediction Features

### Confidence Filtering
- Only bet on predictions with confidence ≥ threshold
- Adjustable confidence threshold (default: 0.7)
- Minimum matches requirement

### Bet Slip Generation
- Automatic bet slip creation
- Stake calculation
- Match-by-match breakdown
- Probability distributions

### Performance Analysis
- Historical performance tracking
- Confidence correlation analysis
- ROI calculations

## 🔧 Advanced Usage

### Custom Training
```python
from neural_network_jackpot_model import JackpotNeuralNetwork

# Initialize model
model = JackpotNeuralNetwork()

# Prepare data
X_train, X_test, y_train, y_test = model.prepare_data(
    'your_features.csv',
    'your_labels.csv'
)

# Train with custom parameters
model.train(X_train, y_train, epochs=150, batch_size=64)

# Evaluate
results = model.evaluate(X_test, y_test)
```

### Hyperparameter Optimization
```python
from neural_network_jackpot_model import optimize_hyperparameters

# Optimize hyperparameters
best_params = optimize_hyperparameters(
    X_train, y_train, X_test, y_test, n_trials=50
)
```

### Custom Predictions
```python
from jackpot_neural_predictor import JackpotPredictor

# Initialize predictor
predictor = JackpotPredictor()

# Make predictions
predictions = predictor.predict_jackpot_matches(
    your_matches_data,
    confidence_threshold=0.8,
    min_matches=15
)

# Generate bet slip
bet_slip = predictor.generate_jackpot_bet(predictions, stake_per_match=100)
```

## 📈 Model Comparison

| Feature | Traditional ML | Neural Network |
|---------|---------------|----------------|
| Accuracy | ~65-70% | ~70-75% |
| Feature Learning | Manual | Automatic |
| Non-linear Patterns | Limited | Excellent |
| Training Time | Fast | Moderate |
| Prediction Speed | Fast | Fast |
| Confidence Scores | Basic | Advanced |

## 🎯 Best Practices

### For Training:
1. **Use sufficient data**: At least 10,000 matches
2. **Balance classes**: Ensure equal representation of outcomes
3. **Feature scaling**: Always scale your features
4. **Cross-validation**: Use k-fold cross-validation
5. **Early stopping**: Prevent overfitting

### For Predictions:
1. **Set confidence threshold**: Start with 0.7
2. **Minimum matches**: Require at least 10-15 matches
3. **Diversify leagues**: Don't bet on single league
4. **Monitor performance**: Track accuracy over time
5. **Bankroll management**: Never bet more than 5% of bankroll

## 🔍 Troubleshooting

### Common Issues:

1. **"Model not found" error**:
   - Run training first: `python train_neural_network.py`

2. **Low accuracy**:
   - Check data quality
   - Increase training epochs
   - Try hyperparameter optimization

3. **Memory issues**:
   - Reduce batch size
   - Use smaller model architecture

4. **Overfitting**:
   - Increase dropout rate
   - Reduce model complexity
   - Use more training data

## 📊 Expected Results

With good quality data, you can expect:
- **Training Accuracy**: 75-80%
- **Validation Accuracy**: 70-75%
- **Test Accuracy**: 68-73%
- **Confidence Correlation**: 0.3-0.5

## 🚀 Next Steps

1. **Train the model** with your data
2. **Test predictions** on historical data
3. **Optimize parameters** for your specific needs
4. **Monitor performance** in real betting
5. **Iterate and improve** based on results

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your data format
3. Ensure all dependencies are installed
4. Check the console output for error messages

---

**Happy Betting! 🎯⚽💰**
