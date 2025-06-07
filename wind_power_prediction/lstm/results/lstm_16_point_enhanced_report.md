
# Enhanced LSTM 16-Point Prediction Model Performance Report

## Model Configuration
- Model Type: Multi-Step LSTM with Attention + Enhanced Feature Engineering
- Input Sequence Length: 96
- Prediction Sequence Length: 16
- Hidden Size: 128
- LSTM Layers: 2
- Dropout: 0.3
- Loss Function: Huber Loss (delta=1.0)
- Total Parameters: 362,800
- Trainable Parameters: 362,800
- Model Size: 1.38 MB

## Enhanced Feature Engineering
- Total Features: 57
- Feature Categories: {'original': 1, 'lag': 8, 'ramp': 9, 'other': 3, 'statistical': 30, 'anomaly': 4, 'trend': 2}

### Feature Types Enabled:
- Trend Features: True

## Training Strategy
- Early Stopping Strategy: Based on CR metric (higher CR is better)
- Best Validation CR: 65.68%
- Training Epochs: 26
- Batch Size: 128
- Learning Rate: 0.01
- Step-wise CR Tracking: Enabled



## Step-wise Detailed Metrics

### Step 1
- MSE: 0.001795
- MAE: 0.027364
- CR: 85.94%

### Step 2
- MSE: 0.004412
- MAE: 0.044088
- CR: 78.50%

### Step 3
- MSE: 0.006748
- MAE: 0.055802
- CR: 73.92%

### Step 4
- MSE: 0.008665
- MAE: 0.064278
- CR: 70.65%

### Step 5
- MSE: 0.010396
- MAE: 0.071441
- CR: 67.82%

### Step 6
- MSE: 0.012024
- MAE: 0.078024
- CR: 65.19%

### Step 7
- MSE: 0.013859
- MAE: 0.083722
- CR: 63.52%

### Step 8
- MSE: 0.015683
- MAE: 0.089336
- CR: 61.75%

### Step 9
- MSE: 0.017379
- MAE: 0.094110
- CR: 59.90%

### Step 10
- MSE: 0.019118
- MAE: 0.099670
- CR: 58.21%

### Step 11
- MSE: 0.020990
- MAE: 0.105036
- CR: 56.60%

### Step 12
- MSE: 0.022823
- MAE: 0.110062
- CR: 55.09%

### Step 13
- MSE: 0.024645
- MAE: 0.114840
- CR: 53.47%

### Step 14
- MSE: 0.026236
- MAE: 0.118978
- CR: 51.57%

### Step 15
- MSE: 0.028250
- MAE: 0.123518
- CR: 50.11%

### Step 16
- MSE: 0.030293
- MAE: 0.127819
- CR: 48.65%
