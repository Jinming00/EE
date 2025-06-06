
# iTransformer Wind Power Prediction - Performance Report

## Model Architecture
- **Model Type**: iTransformer (Inverted Transformer)
- **Input Sequence**: 96 points (24.0 hours)
- **Prediction Length**: 16 points (4.0 hours)
- **Model Dimension**: 64
- **Attention Heads**: 8
- **Encoder Layers**: 2
- **Total Parameters**: 107,216
- **Input Features**: 57

## Training Results
- **Best Validation CR**: 58.37% (Epoch 84)
- **Total Training Epochs**: 84
- **Final Learning Rate**: 6.25e-06

## Test Set Performance
- **Overall CR**: 52.54%
- **RMSE**: 0.135399
- **MAE**: 0.096981
- **MSE**: 0.018333

## Step-wise Performance Analysis
- **Average CR**: 53.85%
- **CR Standard Deviation**: 11.06%
- **Best Step**: Step 1 (CR: 76.86%)
- **Worst Step**: Step 16 (CR: 38.03%)

## Detailed Step-wise Metrics

### Step 1 (15 minutes ahead)
- **CR**: 76.86%
- **RMSE**: 0.065127
- **MAE**: 0.046199

### Step 2 (30 minutes ahead)
- **CR**: 70.72%
- **RMSE**: 0.083767
- **MAE**: 0.059274

### Step 3 (45 minutes ahead)
- **CR**: 66.71%
- **RMSE**: 0.095165
- **MAE**: 0.068051

### Step 4 (60 minutes ahead)
- **CR**: 63.19%
- **RMSE**: 0.105122
- **MAE**: 0.076041

### Step 5 (75 minutes ahead)
- **CR**: 60.35%
- **RMSE**: 0.112643
- **MAE**: 0.081755

### Step 6 (90 minutes ahead)
- **CR**: 57.56%
- **RMSE**: 0.121109
- **MAE**: 0.088254

### Step 7 (105 minutes ahead)
- **CR**: 55.95%
- **RMSE**: 0.127396
- **MAE**: 0.092949

### Step 8 (120 minutes ahead)
- **CR**: 53.12%
- **RMSE**: 0.134049
- **MAE**: 0.099365

### Step 9 (135 minutes ahead)
- **CR**: 51.41%
- **RMSE**: 0.139841
- **MAE**: 0.103213

### Step 10 (150 minutes ahead)
- **CR**: 49.75%
- **RMSE**: 0.144089
- **MAE**: 0.107109

### Step 11 (165 minutes ahead)
- **CR**: 46.72%
- **RMSE**: 0.152393
- **MAE**: 0.113076

### Step 12 (180 minutes ahead)
- **CR**: 45.46%
- **RMSE**: 0.156659
- **MAE**: 0.116387

### Step 13 (195 minutes ahead)
- **CR**: 44.04%
- **RMSE**: 0.160045
- **MAE**: 0.118723

### Step 14 (210 minutes ahead)
- **CR**: 41.73%
- **RMSE**: 0.165261
- **MAE**: 0.123117

### Step 15 (225 minutes ahead)
- **CR**: 39.96%
- **RMSE**: 0.170124
- **MAE**: 0.127192

### Step 16 (240 minutes ahead)
- **CR**: 38.03%
- **RMSE**: 0.174517
- **MAE**: 0.130993
