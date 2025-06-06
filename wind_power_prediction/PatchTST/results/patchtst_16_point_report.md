
# PatchTST Wind Power Prediction - Performance Report

## Model Architecture
- **Model Type**: PatchTST (Patching Time Series Transformer)
- **Input Sequence**: 96 points (24.0 hours)
- **Prediction Length**: 16 points (4.0 hours)
- **Patch Configuration**: 8 points per patch, 23 patches total
- **Model Dimension**: 64
- **Attention Heads**: 8
- **Encoder Layers**: 2
- **Total Parameters**: 132,752

## Training Results
- **Best Validation CR**: 64.41% (Epoch 46)
- **Total Training Epochs**: 46
- **Final Learning Rate**: 1.25e-03

## Test Set Performance
- **Overall CR**: 57.14%
- **RMSE**: 0.126239
- **MAE**: 0.086204
- **MSE**: 0.015936

## Step-wise Performance Analysis
- **Average CR**: 58.91%
- **CR Standard Deviation**: 12.20%
- **Best Step**: Step 1 (CR: 85.97%)
- **Worst Step**: Step 16 (CR: 42.26%)

## Detailed Step-wise Metrics

### Step 1 (15 minutes ahead)
- **CR**: 85.97%
- **RMSE**: 0.041711
- **MAE**: 0.026333

### Step 2 (30 minutes ahead)
- **CR**: 77.98%
- **RMSE**: 0.066382
- **MAE**: 0.043127

### Step 3 (45 minutes ahead)
- **CR**: 72.55%
- **RMSE**: 0.083062
- **MAE**: 0.055141

### Step 4 (60 minutes ahead)
- **CR**: 69.00%
- **RMSE**: 0.094455
- **MAE**: 0.063737

### Step 5 (75 minutes ahead)
- **CR**: 65.52%
- **RMSE**: 0.103242
- **MAE**: 0.071507

### Step 6 (90 minutes ahead)
- **CR**: 62.42%
- **RMSE**: 0.110753
- **MAE**: 0.077881

### Step 7 (105 minutes ahead)
- **CR**: 59.73%
- **RMSE**: 0.117746
- **MAE**: 0.083088

### Step 8 (120 minutes ahead)
- **CR**: 57.49%
- **RMSE**: 0.124285
- **MAE**: 0.088546

### Step 9 (135 minutes ahead)
- **CR**: 55.48%
- **RMSE**: 0.130607
- **MAE**: 0.093079

### Step 10 (150 minutes ahead)
- **CR**: 53.71%
- **RMSE**: 0.136415
- **MAE**: 0.097481

### Step 11 (165 minutes ahead)
- **CR**: 51.82%
- **RMSE**: 0.142318
- **MAE**: 0.102014

### Step 12 (180 minutes ahead)
- **CR**: 50.20%
- **RMSE**: 0.147802
- **MAE**: 0.106373

### Step 13 (195 minutes ahead)
- **CR**: 48.23%
- **RMSE**: 0.152641
- **MAE**: 0.110885

### Step 14 (210 minutes ahead)
- **CR**: 46.15%
- **RMSE**: 0.157626
- **MAE**: 0.115456

### Step 15 (225 minutes ahead)
- **CR**: 44.09%
- **RMSE**: 0.162973
- **MAE**: 0.120176

### Step 16 (240 minutes ahead)
- **CR**: 42.26%
- **RMSE**: 0.168236
- **MAE**: 0.124445
