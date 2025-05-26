# 电力系统稳定性分类 (Power System Stability Classification)

本项目使用机器学习方法对电力系统稳定性进行分类预测，实现了基于CatBoost、LightGBM和XGBoost的多模型比较分析。


## 快速开始

### 1. 克隆仓库

首先克隆项目仓库到本地：

```bash
git clone https://github.com/Jinming00/EE.git
cd EE/power_system_stability_classification
```

### 2. 环境安装

推荐使用conda创建环境，一键安装所有依赖：

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate ee
```

或者手动安装主要依赖：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install catboost lightgbm xgboost
pip install optuna shap
pip install jupyter
```

### 3. 查看最佳结果

如果你只想查看训练好的最佳模型效果，直接运行：

```bash
best_model_inference.ipynb
```

最佳模型保存在：
- **模型路径**: `catboost/best_model/best.cbm`
- **模型类型**: CatBoost分类器


### 4. 从头开始训练

如果想从头训练所有模型并进行完整的比较分析，参考：

```bash
train.ipynb
```

按照notebook中的说明，逐个单元格运行即可。训练流程包括：

1. **数据加载与预处理**
2. **特征工程**
3. **CatBoost模型训练与优化**
4. **LightGBM模型训练与优化**
5. **XGBoost模型训练与优化**
6. **模型性能比较**
7. **SHAP可解释性分析**

## 模型性能

基于测试集的CatBoost最佳模型性能表现：

| 类别 | 精确率 | 召回率 | F1分数 | 准确率 | 加权F1分数 | ROC-AUC |
|------|--------|--------|--------|--------|-----------|---------|
| 稳定 | 0.9781 | 0.9936 | 0.9754 | - | - | - |
| 不稳定 | 0.9889 | 0.9623 | 0.9858 | - | - | - |
| **整体** | **-** | **-** | **-** | **0.982** | **0.9819** | **0.9976** |







