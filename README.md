# E-Commerce Fraud Detection

A comprehensive, end-to-end machine learning pipeline for detecting fraudulent transactions in e-commerce. This project systematically walks through every stage of the data science workflow — from raw data ingestion to hyperparameter-optimised ensemble models — with thorough analysis of every design decision.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Key Findings](#key-findings)
- [Authors](#authors)

---

## Overview

E-commerce fraud causes billions of dollars in losses annually. This project builds a complete fraud detection system using the publicly available E-Commerce Fraud Detection Dataset (299,695 transactions). Rather than jumping straight to a final model, every stage is explored and justified:

- Why standard accuracy metrics fail on imbalanced fraud data
- How feature engineering and selection affect model performance
- Why a from-scratch neural network failed despite showing healthy training loss
- How hyperparameter tuning changes precision-recall trade-offs
- Why Random Forest outperforms all linear models on this task

**Best model:** Random Forest tuned with RandomizedSearchCV
- Accuracy: **99.27%**
- F1 Score: **0.7768**
- AUC-ROC: **0.9464**
- Precision: **0.9191**
- Recall: **0.6726**

---

## Dataset

**Source:** [Kaggle — E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)

**Downloaded using:**
```python
import kagglehub
path = kagglehub.dataset_download("umuttuygurr/e-commerce-fraud-detection-dataset")
```

**Shape:** 299,695 rows × 17 columns

**Class distribution:** 98.15% legitimate / 1.85% fraud (severely imbalanced)

**Features:**

| Feature | Type | Description |
|---|---|---|
| `transaction_id` | int | Unique transaction identifier (dropped) |
| `user_id` | int | User identifier (dropped) |
| `account_age_days` | int | Days since account creation |
| `total_transactions_user` | int | User's total historical transactions |
| `avg_amount_user` | float | User's historical average transaction value |
| `amount` | float | Current transaction amount |
| `country` | object | Transaction origin country |
| `bin_country` | object | Card-issuing country (from BIN) |
| `channel` | object | Payment channel (web / app) |
| `merchant_category` | object | Merchant business sector |
| `promo_used` | int | Whether a promo code was used |
| `avs_match` | int | Address Verification System result (0/1) |
| `cvv_result` | int | CVV code match result (0/1) |
| `three_ds_flag` | int | 3-D Secure authentication (0/1) |
| `shipping_distance_km` | float | Shipping destination distance |
| `is_fraud` | int | **Target variable** (0 = legitimate, 1 = fraud) |
| `transaction_time` | object | Transaction timestamp (decomposed) |

---

## Project Structure

```
ecommerce-fraud-detection/
│
├── E_commerce_Fraud_Detection.ipynb   # Main notebook — full pipeline
│
├── README.md                          # This file
│
└── Report
```

---

## Pipeline Stages

### 1. Data Preprocessing

- Dropped identifier columns (`transaction_id`, `user_id`)
- Parsed `transaction_time` → extracted `hour`, `day`, `month`, `weekday`
- IQR-based **outlier capping** (Winsorisation) on 5 continuous features
  - Capping chosen over deletion to preserve extreme fraud cases
  - Binary feature artefacts (avs_match, cvv_result flagged as outliers due to IQR=0) correctly ignored

```python
# Outlier capping
for feature in continuous_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    df[feature] = df[feature].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```

---

### 2. Exploratory Data Analysis

Key fraud signals discovered through EDA:

| Signal | Finding |
|---|---|
| Hour of day | Non-uniform fraud rate — late night / early morning elevated |
| Merchant category | Travel sector has highest fraud rate |
| Country mismatch | Mismatched transactions have ~5× higher fraud rate |
| Shipping distance | Fraudulent transactions have higher median shipping distance |
| Amount | Fraud shows heavier right tail but distributions overlap |

**Engineered feature during EDA:**
```python
df['country_mismatch'] = (df['country'] != df['bin_country']).astype(int)
```

---

### 3. Feature Scaling & KNN Analysis

Three scalers evaluated:

| Scaler | Formula | Best For |
|---|---|---|
| MaxAbsScaler | x / max(\|x\|) | KNN distance analysis |
| MinMaxScaler | (x - min) / (max - min) | Bounded [0,1] inputs |
| StandardScaler | (x - μ) / σ | Gradient-based models (LR, SGD) |

StandardScaler improved Logistic Regression:
```
Before scaling:  F1 = 0.3933,  AUC-ROC = 0.9176
After scaling:   F1 = 0.4098,  AUC-ROC = 0.9212
```

> ⚠️ **Data leakage prevention:** Scalers always fitted on `X_train` only, then `.transform()` applied to `X_test`.

---

### 4. Feature Selection

Five methods compared, evaluated by MSE on test set using RandomForestRegressor:

```
Method                 MSE
──────────────────────────
RFE (LinearReg)      0.01524   ← worst
Lasso selected       0.00955
Univariate           0.00858
Tree importance      0.00725
Correlation          0.00700
Full feature set     0.00679   ← best
```

Top features consistently across methods:
- **Negative correlation with fraud:** `avs_match` (-0.199), `cvv_result` (-0.192), `three_ds_flag` (-0.142), `account_age_days` (-0.126)
- **Positive correlation with fraud:** `shipping_distance_km` (+0.152), `country_mismatch` (+0.131)

`avg_amount_user` dropped due to high collinearity with `amount` and near-zero unique signal.

---

### 5. Anomaly Detection (Local Outlier Factor)

```python
LocalOutlierFactor(
    n_neighbors=20,
    contamination=df['is_fraud'].mean(),  # ~1.85%
    novelty=True
)
```

- Applied on 6 most fraud-informative features
- Detected and removed **5,877 anomalies**
- Dataset reduced: 299,695 → 293,818 rows
- Class balance unchanged (98.15% / 1.85%)

---

### 6. Class Imbalance Handling

Original ratio: **44:1** (legitimate:fraud)

Three strategies implemented:

| Strategy | Fraud After | Ratio | Method |
|---|---|---|---|
| Upsampling (repeat=4) | 21,160 | 11:1 | Duplicate minority class 4× |
| Downsampling (frac=0.3) | 5,290 | 13:1 | Keep 30% of majority class |
| **SMOTE** ✓ | **234,466** | **1:1** | Synthetic interpolation |

**SMOTE formula:**
$$x_{\text{new}} = x_i + \lambda \cdot (x_j - x_i), \quad \lambda \sim U[0,1]$$

SMOTE selected for final pipeline — perfect 1:1 balance, no data loss, no exact duplication.

---

### 7. Categorical Encoding & Domain Feature Engineering

**Encoding:**
```python
df = pd.get_dummies(df, drop_first=True,
                    columns=['country', 'bin_country', 'channel', 'merchant_category'])
```

**4 Domain-engineered features:**

```python
# Composite security score (0 = all checks failed, 3 = all passed)
df['security_score'] = df['avs_match'] + df['cvv_result'] + df['three_ds_flag']

# Geographic risk — non-zero only when both mismatch AND long distance
df['geo_risk'] = df['country_mismatch'] * df['shipping_distance_km']

# User trustworthiness — older account × more transactions = safer
df['user_activity_score'] = df['account_age_days'] * df['total_transactions_user']

# Night transaction flag
df['night_transaction'] = df['hour'].apply(lambda x: 1 if x < 6 else 0)
```

---

### 8. Logistic Regression

```python
LogisticRegression(class_weight='balanced')
```

| Threshold | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| τ = 0.5 | 0.1174 | 0.8327 | 0.0639 | high |
| **τ = 0.9** | **0.3437** | **0.8327** | **0.4410** | 0.28 |

Confusion matrix at τ = 0.9:
```
              Predicted 0    Predicted 1
Actual 0        57,251           398
Actual 1           801           314
```

AUC-ROC unchanged between thresholds (threshold-independent metric). Raising τ from 0.5 to 0.9 tripled F1 by eliminating false alarms.

---

### 9. Overfitting Analysis (Decision Tree)

Unconstrained tree (max_depth=None):
```
Train Accuracy: 1.0000    Train Loss: 2.2e-16
Test Accuracy:  0.9865    Test Loss:  0.4882
```

Trained at depths 1-20 to visualise bias-variance trade-off:
- **Shallow depths:** High bias — both train and test perform poorly
- **Intermediate depths (~7-10):** Test accuracy peaks
- **Deep depths:** Train → 1.0, test diverges — overfitting

---

### 10. Regularisation (SGDClassifier on Polynomial Features)

Polynomial expansion: ~37 features → **702 features** (degree=2)
Trained for 100 epochs with 10-epoch rolling average smoothing.

| Strategy | α | Train Loss | Val Loss | Gap | Verdict |
|---|---|---|---|---|---|
| **L1 (Lasso)** | 0.05 | 0.0927 | 0.0940 | **0.0013** | ✅ Best |
| L2 (Ridge) | 0.01 | 0.0523 | 0.1336 | 0.0813 | ❌ Overfits |
| Elastic Net | 0.01 | 0.0576 | 0.1118 | 0.0542 | ⚠️ Intermediate |
| **Early Stopping** | 0.10 | 0.0927 | 0.0940 | **0.0013** | ✅ Best |

Early stopping triggered at **epoch 23** (best epoch: **3**), patience=20.

> **Key insight:** Lower training loss ≠ better model. L2 had the lowest training loss (0.0523) but the highest overfitting gap (0.0813). L1 had higher training loss but near-zero generalisation gap.

---

### 11. Neural Network (From Scratch — NumPy Only)

**Architecture:**
```
702 → [Dense 64, ReLU] → [Dense 32, ReLU] → [Dense 1, Sigmoid]
```

**Implementation details:**
- He (Kaiming) initialisation: $W \sim \mathcal{N}(0, \sqrt{2/n^{[l-1]}})$
- Binary cross-entropy loss with ε=1e-9 for numerical stability
- Full backpropagation implemented from scratch
- Vanilla gradient descent, learning rate = 0.01

**Cross-framework validation:**
```
NumPy:       output shape (80000, 1)  ✓
TensorFlow:  output shape (80000, 1)  ✓
PyTorch:     output shape (80000, 1)  ✓
```

**Training convergence (Run 1):**
```
Epoch  0: Cost = 0.5024
Epoch 10: Cost = 0.3181
Epoch 20: Cost = 0.2388
Epoch 50: Cost = 0.1548
Epoch 90: Cost = 0.1228
```

**Test set evaluation:**
```
Accuracy:  0.9810   ← misleading due to class imbalance
F1:        0.0000   ← model predicted zero fraud cases
AUC-ROC:   0.5154   ← barely above random
Precision: 0.0000
Recall:    0.0000
```

**Why it failed — 5 reasons:**

1. **Class imbalance without loss weighting** — 44× more legitimate examples dominated gradients
2. **Imbalanced training subsample** — only ~1,500 fraud cases in 80,000 rows (SMOTE not used)
3. **Threshold collapse** — model learned to output ~0.02 for all inputs; all below 0.5 threshold
4. **Full-batch gradient descent** — no stochastic noise to escape the all-negative attractor
5. **No validation monitoring** — no mechanism to detect or respond to distribution collapse

---

### 12. Hyperparameter Tuning (Random Forest)

#### RandomizedSearchCV
```python
param_dist = {
    'n_estimators':      randint(30, 100),
    'max_depth':         [10, 20, None],
    'min_samples_split': randint(2, 6)
}
RandomizedSearchCV(RF, n_iter=5, cv=3, scoring='f1')
```
**Best:** `{max_depth: None, min_samples_split: 3, n_estimators: 82}` — CV F1: 0.7583

#### Optuna (Bayesian TPE)
```python
def objective(trial):
    n_estimators      = trial.suggest_int('n_estimators', 30, 100)
    max_depth         = trial.suggest_int('max_depth', 10, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 6)
    ...
study.optimize(objective, n_trials=10)
```

**All 10 Optuna trials:**

| Trial | n_est | max_depth | min_split | CV F1 |
|---|---|---|---|---|
| **0** | **53** | **13** | **3** | **0.7592** |
| 1 | 54 | 16 | 5 | 0.7549 |
| 2 | 71 | 29 | 2 | 0.7579 |
| 3 | 99 | 12 | 4 | 0.7586 |
| 4 | 42 | 18 | 4 | 0.7574 |
| 5 | 60 | 25 | 2 | 0.7571 |
| 6 | 62 | 21 | 4 | 0.7567 |
| 7 | 90 | 12 | 4 | 0.7582 |
| 8 | 31 | 24 | 4 | 0.7570 |
| 9 | 96 | 12 | 4 | 0.7587 |

---

## Results

### Final Model Comparison

| Model | Accuracy | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| LR (unscaled) | — | 0.393 | 0.918 | — | — |
| LR (StandardScaler) | — | 0.410 | 0.921 | — | — |
| LR (τ=0.5, balanced) | 0.98 | 0.117 | 0.833 | 0.064 | — |
| LR (τ=0.9, balanced) | 0.98 | 0.344 | 0.833 | 0.441 | 0.28 |
| Decision Tree (unconstrained) | 0.987 | — | — | — | — |
| SGD + L1 (α=0.05) | 0.981 | — | — | — | — |
| SGD + L2 (α=0.01) | 0.976 | — | — | — | — |
| SGD + Elastic Net | 0.979 | — | — | — | — |
| SGD + Early Stopping | 0.981 | — | — | — | — |
| NumPy Neural Network | 0.981 | 0.000 | 0.515 | 0.000 | 0.000 |
| **RF (RandomizedSearchCV)** | **0.9927** | **0.7768** | 0.9464 | **0.9191** | **0.6726** |
| RF (Optuna) | 0.9925 | 0.7714 | **0.9620** | 0.9140 | 0.6673 |

### Best Model Confusion Matrix (RandomizedSearchCV RF)
```
                  Predicted 0    Predicted 1
Actual 0 (legit)    57,583           66
Actual 1 (fraud)       365          750
```

### Metric Interpretation

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | 99.27% | Misleading due to imbalance — not the key metric |
| F1 | 0.7768 | Harmonic mean of precision and recall |
| AUC-ROC | 0.9464 | 94.6% chance model ranks a random fraud higher than a random legitimate |
| Precision | 0.919 | 9 out of 10 fraud alerts are genuine |
| Recall | 0.673 | Catches 67.3% of all fraud; misses 32.7% |

---

## Installation

### Requirements

```bash
Python >= 3.8
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn optuna tensorflow torch kagglehub
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
optuna>=3.0.0
tensorflow>=2.12.0
torch>=2.0.0
kagglehub>=0.1.0
scipy>=1.10.0
```

---

## Usage

### Run the Full Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook E_commerce_Fraud_Detection.ipynb
```

Or in Google Colab — upload the notebook and run all cells sequentially.

### Dataset Download (automatic)

```python
import kagglehub
path = kagglehub.dataset_download("umuttuygurr/e-commerce-fraud-detection-dataset")
file_path = path + "/transactions.csv"
df = pd.read_csv(file_path)
```

### Quick Start — Run Just the Best Model

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Load data
df = pd.read_csv("transactions.csv")

# Basic preprocessing
df.drop(['transaction_id', 'user_id'], axis=1, inplace=True)
df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['hour'] = df['transaction_time'].dt.hour
df['day'] = df['transaction_time'].dt.day
df['month'] = df['transaction_time'].dt.month
df['weekday'] = df['transaction_time'].dt.weekday
df.drop('transaction_time', axis=1, inplace=True)
df['country_mismatch'] = (df['country'] != df['bin_country']).astype(int)
df.drop('avg_amount_user', axis=1, inplace=True)

# Encode categoricals
df = pd.get_dummies(df, drop_first=True,
                    columns=['country', 'bin_country', 'channel', 'merchant_category'])

# Domain features
df['security_score'] = df['avs_match'] + df['cvv_result'] + df['three_ds_flag']
df['geo_risk'] = df['country_mismatch'] * df['shipping_distance_km']
df['user_activity_score'] = df['account_age_days'] * df['total_transactions_user']
df['night_transaction'] = df['hour'].apply(lambda x: 1 if x < 6 else 0)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Split
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Best model
rf = RandomForestClassifier(
    n_estimators=82,
    max_depth=None,
    min_samples_split=3,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_smote, y_train_smote)

# Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Neural network from scratch, array operations |
| `pandas` | ≥1.5 | Data loading, manipulation, feature engineering |
| `matplotlib` | ≥3.6 | All visualisations and training curve plots |
| `seaborn` | ≥0.12 | Statistical visualisations, heatmaps, confusion matrices |
| `scikit-learn` | ≥1.2 | Preprocessing, models, metrics, cross-validation |
| `imbalanced-learn` | ≥0.10 | SMOTE implementation |
| `optuna` | ≥3.0 | Bayesian hyperparameter optimisation |
| `tensorflow` | ≥2.12 | Neural network validation (TF/Keras) |
| `torch` | ≥2.0 | Neural network validation (PyTorch) |
| `kagglehub` | ≥0.1 | Dataset download |
| `scipy` | ≥1.10 | Statistical distributions for RandomizedSearchCV |

---

## Key Findings

### 1. Accuracy is Useless for Fraud Detection
A model predicting "not fraud" for every transaction achieves 98.15% accuracy while catching zero fraud cases. F1, AUC-ROC, precision, and recall are the meaningful metrics.

### 2. The Neural Network Failed Despite Healthy Training Loss
Training loss converged from 0.50 to 0.12 over 100 epochs — looks healthy. But the model achieved F1 = 0.000. The loss improved because the model learned to output ~0.02 (the fraud base rate) for every input, not because it learned fraud patterns. Root cause: imbalanced training subsample without class-weighted loss.

### 3. Threshold Matters as Much as the Model
Logistic Regression at τ=0.5 achieved F1=0.117. The same model at τ=0.9 achieved F1=0.344 — a 3× improvement with zero model changes. AUC-ROC stayed identical (0.8327) because it is threshold-independent.

### 4. L1 Outperformed L2 in Regularisation (at these alpha values)
On 702-dimensional polynomial features, L1 (α=0.05) achieved a train-validation gap of 0.0013. L2 (α=0.01) achieved a gap of 0.0813. The difference is partly the penalty type (sparsity) and partly the alpha value (stronger penalty). The key lesson: the choice of α matters as much as the choice of penalty.

### 5. Random Forest Dominates All Linear Models
RF (F1=0.7768) vs best LR (F1=0.410) — the gap exists because fraud requires compound conditions ("suspicious when country mismatches AND CVV fails AND account is new"). LR can only sum features linearly; trees capture these compound IF-THEN rules naturally. Ensemble averaging of 82 trees further eliminates the overfitting that kills individual decision trees.

### 6. RandomizedSearchCV vs Optuna — Different Strengths
RandomizedSearchCV (max_depth=None) wins on F1 at threshold 0.5. Optuna (max_depth=13) wins on AUC-ROC. Unbounded trees make more aggressive predictions → higher recall at fixed threshold → higher F1. Bounded trees are better calibrated → better probability ranking → higher AUC-ROC.

---

## Authors

| Name | Role |
|---|---|
| Samin Osman | Pipeline development , Feature engineering |
| Robiul Hassan | regularisation analysis , Neural network implementation|
| Walid Talal | Hyperparameter tuning,  Model Evaluation|

---

## License

This project is for academic purposes. The dataset is publicly available on Kaggle under its respective license.
