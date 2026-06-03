# Breast Cancer Classification Pipeline

## Overview

This project builds a machine learning pipeline to classify breast cancer patients as **benign** or **malignant** using the UCI Breast Cancer Wisconsin (Original) dataset. The pipeline covers data loading, preprocessing, cross-validation implemented from scratch, hyperparameter tuning, model evaluation, and visualization.

**Dataset:** UCI ML Repository, ID 15 — Breast Cancer Wisconsin (Original)
**Samples:** 699 | **Features:** 9 | **Classes:** 2 (benign / malignant)

---

## Step 1: Load Dataset

```python
breast_cancer = fetch_ucirepo(id=15)
X = breast_cancer.data.features.copy()
y = breast_cancer.data.targets.copy()
```

The dataset is fetched from the UCI ML Repository using the `ucimlrepo` library. It contains 699 patient samples with 9 cytological features measured from fine needle aspirate (FNA) images of breast masses:

| Feature | Description | Range |
|---------|-------------|-------|
| Clump_thickness | Thickness of cell clumps | 1 - 10 |
| Uniformity_of_cell_size | Consistency of cell sizes | 1 - 10 |
| Uniformity_of_cell_shape | Consistency of cell shapes | 1 - 10 |
| Marginal_adhesion | How much cells stick together | 1 - 10 |
| Single_epithelial_cell_size | Size of epithelial cells | 1 - 10 |
| Bare_nuclei | Proportion of bare nuclei | 1 - 10 |
| Bland_chromatin | Chromatin texture uniformity | 1 - 10 |
| Normal_nucleoli | Size/shape of nucleoli | 1 - 10 |
| Mitoses | Level of mitotic activity | 1 - 10 |

The target variable `Class` has two values: **2** (benign) and **4** (malignant).

---

## Step 2: Encode Target Variable

```python
y_encoded = y.iloc[:, 0].map({2: 0, 4: 1})
```

The original target values (2 and 4) are mapped to standard binary labels:
- **0** = Benign (458 samples, 65.5%)
- **1** = Malignant (241 samples, 34.5%)

This encoding is required by scikit-learn classifiers, which expect numeric labels starting from 0. The dataset has a moderate class imbalance (roughly 2:1 benign-to-malignant ratio), which is why we use stratified splitting in subsequent steps to preserve this ratio.

---

## Step 3: Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=14,
    stratify=y_encoded,
)
```

The dataset is split into 80% training (559 samples) and 20% test (140 samples).

**Parameters:**
- `test_size=0.20` — 20% of data reserved for final evaluation. This is a standard ratio for datasets of this size.
- `random_state=14` — Fixed seed for reproducibility. Any integer produces a valid split; the seed simply determines which specific samples land in each set.
- `stratify=y_encoded` — Ensures both sets maintain the same benign/malignant ratio (~65%/35%) as the full dataset. Without stratification, a random split could over- or under-represent one class, especially in the smaller test set.

**Why split before preprocessing?** The test set simulates unseen real-world data. If we preprocessed the full dataset first, statistics from the test set (medians, means, standard deviations) would leak into the training process, leading to overly optimistic performance estimates.

---

## Step 4: Impute Missing Values

```python
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X.columns)
```

The `Bare_nuclei` feature has 16 missing values (2.3% of the column). We fill them using **median imputation**.

**Why median instead of mean?** The median is robust to skewed distributions and outliers. Since `Bare_nuclei` is heavily right-skewed (most values are low, with some extreme high values), the median provides a more representative central value than the mean.

**Preventing data leakage:** The imputer is **fit on the training set only** (`fit_transform`), learning the median from training data. It then **transforms** the test set using that same median. If we computed the median on the full dataset, test-set values would influence the fill value used during training.

---

## Step 5: Feature Scaling

```python
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)
```

StandardScaler transforms each feature to have **mean = 0** and **standard deviation = 1** using the formula:

```
x_scaled = (x - mean) / std
```

**Why scale?** Although all 9 features are already on a 1-10 integer scale, their actual distributions differ significantly. Features like `Mitoses` cluster near 1 while `Clump_thickness` spreads more evenly. Distance-based algorithms (SVM with RBF kernel, KNN) are sensitive to feature magnitudes — a feature with larger variance would dominate the distance calculation. Scaling ensures all features contribute equally.

**Same leakage prevention as imputation:** The scaler learns mean and standard deviation from the training set only, then applies those same parameters to the test set.

---

## Step 6: K-Fold Cross-Validation (from scratch)

This is a custom implementation that follows the principle from the lecture: *"Feature normalisation or dimensionality reduction has to be done independently for each cross-validation run."*

### Why from scratch?

Calling `cross_validate(model, X_train, y_train)` on already-preprocessed data is a subtle form of data leakage. The imputer and scaler were fit on the full training set (which includes the validation fold). Our from-scratch implementation avoids this by re-fitting preprocessing inside each fold.

### Algorithm

**QM_cv,k = (1/k) * sum(QM_i)** for i = 1..k

#### 1. Stratified fold assignment (`stratified_k_fold_indices`)

```python
def stratified_k_fold_indices(y, k, random_state=14):
```

- Groups all sample indices by their class label.
- Shuffles each class independently using a seeded random generator.
- Distributes indices across k folds using round-robin: sample 0 goes to fold 0, sample 1 to fold 1, ..., sample k to fold 0 again, etc.
- This guarantees each fold has approximately the same class ratio as the full dataset.
- Finally shuffles within each fold so samples are not ordered by class.

#### 2. Per-fold train-validate loop (`kfold_cross_validate`)

```python
def kfold_cross_validate(X, y, model_fn, k=5, random_state=14):
```

For each fold i = 1..5:

| Sub-step | Code | Purpose |
|----------|------|---------|
| **a) Split** | `val_idx = folds[i]`, `train_idx = concat(others)` | Fold i becomes the validation set; all other folds become training data |
| **b) Impute** | `fold_imputer.fit_transform(X_fold_train)` then `.transform(X_fold_val)` | Fill missing values using only this fold's training median |
| **c) Scale** | `fold_scaler.fit_transform(X_fold_train)` then `.transform(X_fold_val)` | Normalize using only this fold's training mean/std |
| **d) Train** | `model = model_fn()` then `model.fit(...)` | Create a fresh model instance (no state from previous folds) and train it |
| **e) Evaluate** | `accuracy_score(...)`, `precision_score(...)`, etc. | Compute quality measures on the validation predictions |

The `model_fn` parameter is a factory function (lambda) that returns a new, unfitted model each time. This ensures no information carries over between folds.

#### 3. Aggregate results

After all k folds complete, each metric has k values. The final reported score is the **mean +/- standard deviation** across folds.

### Cross-Validation Results (5 folds)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.9671 +/- 0.0098 | 0.9564 +/- 0.0285 | 0.9501 +/- 0.0504 | 0.9516 +/- 0.0164 | 0.9958 +/- 0.0034 |
| SVM (RBF kernel) | 0.9657 +/- 0.0083 | 0.9353 +/- 0.0400 | 0.9708 +/- 0.0386 | 0.9512 +/- 0.0116 | 0.9913 +/- 0.0095 |
| Random Forest | 0.9599 +/- 0.0086 | 0.9413 +/- 0.0377 | 0.9460 +/- 0.0468 | 0.9419 +/- 0.0137 | 0.9928 +/- 0.0045 |

These are unbiased estimates because no validation data ever influenced the preprocessing it was evaluated with.

---

## Step 7: Hyperparameter Tuning (GridSearchCV)

```python
grid = GridSearchCV(
    estimator=cfg["model"],
    param_grid=cfg["params"],
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)
```

GridSearchCV performs an **exhaustive search** over all specified parameter combinations. For each combination, it runs 5-fold cross-validation on the training set and selects the combination with the best mean F1 score.

**Parameters:**
- `cv=5` — 5-fold cross-validation for each parameter combination.
- `scoring="f1"` — Optimizes for F1 score, which balances precision and recall. In a medical context, this is preferable to accuracy alone because it penalizes both false positives (unnecessary biopsies) and false negatives (missed cancers).
- `n_jobs=-1` — Uses all CPU cores for parallel computation.
- `refit=True` — After finding the best parameters, automatically retrains the model on the full training set using those parameters.

### Parameter grids and results

#### Logistic Regression

| Parameter | Values tried | Best |
|-----------|-------------|------|
| `C` | 0.01, 0.1, 1.0, 10.0, 100.0 | **0.1** |
| `penalty` | l2 | l2 |

`C` is the inverse regularization strength. A smaller value (0.1) means stronger regularization, which simplifies the model and reduces overfitting. The L2 penalty shrinks coefficients toward zero without eliminating them.

**Best CV F1: 0.9497**

#### SVM (RBF kernel)

| Parameter | Values tried | Best |
|-----------|-------------|------|
| `C` | 0.1, 1.0, 10.0, 100.0 | **1.0** |
| `gamma` | scale, auto, 0.01, 0.1 | **0.01** |

`C` controls the trade-off between a smooth decision boundary and correctly classifying training points. `gamma` defines how far the influence of a single training point reaches — a smaller gamma (0.01) means each point influences a wider area, creating a smoother boundary.

**Best CV F1: 0.9535**

#### Random Forest

| Parameter | Values tried | Best |
|-----------|-------------|------|
| `n_estimators` | 50, 100, 200 | **200** |
| `max_depth` | None, 5, 10, 20 | **5** |
| `min_samples_split` | 2, 5, 10 | **2** |

More trees (200) give more stable predictions. A shallow depth (5) prevents individual trees from memorizing the training data. Together, these settings create an ensemble that generalizes well.

**Best CV F1: 0.9592**

---

## Step 8: Test Set Evaluation

The tuned models are evaluated on the held-out test set (140 samples) that was never used during training or tuning.

### Logistic Regression

```
              precision    recall  f1-score   support
      Benign       0.95      0.99      0.97        92
   Malignant       0.98      0.90      0.93        48
    accuracy                           0.96       140
```

5 false negatives (missed malignant cases), 1 false positive.

### SVM (RBF kernel)

```
              precision    recall  f1-score   support
      Benign       0.98      0.99      0.98        92
   Malignant       0.98      0.96      0.97        48
    accuracy                           0.98       140
```

2 false negatives, 1 false positive. Best balance between precision and recall.

### Random Forest

```
              precision    recall  f1-score   support
      Benign       0.98      0.99      0.98        92
   Malignant       0.98      0.96      0.97        48
    accuracy                           0.98       140
```

2 false negatives, 1 false positive. Identical to SVM on this test split.

---

## Step 9: Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9571 | 0.9773 | 0.8958 | 0.9348 | 0.9968 |
| SVM (RBF kernel) | 0.9786 | 0.9787 | 0.9583 | 0.9684 | 0.9968 |
| Random Forest | 0.9786 | 0.9787 | 0.9583 | 0.9684 | 0.9966 |

### Metric definitions

- **Accuracy** = (TP + TN) / total — overall correctness, but can be misleading with class imbalance.
- **Precision** = TP / (TP + FP) — of all patients predicted malignant, how many actually are. High precision means fewer unnecessary biopsies.
- **Recall (Sensitivity)** = TP / (TP + FN) — of all actual malignant patients, how many were caught. High recall means fewer missed cancers.
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall) — harmonic mean balancing precision and recall.
- **AUC-ROC** = Area Under the ROC Curve — measures the model's ability to distinguish between classes across all probability thresholds. A value of 1.0 is perfect; 0.5 is random guessing.

### Key takeaways

- **SVM and Random Forest** tie at 97.86% accuracy and 0.9684 F1 on this test split. Both miss only 2 malignant cases out of 48.
- **Logistic Regression** is slightly behind at 95.71% accuracy, with 5 missed malignant cases. However, it achieves the highest AUC-ROC (0.9968), indicating its probability estimates are well-calibrated.
- In a **clinical setting**, recall is the most critical metric — a false negative (missed cancer) has far worse consequences than a false positive (extra follow-up test). SVM and Random Forest both achieve 95.83% recall.

---

## Step 10: Visualizations

Two figures are generated and saved:

### `model_comparison.png`

Contains four panels:
1. **ROC Curves** — Plots True Positive Rate vs. False Positive Rate for all three models. All curves hug the top-left corner (near-perfect classification). The diagonal dashed line represents a random classifier (AUC = 0.5).
2. **Confusion Matrix — Logistic Regression** — Shows TN=91, FP=1, FN=5, TP=43.
3. **Confusion Matrix — SVM** — Shows TN=91, FP=1, FN=2, TP=46.
4. **Confusion Matrix — Random Forest** — Shows TN=91, FP=1, FN=2, TP=46.

### `feature_importance.png`

Contains two panels comparing which features matter most:

1. **Logistic Regression — Absolute Coefficients** — The magnitude of each feature's weight in the linear decision function. Larger values mean the feature has more influence on the prediction. Top features: `Bare_nuclei`, `Clump_thickness`, `Bland_chromatin`.

2. **Random Forest — Gini Importance** — Measures how much each feature reduces impurity (Gini index) across all decision trees. Features used in more splits and closer to the root have higher importance. Top features: `Uniformity_of_cell_size`, `Uniformity_of_cell_shape`, `Bare_nuclei`.

Both models agree that `Bare_nuclei` is a strong predictor, which aligns with medical knowledge — bare nuclei are a cytological hallmark of malignancy. The models rank other features differently because they capture different types of relationships (linear vs. nonlinear).

---

## Data Leakage Prevention Summary

Throughout the pipeline, we follow the principle: *"Train and test data must have no unnecessary connection to each other."*

| Where | How leakage is prevented |
|-------|--------------------------|
| **Train/Test split** | Split happens before any preprocessing |
| **Imputation** | Median computed from training set only; applied to test set |
| **Scaling** | Mean/std computed from training set only; applied to test set |
| **Cross-validation** | From-scratch implementation re-fits imputer and scaler inside each fold using only that fold's training portion |
| **Hyperparameter tuning** | GridSearchCV runs its own internal CV on the training set; the test set is never seen during tuning |

---

## Improvement 1: Expanded Logistic Regression with L1 Penalty

### Problem identified

Logistic Regression was the weakest model in the pipeline, with a recall of just 89.58% (5 missed malignant cases out of 48). The original configuration only used L2 regularization (`penalty="l2"`) with the `lbfgs` solver, which limited the model's ability to perform feature selection.

### What changed

A second Logistic Regression variant was added using L1 regularization:

```python
"Logistic Regression (L1)": LogisticRegression(
    solver="liblinear", penalty="l1", max_iter=1000,
    class_weight="balanced", random_state=14)
```

The hyperparameter grid was also expanded to search over both penalties:

| Parameter | L2 grid | L1 grid |
|-----------|---------|---------|
| `C` | 0.01, 0.1, 1.0, 10.0, 100.0 | 0.01, 0.1, 1.0, 10.0, 100.0 |
| `penalty` | l2 | l1 |
| `class_weight` | None, balanced | None, balanced |

### Why L1 (Lasso) regularization

L1 and L2 regularization both add a penalty term to the loss function to prevent overfitting, but they work differently:

- **L2 (Ridge)** adds the sum of squared coefficients to the loss. This shrinks all coefficients toward zero proportionally but never eliminates any. It uses the `lbfgs` solver (a quasi-Newton optimization method).

- **L1 (Lasso)** adds the sum of absolute coefficients to the loss. This can shrink coefficients all the way to exactly zero, effectively removing those features from the model. It requires the `liblinear` solver (a coordinate descent method that handles the non-differentiable L1 penalty).

L1 is valuable because it performs **embedded feature selection** — if some of the 9 features are noise or redundant, L1 will zero them out automatically. This can produce a simpler, more interpretable model.

### Why `liblinear` solver

The `lbfgs` solver (used for L2) does not support L1 regularization because L1's absolute value function is not differentiable at zero, which breaks gradient-based optimization. The `liblinear` solver uses coordinate descent, which handles L1 natively by optimizing one coefficient at a time.

### Results

GridSearchCV selected `C=0.1` with `class_weight="balanced"` for L1, applying stronger regularization than L2's best (`C=1.0`). Despite this, the L1 model achieved identical test performance to L2:

| Variant | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---------|----------|-----------|--------|-----|---------|
| LR (L2) | 0.9714 | 0.9783 | 0.9375 | 0.9574 | 0.9964 |
| LR (L1) | 0.9714 | 0.9783 | 0.9375 | 0.9574 | 0.9966 |

The feature importance plot shows that L1 kept all 9 features non-zero, confirming that every cytological feature in this dataset carries meaningful predictive signal. The coefficient magnitudes are more compressed under L1 (due to the lower `C`), but the ranking remains similar: `Bare_nuclei` and `Clump_thickness` remain the strongest predictors.

---

## Improvement 2: Class Weight Balancing

### Problem identified

The dataset has a moderate class imbalance: 458 benign samples (65.5%) vs. 241 malignant samples (34.5%). By default, scikit-learn classifiers treat all samples equally during training, which means the model optimizes more for the majority class (benign). This was evident in Logistic Regression's low recall — the model was biased toward predicting benign, causing it to miss malignant cases.

In a clinical context, this bias is dangerous. A false negative (missed cancer) has far worse consequences than a false positive (an unnecessary follow-up biopsy).

### What changed

The `class_weight="balanced"` parameter was added to all three classifiers and included in the hyperparameter search grid:

```python
LogisticRegression(..., class_weight="balanced")
SVC(..., class_weight="balanced")
RandomForestClassifier(..., class_weight="balanced")
```

For Random Forest, a third option `"balanced_subsample"` was also searched. This recomputes weights for each tree based on the bootstrap sample rather than the full dataset.

### How `class_weight="balanced"` works

When set to `"balanced"`, scikit-learn automatically adjusts the weight of each class inversely proportional to its frequency:

```
weight_class = n_samples / (n_classes * n_samples_in_class)
```

For this dataset:
- **Benign weight** = 699 / (2 * 458) = **0.76**
- **Malignant weight** = 699 / (2 * 241) = **1.45**

This means misclassifying a malignant sample incurs approximately 1.9x the penalty of misclassifying a benign sample. The effect on each model:

- **Logistic Regression**: The weighted samples shift the decision boundary. Malignant errors contribute more to the log-loss, so the optimizer finds a boundary that catches more malignant cases (higher recall) even if it means a few more false positives.

- **SVM**: The `C` parameter (misclassification penalty) is scaled per class. Effectively, `C_malignant = C * 1.45` and `C_benign = C * 0.76`. The support vectors shift to create a wider margin on the malignant side.

- **Random Forest**: At each tree split, the Gini impurity calculation weights samples by their class weight. Splits that correctly separate malignant samples are rewarded more, so the trees prioritize getting malignant classifications right.

### Results

GridSearchCV selected `class_weight="balanced"` as optimal for **every model**, confirming that addressing the imbalance helps universally.

#### Logistic Regression — before vs. after balancing

| Metric | Before (no balancing) | After (balanced) | Change |
|--------|----------------------|------------------|--------|
| Accuracy | 0.9571 | **0.9714** | +1.4% |
| Precision | 0.9773 | 0.9783 | +0.1% |
| Recall | 0.8958 | **0.9375** | **+4.2%** |
| F1 | 0.9348 | **0.9574** | +2.3% |
| False negatives | 5 | **3** | -2 missed cancers |

The most significant improvement: recall jumped from 89.58% to 93.75%, reducing missed malignant cases from 5 to 3.

#### Random Forest — recall champion

With balanced weights, Random Forest achieved the highest recall of all models:

| Metric | Value |
|--------|-------|
| Recall | **0.9792** (only 1 missed cancer out of 48) |
| Precision | 0.9400 (3 false positives) |
| F1 | 0.9592 |

This makes Random Forest the most clinically conservative model — it flags nearly every malignant case at the cost of slightly more false alarms.

#### Updated model comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression (L2) | 0.9714 | 0.9783 | 0.9375 | 0.9574 | 0.9964 |
| Logistic Regression (L1) | 0.9714 | 0.9783 | 0.9375 | 0.9574 | 0.9966 |
| SVM (RBF kernel) | 0.9786 | 0.9787 | 0.9583 | 0.9684 | 0.9968 |
| Random Forest | 0.9714 | 0.9400 | **0.9792** | 0.9592 | 0.9966 |

### Updated visualizations

The `model_comparison.png` figure now includes confusion matrices for all four models (LR L2, LR L1, SVM, Random Forest) and the ROC curve panel shows four overlapping curves with AUC values above 0.996 for all.

The `feature_importance.png` figure now has three panels:
1. **LR (L2) coefficients** — all 9 features non-zero, `Bare_nuclei` dominant
2. **LR (L1) coefficients** — all 9 features non-zero (red bars indicate zeroed-out features; none were eliminated), confirming all features are informative
3. **Random Forest Gini importance** — `Uniformity_of_cell_size` dominant, with a more spread distribution across features compared to LR

### Key takeaway

Class weight balancing is a simple, zero-cost improvement for imbalanced medical datasets. It requires no additional data, no new features, and no architectural changes — just a single parameter. The trade-off it makes (slightly more false positives in exchange for fewer false negatives) aligns perfectly with the clinical priority of catching every cancer case.
