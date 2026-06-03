from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

breast_cancer = fetch_ucirepo(id=15)

X = breast_cancer.data.features.copy()
y = breast_cancer.data.targets.copy()

y_encoded = y.iloc[:, 0].map({2: 0, 4: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=14,
    stratify=y_encoded,
)

# IMPUTE MISSING VALUES
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X.columns)

# FEATURE SCALING
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)


# Algorithm:
#   1. Shuffle the dataset indices, then split into k roughly equal groups.
#   2. For each fold i = 1..k:
#        - Validation set = group i,  Training set = all other groups
#        - Fit imputer on training set → transform both sets
#        - Fit scaler  on training set → transform both sets
#        - Fit model   on training set → predict on validation set
#        - Compute quality measures on validation predictions
#   3. Final quality measure = (1/k) * Σ QM_i
#
# We use stratified splitting to preserve the class ratio in each fold.

def stratified_k_fold_indices(y, k, random_state=14):
    """Split indices into k stratified folds (preserving class ratio)."""
    rng = np.random.RandomState(random_state)
    indices_by_class = {}
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        indices_by_class[cls] = cls_indices

    # Distribute each class's indices across k folds round-robin
    folds = [[] for _ in range(k)]
    for cls, indices in indices_by_class.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    # shuffle within each fold so samples aren't ordered by class
    for fold in folds:
        rng.shuffle(fold)

    return [np.array(fold) for fold in folds]


def kfold_cross_validate(X, y, model_fn, k=5, random_state=14):
    X_arr = np.array(X)
    y_arr = np.array(y)
    folds = stratified_k_fold_indices(y_arr, k, random_state)

    fold_scores = {m: [] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}

    for i in range(k):
        # --- a) Split into train / validation for this fold ---
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
        y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

        # --- b) Imputation — fit on this fold's training data only ---
        fold_imputer = SimpleImputer(strategy="median")
        X_fold_train = fold_imputer.fit_transform(X_fold_train)
        X_fold_val   = fold_imputer.transform(X_fold_val)

        # --- c) Scaling — fit on this fold's training data only ---
        fold_scaler = StandardScaler()
        X_fold_train = fold_scaler.fit_transform(X_fold_train)
        X_fold_val   = fold_scaler.transform(X_fold_val)

        # --- d) Train a fresh model and predict ---
        model = model_fn()
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        y_prob = model.predict_proba(X_fold_val)[:, 1]

        # --- e) Compute quality measures for this fold ---
        fold_scores["accuracy"].append(accuracy_score(y_fold_val, y_pred))
        fold_scores["precision"].append(precision_score(y_fold_val, y_pred))
        fold_scores["recall"].append(recall_score(y_fold_val, y_pred))
        fold_scores["f1"].append(f1_score(y_fold_val, y_pred))
        fold_scores["roc_auc"].append(roc_auc_score(y_fold_val, y_prob))

    return {m: np.array(v) for m, v in fold_scores.items()}


# Models to evaluate — each entry is a factory function returning a fresh instance
# class_weight="balanced" automatically upweights the minority class (malignant)
# by setting weights inversely proportional to class frequencies:
#   w_class = n_samples / (n_classes * n_samples_class)
#   benign weight  ≈ 699 / (2 * 458) ≈ 0.76
#   malignant weight ≈ 699 / (2 * 241) ≈ 1.45
# This penalizes misclassifying malignant samples ~1.9x more than benign,
# pushing the models to improve recall (catch more cancers).
baseline_models = {
    "Logistic Regression (L2)": lambda: LogisticRegression(
        solver="lbfgs", penalty="l2", max_iter=1000,
        class_weight="balanced", random_state=14),
    "Logistic Regression (L1)": lambda: LogisticRegression(
        solver="liblinear", penalty="l1", max_iter=1000,
        class_weight="balanced", random_state=14),
    "SVM (RBF kernel)": lambda: SVC(
        kernel="rbf", probability=True,
        class_weight="balanced", random_state=14),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced", random_state=14),
}

print("=" * 60)
print(" STEP 6: 5-Fold Cross-Validation (from scratch)")
print("=" * 60)
print(" Preprocessing (impute + scale) is re-fitted inside each fold.")

cv_results = {}
for name, model_fn in baseline_models.items():
    scores = kfold_cross_validate(X, y_encoded, model_fn, k=5, random_state=14)
    cv_results[name] = scores
    print(f"\n  {name}:")
    for metric, vals in scores.items():
        print(f"    {metric:>12}: {vals.mean():.4f} ± {vals.std():.4f}  (per-fold: {np.round(vals, 4)})")

#HYPERPARAMETER TUNING
print(f"\n{'=' * 60}")
print(" STEP 7: Hyperparameter Tuning (GridSearchCV, 5-fold)")
print("=" * 60)

# Logistic Regression: two separate grids because lbfgs supports L2 only,
# while liblinear supports both L1 and L2.
# L1 penalty can zero out weak features entirely (embedded feature selection).
# L2 penalty shrinks all coefficients but keeps them non-zero.
# class_weight is tuned: None (default equal weights) vs "balanced" (auto-upweight minority).
param_grids = {
    "Logistic Regression (L2)": {
        "model": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=14),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "class_weight": [None, "balanced"],
        },
    },
    "Logistic Regression (L1)": {
        "model": LogisticRegression(solver="liblinear", max_iter=1000, random_state=14),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1"],
            "class_weight": [None, "balanced"],
        },
    },
    "SVM (RBF kernel)": {
        "model": SVC(kernel="rbf", probability=True, random_state=14),
        "params": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "class_weight": [None, "balanced"],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=14),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
    },
}

best_models = {}

for name, cfg in param_grids.items():
    grid = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"\n  {name}:")
    print(f"    Best params : {grid.best_params_}")
    print(f"    Best CV F1  : {grid.best_score_:.4f}")

#EVALUATION
print(f"\n{'=' * 60}")
print(" STEP 8: Test Set Evaluation (tuned models)")
print("=" * 60)

results = []
roc_data = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, metrics["AUC-ROC"])

    print(f"\n  {name}:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

#MODEL COMPARISON
print(f"\n{'=' * 60}")
print(" STEP 9: Model Comparison")
print("=" * 60)
comparison = pd.DataFrame(results).set_index("Model")
print(comparison.round(4).to_string())

#VISUALIZATIONS
n_models = len(best_models)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ROC Curves (top-left, spanning focus)
ax = axes[0, 0]
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

# Confusion Matrices (remaining slots)
cm_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
for idx, (name, model) in enumerate(best_models.items()):
    ax = cm_axes[idx]
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Benign", "Malignant"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"CM — {name}", fontsize=10)

# Hide unused subplot if any
if n_models < 5:
    axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: model_comparison.png")

# Feature Importance — 3 panels: LR (L2), LR (L1), Random Forest
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Logistic Regression L2 coefficients
lr_l2 = best_models["Logistic Regression (L2)"]
lr_l2_imp = pd.Series(np.abs(lr_l2.coef_[0]), index=X.columns).sort_values()
lr_l2_imp.plot.barh(ax=axes2[0], color="steelblue")
axes2[0].set_title("LR (L2) — |Coefficients|")
axes2[0].set_xlabel("Absolute Coefficient Value")

# Logistic Regression L1 coefficients — shows which features get zeroed out
lr_l1 = best_models["Logistic Regression (L1)"]
lr_l1_imp = pd.Series(np.abs(lr_l1.coef_[0]), index=X.columns).sort_values()
colors = ["lightcoral" if v == 0 else "steelblue" for v in lr_l1_imp]
lr_l1_imp.plot.barh(ax=axes2[1], color=colors)
axes2[1].set_title("LR (L1) — |Coefficients|  (red = zeroed out)")
axes2[1].set_xlabel("Absolute Coefficient Value")

# Random Forest feature importance (Gini)
rf = best_models["Random Forest"]
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
rf_importance.plot.barh(ax=axes2[2], color="forestgreen")
axes2[2].set_title("Random Forest — Gini Importance")
axes2[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved: feature_importance.png")


