from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
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

# ─── 1. LOAD DATASET ─────────────────────────────────────────────────────────
breast_cancer = fetch_ucirepo(id=15)

X = breast_cancer.data.features.copy()
y = breast_cancer.data.targets.copy()

# ─── 2. ENCODE TARGET VARIABLE (2 → 0 benign, 4 → 1 malignant) ──────────────
y_encoded = y.iloc[:, 0].map({2: 0, 4: 1})

# ─── 3. TRAIN / TEST SPLIT (80/20, stratified) ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

# ─── 4. IMPUTE MISSING VALUES (median, fit on train only) ────────────────────
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X.columns)

# ─── 5. FEATURE SCALING (StandardScaler, fit on train only) ──────────────────
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

# ─── 6. CROSS-VALIDATION (5-fold stratified, before tuning) ──────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

baseline_models = {
    "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

print("=" * 60)
print(" STEP 6: 5-Fold Cross-Validation (baseline models)")
print("=" * 60)

cv_results = {}
for name, model in baseline_models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_results[name] = scores
    print(f"\n  {name}:")
    for metric in scoring:
        key = f"test_{metric}"
        vals = scores[key]
        print(f"    {metric:>12}: {vals.mean():.4f} ± {vals.std():.4f}")

# ─── 7. HYPERPARAMETER TUNING (GridSearchCV) ─────────────────────────────────
print(f"\n{'=' * 60}")
print(" STEP 7: Hyperparameter Tuning (GridSearchCV, 5-fold)")
print("=" * 60)

param_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
        },
    },
    "SVM (RBF kernel)": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "params": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
    },
}

best_models = {}

for name, cfg in param_grids.items():
    grid = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"\n  {name}:")
    print(f"    Best params : {grid.best_params_}")
    print(f"    Best CV F1  : {grid.best_score_:.4f}")

# ─── 8. EVALUATE TUNED MODELS ON TEST SET ────────────────────────────────────
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

# ─── 9. MODEL COMPARISON TABLE ───────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(" STEP 9: Model Comparison")
print("=" * 60)
comparison = pd.DataFrame(results).set_index("Model")
print(comparison.round(4).to_string())

# ─── 10. VISUALIZATIONS ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 10a. ROC Curves
ax = axes[0, 0]
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# 10b–d. Confusion Matrices
for idx, (name, model) in enumerate(best_models.items()):
    row, col = divmod(idx + 1, 2)
    ax = axes[row, col]
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Benign", "Malignant"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: model_comparison.png")

# 10e. Feature Importance
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression coefficients (absolute value)
lr = best_models["Logistic Regression"]
lr_importance = pd.Series(np.abs(lr.coef_[0]), index=X.columns).sort_values()
lr_importance.plot.barh(ax=axes2[0], color="steelblue")
axes2[0].set_title("Logistic Regression — |Coefficients|")
axes2[0].set_xlabel("Absolute Coefficient Value")

# Random Forest feature importance (Gini)
rf = best_models["Random Forest"]
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
rf_importance.plot.barh(ax=axes2[1], color="forestgreen")
axes2[1].set_title("Random Forest — Gini Importance")
axes2[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved: feature_importance.png")

