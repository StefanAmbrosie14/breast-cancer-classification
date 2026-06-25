from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

print(" Dataset: Breast Cancer Wisconsin (Original)")
print(f"  Source       : UCI ML Repository (id=15)")
print(f"  Samples      : {len(X)}")
print(f"  Features     : {X.shape[1]} ({', '.join(X.columns)})")
print(f"  Target       : Class (2 = Benign, 4 = Malignant)")
print(f"  Missing vals : {X.isna().sum().sum()} (in {X.columns[X.isna().any()].tolist()})")

print(f"\nFeature summary:")
class_counts = y.iloc[:, 0].value_counts().sort_index()
for val, count in class_counts.items():
    label = "Benign" if val == 2 else "Malignant"
    print(f"  {label} ({val}): {count} ({count/len(y)*100:.1f}%)")

y_encoded = y.iloc[:, 0].map({2: 0, 4: 1})

# 16 rows have missing Bare_nuclei, so we drop
mask = X.notna().all(axis=1)
X = X[mask].reset_index(drop=True)
y_encoded = y_encoded[mask].reset_index(drop=True)
print(f"\nDropped {(~mask).sum()} rows with missing values. Remaining: {len(X)} samples.")

print(f"\nClass distribution (after cleaning):")
for cls, label in [(0, "Benign"), (1, "Malignant")]:
    count = (y_encoded == cls).sum()
    print(f"  {label} ({cls}): {count} ({count/len(y_encoded)*100:.1f}%)")

# ─── TRAIN/TEST SPLIT ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=14,
    stratify=y_encoded,
)
print(f"\nTrain/Test split: {len(X_train)} train, {len(X_test)} test")
print(f"  Train class balance: {(y_train==0).sum()} benign, {(y_train==1).sum()} malignant")
print(f"  Test  class balance: {(y_test==0).sum()} benign, {(y_test==1).sum()} malignant")


def stratified_k_fold_indices(y, k, random_state=14):
    rng = np.random.RandomState(random_state)
    indices_by_class = {}
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        indices_by_class[cls] = cls_indices

    folds = [[] for _ in range(k)]
    for cls, indices in indices_by_class.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    # shuffle within each fold so samples aren't ordered by class
    for fold in folds:
        rng.shuffle(fold)

    return [np.array(fold) for fold in folds]


def kfold_cross_validate(X, y, model_fn, k=5, random_state=14):
    """Per-fold: scale → train → predict → score. No data leakage."""
    X_arr = np.array(X)
    y_arr = np.array(y)
    folds = stratified_k_fold_indices(y_arr, k, random_state)

    fold_scores = {m: [] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}

    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
        y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

        fold_scaler = StandardScaler()
        X_fold_train = fold_scaler.fit_transform(X_fold_train)
        X_fold_val   = fold_scaler.transform(X_fold_val)

        model = model_fn()
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        y_prob = model.predict_proba(X_fold_val)[:, 1]

        fold_scores["accuracy"].append(accuracy_score(y_fold_val, y_pred))
        fold_scores["precision"].append(precision_score(y_fold_val, y_pred))
        fold_scores["recall"].append(recall_score(y_fold_val, y_pred))
        fold_scores["f1"].append(f1_score(y_fold_val, y_pred))
        fold_scores["roc_auc"].append(roc_auc_score(y_fold_val, y_prob))

    return {m: np.array(v) for m, v in fold_scores.items()}

#Crossvalidation
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

print(" Baseline Cross-Validation ")
cv_results = {}
for name, model_fn in baseline_models.items():
    scores = kfold_cross_validate(X_train, y_train, model_fn, k=5, random_state=14)
    cv_results[name] = scores
    print(f"\n  {name}:")
    for metric, vals in scores.items():
        print(f"    {metric:>12}: {vals.mean():.4f} ± {vals.std():.4f}  (per-fold: {np.round(vals, 4)})")


#Hyperparameter Tuning
print(" Hyperparameter Tuning ")

param_grids = {
    "Logistic Regression (L2)": {
        "pipeline": Pipeline([
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(solver="lbfgs", max_iter=1000, random_state=14)),
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l2"],
            "model__class_weight": [None, "balanced"],
        },
    },
    "Logistic Regression (L1)": {
        "pipeline": Pipeline([
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(solver="liblinear", max_iter=1000, random_state=14)),
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l1"],
            "model__class_weight": [None, "balanced"],
        },
    },
    "SVM (RBF kernel)": {
        "pipeline": Pipeline([
            ("scaler",  StandardScaler()),
            ("model",   SVC(kernel="rbf", probability=True, random_state=14)),
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "model__shrinking": [True, False],
            "model__class_weight": [None, "balanced"],
        },
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("scaler",  StandardScaler()),
            ("model",   RandomForestClassifier(random_state=14)),
        ]),
        "params": {
            "model__n_estimators": [50, 100, 200, 500],
            "model__max_depth": [None, 3, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
    },
}

best_pipelines = {}

for name, cfg in param_grids.items():
    grid = GridSearchCV(
        estimator=cfg["pipeline"],
        param_grid=cfg["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_pipelines[name] = grid.best_estimator_

    clean_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
    print(f"\n  {name}:")
    print(f"    Best params : {clean_params}")
    print(f"    Best CV F1  : {grid.best_score_:.4f}")


#Evaluation
print(f"\n{'=' * 60}")
print(" STEP 8: Test Set Evaluation (tuned pipelines)")
print("=" * 60)

results = []
roc_data = {}

for name, pipeline in best_pipelines.items():
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

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

print("Model Comparison")
comparison = pd.DataFrame(results).set_index("Model")
print(comparison.round(4).to_string())


# ROC Curves
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax_roc.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves")
ax_roc.legend(loc="lower right", fontsize=9)
ax_roc.grid(alpha=0.3)
fig_roc.tight_layout()
fig_roc.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
print("\nSaved: roc_curves.png")

# Confusion Matrices
for name, pipeline in best_pipelines.items():
    y_pred = pipeline.predict(X_test)

    fig_cm, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Benign", "Malignant"],
        cmap="Blues",
        ax=ax_abs,
    )
    ax_abs.set_title("Absolute Counts")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Benign", "Malignant"],
        normalize="true",
        values_format=".1%",
        cmap="Blues",
        ax=ax_norm,
    )
    ax_norm.set_title("Normalized (per true class)")

    fig_cm.suptitle(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    fig_cm.tight_layout()

    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"confusion_matrix_{safe_name}.png"
    fig_cm.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig_cm)

# Feature Importance
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))

lr_l2_model = best_pipelines["Logistic Regression (L2)"].named_steps["model"]
lr_l1_model = best_pipelines["Logistic Regression (L1)"].named_steps["model"]
rf_model    = best_pipelines["Random Forest"].named_steps["model"]

# Logistic Regression L2 coefficients
lr_l2_imp = pd.Series(np.abs(lr_l2_model.coef_[0]), index=X.columns).sort_values()
lr_l2_imp.plot.barh(ax=axes2[0], color="steelblue")
axes2[0].set_title("LR (L2) — |Coefficients|")
axes2[0].set_xlabel("Absolute Coefficient Value")

# Logistic Regression L1 coefficients — shows which features get zeroed out
lr_l1_imp = pd.Series(np.abs(lr_l1_model.coef_[0]), index=X.columns).sort_values()
colors = ["lightcoral" if v == 0 else "steelblue" for v in lr_l1_imp]
lr_l1_imp.plot.barh(ax=axes2[1], color=colors)
axes2[1].set_title("LR (L1) — |Coefficients|  (red = zeroed out)")
axes2[1].set_xlabel("Absolute Coefficient Value")

# SVM permutation importance — uses the full pipeline so shuffled features
# go through the same scaler the model was trained with
svm_pipeline = best_pipelines["SVM (RBF kernel)"]
perm_result = permutation_importance(
    svm_pipeline, X_test, y_test,
    n_repeats=30, random_state=14, scoring="accuracy",
)
svm_imp = pd.Series(perm_result.importances_mean, index=X.columns).sort_values()
svm_std = pd.Series(perm_result.importances_std, index=X.columns).reindex(svm_imp.index)
svm_imp.plot.barh(ax=axes2[2], xerr=svm_std, color="darkorange", capsize=3)
axes2[2].set_title("SVM (RBF) — Permutation Importance")
axes2[2].set_xlabel("Mean Accuracy Drop")

# Random Forest feature importance (Gini)
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
rf_importance.plot.barh(ax=axes2[3], color="forestgreen")
axes2[3].set_title("Random Forest — Gini Importance")
axes2[3].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved: feature_importance.png")
