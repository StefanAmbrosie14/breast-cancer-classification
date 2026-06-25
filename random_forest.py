from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

y_encoded = y.iloc[:,0].map({2: 0, 4: 1})

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state = 14, test_size = 0.2)

#impute missing values
imputer = SimpleImputer(strategy = "median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X.columns)

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X.columns)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy score {accuracy_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred)}")
print(f"ROC_AUC: {roc_auc_score(y_test, y_pred)}")
print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")
