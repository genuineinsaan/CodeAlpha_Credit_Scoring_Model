import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/credit_data.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")

# ==========================================
# ENCODE TARGET
# ==========================================

df["Risk"] = df["Risk"].map({
    "good": 1,
    "bad": 0
})

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

categorical_columns = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Risk", axis=1)
y = df["Risk"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

log_model = LogisticRegression(
    max_iter=5000,
    solver="liblinear"
)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)

print(f"\nAccuracy: {log_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))

print("\nClassification Report:")
print(classification_report(y_test, log_pred))

# ==========================================
# RANDOM FOREST
# ==========================================

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\nAccuracy: {rf_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(f"Logistic Regression Accuracy : {log_accuracy:.4f}")
print(f"Random Forest Accuracy       : {rf_accuracy:.4f}")

if rf_accuracy > log_accuracy:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Logistic Regression")

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    rf_model,
    "models/credit_scoring_model.pkl"
)

print("\nSaved: models/credit_scoring_model.pkl")

print("\nPROJECT COMPLETED SUCCESSFULLY!")