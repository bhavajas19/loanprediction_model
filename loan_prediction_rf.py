# =====================================================
# LOAN PREDICTION - RANDOM FOREST MODEL
# =====================================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =====================================================
# 1. PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")


# =====================================================
# 2. LOAD DATA
# =====================================================

train_data = pd.read_csv(TRAIN_PATH)
test_data  = pd.read_csv(TEST_PATH)

test_loan_ids = test_data["Loan_ID"]

train_data.drop("Loan_ID", axis=1, inplace=True)
test_data.drop("Loan_ID", axis=1, inplace=True)

print("Data loaded successfully")


# =====================================================
# 3. HANDLE MISSING VALUES
# =====================================================

for col in train_data.columns:
    if train_data[col].dtype == "object":
        train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    else:
        train_data[col].fillna(train_data[col].mean(), inplace=True)

for col in test_data.columns:
    if test_data[col].dtype == "object":
        test_data[col].fillna(test_data[col].mode()[0], inplace=True)
    else:
        test_data[col].fillna(test_data[col].mean(), inplace=True)


# =====================================================
# 4. ENCODE CATEGORICAL VARIABLES
# =====================================================

categorical_cols = train_data.select_dtypes(include="object").columns.tolist()
categorical_cols.remove("Loan_Status")

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])

    test_data[col] = test_data[col].apply(
        lambda x: x if x in le.classes_ else le.classes_[0]
    )
    test_data[col] = le.transform(test_data[col])

# Encode target
target_encoder = LabelEncoder()
train_data["Loan_Status"] = target_encoder.fit_transform(train_data["Loan_Status"])


# =====================================================
# 5. SPLIT FEATURES & TARGET
# =====================================================

X = train_data.drop("Loan_Status", axis=1)
y = train_data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================
# 6. TRAIN RANDOM FOREST MODEL
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest model trained")


# =====================================================
# 7. EVALUATE MODEL
# =====================================================

y_pred = rf_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =====================================================
# 8. PREDICT ON TEST DATA
# =====================================================

test_predictions = rf_model.predict(test_data)

final_predictions = [
    "Approved" if p == 1 else "Not Approved"
    for p in test_predictions
]


# =====================================================
# 9. SAVE OUTPUT
# =====================================================

output_path = os.path.join(DATA_DIR, "loan_predictions_rf.csv")

output_df = pd.DataFrame({
    "Loan_ID": test_loan_ids,
    "Loan_Status_Prediction": final_predictions
})

output_df.to_csv(output_path, index=False)

print("\nPredictions saved at:", output_path)
