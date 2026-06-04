#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning and Training for Student Performance Prediction
- Loads dataset: student_data.csv
- Creates target risk categories
- Preprocesses (impute, encode, scale)
- Tunes key XGBoost hyperparameters using RandomizedSearchCV
- Trains best model, evaluates, prints best params & accuracy
- Saves artifacts to student_performance_xgb_tuned.pkl compatible with existing API
"""

import warnings
warnings.filterwarnings('ignore')

import json
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from scipy.stats import uniform, randint
import xgboost as xgb
import joblib


DATA_PATH = 'student_performance_data.csv'  # Ensure this file exists in the working directory
OUTPUT_PKL = 'student_performance_xgb_tuned.pkl'

# Target binning setup (align with previous training defaults)
TARGET_COLUMN = 'final_exam_score'
BINS = [0, 60, 75, 90, 101]
LABELS = ['High Risk', 'Medium Risk', 'Low Risk', 'Excellent']


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    df = df.copy()
    df['risk_category'] = pd.cut(df[TARGET_COLUMN], bins=BINS, labels=LABELS, right=False)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], list]:
    """Preprocess data to match existing pipeline (encoders, scaler, feature order)."""
    df = df.copy()

    # Separate target
    if 'risk_category' not in df.columns:
        raise ValueError("Column 'risk_category' is missing; call create_target_variable first")

    # Identify numeric and categorical columns (excluding target)
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if 'risk_category' in numeric_cols:
        numeric_cols.remove('risk_category')
    if 'risk_category' in categorical_cols:
        categorical_cols.remove('risk_category')

    # Impute
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Encode categorical features
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    df['risk_category_encoded'] = target_le.fit_transform(df['risk_category'])
    label_encoders['risk_category'] = target_le

    # Prepare X, y (drop the target's raw score to avoid leakage)
    X = df.drop(['risk_category', 'risk_category_encoded', 'final_exam_score'], axis=1, errors='ignore')
    y = df['risk_category_encoded'].values

    # Preserve feature order BEFORE scaling
    feature_names = X.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    artifacts = {
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_names': feature_names,
    }
    return X_scaled, y, artifacts, feature_names


def tune_and_train(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 40, random_state: int = 42) -> RandomizedSearchCV:
    """Randomized hyperparameter search over key XGBoost params."""
    # Base estimator
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        random_state=random_state,
        n_jobs=-1,
        tree_method='hist',
    )

    # Parameter distributions (match user ranges)
    param_dist = {
        'n_estimators': randint(200, 1001),
        'learning_rate': uniform(0.01, 0.09),  # 0.01–0.1
        'max_depth': randint(4, 9),
        'subsample': uniform(0.7, 0.2),        # 0.7–0.9
        'colsample_bytree': uniform(0.7, 0.2), # 0.7–0.9
        # Keep additional stabilizers
        'min_child_weight': randint(1, 6),
        'gamma': uniform(0.0, 0.4),
        'reg_alpha': uniform(0.0, 0.2),
        'reg_lambda': uniform(0.7, 0.6),
    }

    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    return search


def main():
    # 1) Load dataset
    print(f"[INFO] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 2) Create target variable
    print("[INFO] Creating target categories (risk_category)")
    df = create_target_variable(df)

    # 3) Preprocess
    print("[INFO] Preprocessing (impute, encode, scale)")
    X, y, artifacts, feature_names = preprocess(df)

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Tune hyperparameters and train
    print("[INFO] Starting hyperparameter tuning (RandomizedSearchCV)")
    search = tune_and_train(X_train, y_train)

    # 6) Evaluate
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    target_le = artifacts['label_encoders']['risk_category']

    print("\n[RESULT] Best Parameters:")
    print(json.dumps(search.best_params_, indent=2))
    print(f"\n[RESULT] Best CV Score: {search.best_score_:.4f}")
    print(f"[RESULT] Test Accuracy: {acc:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    # 7) Save artifacts
    all_artifacts = {
        'model': best_model,
        'scaler': artifacts['scaler'],
        'label_encoders': artifacts['label_encoders'],
        'feature_names': artifacts['feature_names'],
        'model_type': 'xgboost',
    }
    joblib.dump(all_artifacts, OUTPUT_PKL)
    print(f"\n[SUCCESS] Tuned model saved to {OUTPUT_PKL}")


if __name__ == '__main__':
    main()
