import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_xgboost(df, target_col, params):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical features
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        scale_pos_weight=params['scale_pos_weight'],
        max_leaves=params['max_leaves'],
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return model, accuracy, X.columns

def predict_xgboost(model, df_test, train_columns):
    X_test = df_test.copy()
    for col in X_test.select_dtypes(include='object').columns:
        X_test[col] = LabelEncoder().fit_transform(X_test[col])

    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_columns]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    df_test['Prediction'] = preds
    df_test['Probability_of_Default'] = probs

    return df_test

