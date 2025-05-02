import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_random_forest(df, target_col, params):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        criterion=params['criterion'],
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return model, accuracy, X.columns

def predict_random_forest(model, df_test, train_columns):
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
