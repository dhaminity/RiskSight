import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def train_catboost(df, target_col, params):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(
        iterations=params['iterations'],
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        cat_features=categorical_features,
        verbose=100,
        early_stopping_rounds=params['early_stopping_rounds']
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    y_pred = model.predict(X_val)
    accuracy = (y_pred == y_val).mean()

    return model, accuracy, X.columns

def predict_catboost(model, df_test, train_columns):
    X_test = df_test.copy()
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_columns]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    df_test['Prediction'] = preds
    df_test['Probability_of_Default'] = probs

    return df_test
