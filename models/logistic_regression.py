import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def train_logistic_regression(df, target_col, params):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create and train the logistic regression model
    model = LogisticRegression(
        solver=params['solver'],
        penalty=params['penalty'],
        C=params['C'],
        max_iter=params['max_iter'],
        tol=params['tol'],
        fit_intercept=params['fit_intercept'],
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)

    # Return model, accuracy, training columns, and scaler for predictions
    return model, accuracy, X_train.columns, scaler


def predict_logistic_regression(model, df_test, train_columns, scaler):
    X_test = df_test.copy()

    # Encode categorical columns in the test set
    for col in X_test.select_dtypes(include='object').columns:
        X_test[col] = LabelEncoder().fit_transform(X_test[col])

    # Ensure all training columns are present in the test set
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_columns]

    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions and probabilities
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Add the results to the test data
    df_test['Prediction'] = preds
    df_test['Probability_of_Default'] = probs

    return df_test