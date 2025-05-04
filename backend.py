from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io

from models.random_forest import train_random_forest, predict_random_forest
from models.catboost_model import train_catboost, predict_catboost
from models.xgboost_model import train_xgboost, predict_xgboost
from models.logistic_regression import train_logistic_regression, predict_logistic_regression

app = FastAPI(title="Credit Risk Insights API")

# Store session-like objects
session_state = {
    'df_train': None,
    'model': None,
    'train_columns': None,
    'scaler': None,
    'predictions': None,
    'params': None
}

@app.post("/upload-train")
async def upload_train(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    session_state['df_train'] = df
    return JSONResponse({
        "message": "Training data uploaded successfully.",
        "top_10_records": df.head(10).to_dict(orient='records')
    })


@app.post("/train-model")
async def train_model(
    model_type: str = Form(...),
    params: str = Form(...)
):
    if session_state['df_train'] is None:
        return JSONResponse({"error": "Please upload training data first."}, status_code=400)

    df = session_state['df_train']
    target_col = df.columns[-1]
    params_dict = eval(params)  # ⚠️ Make sure to pass a dictionary string like '{"iterations":500,"depth":6,...}'

    if model_type == "Random Forest":
        model, accuracy, train_cols = train_random_forest(df, target_col, params_dict)
    elif model_type == "CatBoost":
        model, accuracy, train_cols = train_catboost(df, target_col, params_dict)
    elif model_type == "XGBoost":
        model, accuracy, train_cols = train_xgboost(df, target_col, params_dict)
    elif model_type == "Logistic Regression":
        model, accuracy, train_cols, scaler = train_logistic_regression(df, target_col, params_dict)
        session_state['scaler'] = scaler
    else:
        return JSONResponse({"error": f"Unsupported model type: {model_type}"}, status_code=400)

    session_state['model'] = model
    session_state['train_columns'] = train_cols
    session_state['params'] = params_dict

    return JSONResponse({
        "message": f"{model_type} trained successfully.",
        "accuracy": float(accuracy)
    })


@app.post("/upload-test")
async def upload_test(file: UploadFile = File(...)):
    if session_state['model'] is None:
        return JSONResponse({"error": "Please train the model first."}, status_code=400)

    content = await file.read()
    df_test = pd.read_csv(io.BytesIO(content))

    try:
        if session_state['params']:
            model_type = detect_model_type(session_state['model'])
        else:
            return JSONResponse({"error": "Model type detection failed."}, status_code=400)

        if model_type == "Random Forest":
            result_df = predict_random_forest(session_state['model'], df_test, session_state['train_columns'])
        elif model_type == "CatBoost":
            result_df = predict_catboost(session_state['model'], df_test, session_state['train_columns'])
        elif model_type == "XGBoost":
            result_df = predict_xgboost(session_state['model'], df_test, session_state['train_columns'])
        elif model_type == "Logistic Regression":
            result_df = predict_logistic_regression(
                session_state['model'], df_test, session_state['train_columns'], session_state['scaler']
            )
        else:
            return JSONResponse({"error": "Unsupported model for prediction."}, status_code=400)

        session_state['predictions'] = result_df

        return JSONResponse({
            "message": "Predictions generated.",
            "top_10_with_predictions": result_df.head(10).to_dict(orient='records')
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def detect_model_type(model_obj):
    """
    Utility function to detect the model type based on object class.
    Extend this if you add more models.
    """
    from sklearn.ensemble import RandomForestClassifier
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression

    if isinstance(model_obj, RandomForestClassifier):
        return "Random Forest"
    elif isinstance(model_obj, CatBoostClassifier):
        return "CatBoost"
    elif isinstance(model_obj, XGBClassifier):
        return "XGBoost"
    elif isinstance(model_obj, LogisticRegression):
        return "Logistic Regression"
    else:
        return None
