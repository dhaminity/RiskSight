import streamlit as st
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt

from models.random_forest import train_random_forest, predict_random_forest
from models.catboost_model import train_catboost, predict_catboost
from models.xgboost_model import train_xgboost, predict_xgboost
from models.logistic_regression import train_logistic_regression, predict_logistic_regression

st.set_page_config(page_title="Credit Risk Insights - Unified Classifier", layout="wide")

# Create folders
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Initialize session state
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_columns' not in st.session_state:
    st.session_state.train_columns = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'params' not in st.session_state:
    st.session_state.params = None

# Reset button
if st.sidebar.button("🔄 Reset All"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()


# Page title
st.title("💳 Credit Risk Insights Dashboard")
st.markdown("""
Welcome to the **Credit Risk Insights** app!
Select your preferred model, upload data, fine-tune parameters, run predictions, and visualize/download the results — all in one place.
""")

# Model selection
model_type = st.selectbox("✨ Select Model for Analysis", ["Random Forest", "CatBoost", "XGBoost", "Logistic Regression"])

# Sidebar navigation
menu = ["Input Data", "Fine-tune Parameters", "Run Model", "View & Download Output"]
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", menu)

# Step 1: Input Data
if choice == "Input Data":
    st.header("Step 1️⃣: Upload Training Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_train = df
        df.to_csv("data/input/train.csv", index=False)
        st.success("Training data uploaded successfully.")
        st.dataframe(df.head(10))
    elif st.session_state.df_train is not None:
        st.info("Uploaded training data:")
        st.dataframe(st.session_state.df_train.head(10))

# Step 2: Fine-tune Parameters
elif choice == "Fine-tune Parameters":
    st.header("Step 2️⃣: Fine-tune Model Parameters")
    if st.session_state.df_train is not None:
        df = st.session_state.df_train
        target_col = df.columns[-1]

        if model_type == "Random Forest":
            st.subheader("🌲 Random Forest Parameters")
            params = {
                'n_estimators': st.slider("n_estimators", 10, 200, 100),
                'max_depth': st.slider("max_depth", 1, 20, 10),
                'min_samples_split': st.slider("min_samples_split", 2, 20, 10),
                'min_samples_leaf': st.slider("min_samples_leaf", 1, 20, 5),
                'max_features': st.selectbox("max_features", ['sqrt', 'log2', None]),
                'criterion': st.selectbox("criterion", ['gini', 'entropy'])
            }
            if st.button("Train Random Forest Model"):
                model, accuracy, train_cols = train_random_forest(df, target_col, params)
                st.session_state.model = model
                st.session_state.train_columns = train_cols
                st.session_state.params = params
                st.success(f"Random Forest trained with accuracy: {accuracy:.2f}")

        elif model_type == "CatBoost":
            st.subheader("🐱 CatBoost Parameters")
            params = {
                'iterations': st.slider("iterations", 100, 1000, 500),
                'depth': st.slider("depth", 1, 10, 6),
                'learning_rate': st.slider("learning_rate", 0.01, 0.5, 0.1),
                'l2_leaf_reg': st.slider("l2_leaf_reg", 1, 10, 3),
                'early_stopping_rounds': st.slider("Early Stopping Rounds", 10, 100, 50),
                'loss_function': st.selectbox("Loss Function", ['Logloss', 'CrossEntropy'])
            }
            if st.button("Train CatBoost Model"):
                model, accuracy, train_cols = train_catboost(df, target_col, params)
                st.session_state.model = model
                st.session_state.train_columns = train_cols
                st.session_state.params = params
                st.success(f"CatBoost trained with accuracy: {accuracy:.4f}")

        elif model_type == "XGBoost":
            st.subheader("🚀 XGBoost Parameters")
            params = {
                'n_estimators': st.slider("n_estimators", 100, 1000, 300),
                'max_depth': st.slider("max_depth", 1, 20, 6),
                'learning_rate': st.slider("learning_rate", 0.01, 0.5, 0.1),
                'scale_pos_weight': st.slider("scale_pos_weight", 1, 10, 1),
                'max_leaves': st.slider("max_leaves", 1, 100, 10),
                'subsample': st.slider("subsample", 0.5, 1.0, 0.8)
            }
            if st.button("Train XGBoost Model"):
                model, accuracy, train_cols = train_xgboost(df, target_col, params)
                st.session_state.model = model
                st.session_state.train_columns = train_cols
                st.session_state.params = params
                st.success(f"XGBoost trained with accuracy: {accuracy:.4f}")

        elif model_type == "Logistic Regression":
            st.subheader("📈 Logistic Regression Parameters")
            params = {
                'solver': st.selectbox("Solver", ['lbfgs', 'liblinear', 'saga']),
                'penalty': st.selectbox("Penalty", ['l2', 'l1', 'elasticnet', 'none']),
                'C': st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0),
                'max_iter': st.slider("Max Iterations", 100, 1000, 300),
                'tol': st.slider("Tolerance", 1e-6, 1e-2, 1e-4),
                'fit_intercept': st.checkbox("Fit Intercept", True)
            }
            if st.button("Train Logistic Regression Model"):
                model, accuracy, train_cols, scaler = train_logistic_regression(df, target_col, params)
                st.session_state.model = model
                st.session_state.train_columns = train_cols
                st.session_state.scaler = scaler
                st.session_state.params = params
                st.success(f"Logistic Regression trained with accuracy: {accuracy:.4f}")
    else:
        st.warning("Please upload training data first.")

# Step 3: Run Model
elif choice == "Run Model":
    st.header("Step 3️⃣: Run Model on Test Data")
    if st.session_state.model is not None:
        test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")
        if test_file:
            df_test = pd.read_csv(test_file)
            try:
                if model_type == "Random Forest":
                    result_df = predict_random_forest(st.session_state.model, df_test, st.session_state.train_columns)
                elif model_type == "CatBoost":
                    result_df = predict_catboost(st.session_state.model, df_test, st.session_state.train_columns)
                elif model_type == "XGBoost":
                    result_df = predict_xgboost(st.session_state.model, df_test, st.session_state.train_columns)
                elif model_type == "Logistic Regression":
                    result_df = predict_logistic_regression(
                        st.session_state.model, df_test, st.session_state.train_columns, st.session_state.scaler
                    )

                st.session_state.predictions = result_df
                st.success("Prediction and probability added to test data.")
                st.dataframe(result_df.head(10))

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        elif st.session_state.predictions is not None:
            st.info("Generated predictions:")
            st.dataframe(st.session_state.predictions.head(10))
    else:
        st.warning("Please train the model first.")

# Step 4: View & Download Output
elif choice == "View & Download Output":
    st.header("Step 4️⃣: Visualize & Download Results")
    if st.session_state.predictions is not None:
        df_pred = st.session_state.predictions

        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_pred['Prediction'].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title("Prediction Distribution (0: Paying, 1: Not Paying)")
        ax.set_ylabel('Count')
        ax.set_xlabel('Prediction Class')
        st.pyplot(fig)

        # Add download button
        csv = df_pred.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(
            f"""
            <a href="data:file/csv;base64,{b64}" download="predicted_output.csv"
               style="color: blue; text-decoration: underline; font-size: 16px;">
               📥 Download Full Predicted File
            </a>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please run the model first.")
