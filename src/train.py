import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EXPERIMENT_NAME = "bike-demand-experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_PATH = os.path.join("data", "bike_data.csv")

def load_and_preprocess_data(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    feature_cols = [
        "season", "yr", "mnth", "hr", "holiday", 
        "weekday", "workingday", "weathersit", 
        "temp", "atemp", "hum", "windspeed"
    ]
    target_col = "cnt"
    
    # Cast all integer feature columns to float64 to avoid MLflow schema warnings
    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

def run_training_pipeline():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, 
            max_depth=12, 
            random_state=42
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6, 
            random_state=42
        )
    }

    # Explicitly trust the internal scikit-learn Cython tree type for skops serialization
    trusted_types = ["sklearn.tree._tree.Tree"]

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Model training
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Metric calculation
            r2, mae, rmse = evaluate_metrics(y_test, predictions)
            
            # Log hyperparameters
            for param_key, param_value in model.get_params().items():
                mlflow.log_param(param_key, param_value)
            
            # Log metrics
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            
            # Log trained model artifact with trusted types configured
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    input_example=X_train.iloc[:2],
                    skops_trusted_types=trusted_types
                )
            except TypeError:
                # Fallback for MLflow versions without skops_trusted_types kwarg
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=X_train.iloc[:2]
                )
            
            print(f"[{model_name}] R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

if __name__ == "__main__":
    run_training_pipeline()