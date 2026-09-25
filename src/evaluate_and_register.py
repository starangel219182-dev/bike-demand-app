import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "bike-demand-experiment"
REGISTERED_MODEL_NAME = "BikeDemandChampion"

def promote_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' does not exist.")
    
    # Query runs sorted by RMSE ascending (lowest error wins)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=10
    )

    if not runs:
        raise ValueError("No valid runs found in the experiment.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics["rmse"]
    best_r2 = best_run.data.metrics["r2_score"]
    model_type = best_run.data.tags.get("mlflow.runName", "Unknown")

    print("==========================================")
    print(f"Selected Best Model: {model_type}")
    print(f"Run ID: {best_run_id}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Best R2: {best_r2:.4f}")
    print("==========================================")

    # Register the model
    model_uri = f"runs:/{best_run_id}/model"
    model_version_details = mlflow.register_model(
        model_uri=model_uri, 
        name=REGISTERED_MODEL_NAME
    )

    # Set champion alias to the newly registered model version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version_details.version
    )
    
    print(f"Model version {model_version_details.version} tagged with alias '@champion'.")

if __name__ == "__main__":
    promote_best_model()