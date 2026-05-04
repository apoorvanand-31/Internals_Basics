import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("corrosion_risk_score", axis=1)
y = df["corrosion_risk_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow setup
mlflow.set_experiment("pipewatch-corrosion-risk-score")

models = {
    "Lasso": Lasso(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []
best_mae = float("inf")
best_model_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("team", "ml_engineering")

        mlflow.sklearn.log_model(model, "model")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        if mae < best_mae:
            best_mae = mae
            best_model_name = name

# Save result
output = {
    "experiment_name": "pipewatch-corrosion-risk-score",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 1 Done")