import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import json
import joblib
import os

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("corrosion_risk_score", axis=1)
y = df["corrosion_risk_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

# Set MLflow experiment
mlflow.set_experiment("pipewatch-corrosion-risk-score")

with mlflow.start_run(run_name="tuning-pipewatch"):

    model = GradientBoostingRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # 🔥 Save model (IMPORTANT)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")

    # Prepare JSON output
    output = {
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": grid.best_params_,
        "best_mae": mae,
        "best_cv_mae": -grid.best_score_,
        "parent_run_name": "tuning-pipewatch"
    }

    # Save JSON
    os.makedirs("results", exist_ok=True)
    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=2)

print("Task 2 Done + Model Saved")