import json
import pandas as pd

# Load training data
train_df = pd.read_csv("data/training_data.csv")

train_mean_age = train_df["pipe_age_years"].mean()
train_mean_moisture = train_df["moisture_pct"].mean()

# Load logs
logs = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

live_mean_age = df["input"].apply(lambda x: x["pipe_age_years"]).mean()
live_mean_moisture = df["input"].apply(lambda x: x["moisture_pct"]).mean()

def check(feature, train, live, threshold):
    shift = abs(train - live)
    return {
        "feature": feature,
        "train_mean": train,
        "live_mean": live,
        "shift": shift,
        "threshold": threshold,
        "status": "ALERT" if shift > threshold else "OK"
    }

alerts = [
    check("pipe_age_years", train_mean_age, live_mean_age, 10.09),
    check("moisture_pct", train_mean_moisture, live_mean_moisture, 21.5)
]

output = {
    "total_predictions": len(df),
    "mean_prediction": df["prediction"].mean(),
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

with open("results/step4_s5.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 4 Done")