import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

# Load
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

model = joblib.load("models/model.pkl")

# Predict
preds = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Save predictions
pd.DataFrame(preds, columns=["prediction"]).to_csv("data/predictions.csv", index=False)

# Save metrics
metrics = {"mse": mse, "r2": r2}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)