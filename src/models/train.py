import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# Load params
params = joblib.load("models/best_params.pkl")

# Train
model = RandomForestRegressor(**params)
model.fit(X_train, y_train.values.ravel())

# Save model
joblib.dump(model, "models/model.pkl")