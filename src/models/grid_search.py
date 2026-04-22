import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Load
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# Model
model = RandomForestRegressor()

params = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(model, params, cv=3, scoring="neg_mean_squared_error")

grid.fit(X_train, y_train.values.ravel())

# Save best params
os.makedirs("models", exist_ok=True)
joblib.dump(grid.best_params_, "models/best_params.pkl")