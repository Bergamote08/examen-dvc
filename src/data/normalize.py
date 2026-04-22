import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save
pd.DataFrame(X_train_scaled).to_csv("data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("data/processed_data/X_test_scaled.csv", index=False)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")