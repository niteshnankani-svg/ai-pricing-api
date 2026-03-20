import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# ============================================
# STEP 1 — CREATE FAKE HISTORICAL DATA
# This simulates 500 past pricing decisions
# In real life this comes from your database
# ============================================

np.random.seed(42)
n_samples = 500

data = {
    "cost": np.random.uniform(5, 100, n_samples),
    "demand_score": np.random.randint(1, 11, n_samples),
    "competitor_price": np.random.uniform(20, 300, n_samples),
    "season": np.random.choice([0, 1, 2], n_samples),  # 0=regular, 1=holiday, 2=new_year
}

df = pd.DataFrame(data)

# The "correct" price our model learns from
df["optimal_price"] = (
    df["cost"] * 1.3 +
    df["demand_score"] * 2.5 +
    df["competitor_price"] * 0.3 +
    df["season"] * 3 +
    np.random.normal(0, 2, n_samples)  # adds realistic noise
)

print("Training data sample:")
print(df.head())
print(f"\nTotal records: {len(df)}")

# ============================================
# STEP 2 — TRAIN THE ML MODEL
# ============================================

X = df[["cost", "demand_score", "competitor_price", "season"]]
y = df["optimal_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# ============================================
# STEP 3 — TEST THE MODEL ACCURACY
# ============================================

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nModel trained successfully!")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"This means predictions are off by ~${mae:.2f} on average")

# ============================================
# STEP 4 — SAVE THE MODEL TO A FILE
# ============================================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to model.pkl")
print("Ready to use in Flask API!")
