
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================================================
# 1. FILE PATHS - KAGGLE DATASET ONLY
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "delivery_data.csv"
MODEL_PATH = BASE_DIR / "models" / "delivery_time_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Kaggle dataset not found: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run python train_model.py first."
    )

# =========================================================
# 2. LOAD DATA AND SAVED PIPELINE
# =========================================================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

model = joblib.load(MODEL_PATH)

features = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

target = "Delivery_Time_min"

required_columns = features + [target]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required Kaggle columns: {missing_columns}"
    )

# Convert numeric columns; invalid entries become missing.
numeric_columns = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Match the target filtering used during training.
df = df.dropna(subset=[target]).copy()
df = df[df[target] > 0].copy()

# Mark impossible negative numeric input values as missing.
for col in [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]:
    df.loc[df[col] < 0, col] = np.nan

# Blank categorical values must be treated as missing.
categorical_columns = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

for col in categorical_columns:
    df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

if len(df) < 10:
    raise ValueError("Not enough valid target records to evaluate.")

X = df[features].copy()
y = df[target].copy()

# IMPORTANT:
# Keep the exact split settings used in train_model.py.
# Do not drop rows just because input features are missing.
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# The saved pipeline performs imputation and encoding.
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("=" * 55)
print("SMARTLOGIX - MODEL EVALUATION")
print("=" * 55)
print("Dataset: data/delivery_data.csv")
print("Model: models/delivery_time_model.pkl")
print("Training/test split: 80% / 20%")
print("Random state: 42")
print(f"Test records: {len(y_test)}")
print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R2 score: {r2:.3f}")

# =========================================================
# 3. SAVE EVALUATION RESULTS
# =========================================================
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
})

results["Error"] = results["Actual"] - results["Predicted"]

results.to_csv(
    OUTPUT_DIR / "model_evaluation.csv",
    index=False,
)

metrics = pd.DataFrame([{
    "Dataset": "Kaggle delivery-time dataset",
    "Model_File": "delivery_time_model.pkl",
    "Test_Records": len(y_test),
    "MAE_minutes": mae,
    "RMSE_minutes": rmse,
    "R2_Score": r2,
}])

metrics.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False,
)

sns.set_theme(style="whitegrid")

# =========================================================
# 4. CHART 1: ACTUAL VS PREDICTED
# =========================================================
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    results["Actual"],
    results["Predicted"],
    alpha=0.65,
)

low = min(
    results["Actual"].min(),
    results["Predicted"].min(),
)
high = max(
    results["Actual"].max(),
    results["Predicted"].max(),
)

ax.plot(
    [low, high],
    [low, high],
    linestyle="--",
)

ax.set_xlabel("Actual Delivery Time (minutes)")
ax.set_ylabel("Predicted Delivery Time (minutes)")
ax.set_title("Actual vs Predicted Delivery Time")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150,
)
plt.close(fig)

# =========================================================
# 5. CHART 2: RESIDUAL ERROR DISTRIBUTION
# =========================================================
fig, ax = plt.subplots(figsize=(7, 5))

sns.histplot(
    results["Error"],
    bins=25,
    kde=True,
    ax=ax,
)

ax.axvline(0, linestyle="--")
ax.set_xlabel("Residual Error: Actual - Predicted (minutes)")
ax.set_ylabel("Number of Deliveries")
ax.set_title("Distribution of Prediction Errors")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=150,
)
plt.close(fig)

# =========================================================
# 6. CHART 3: RESIDUALS VS PREDICTED VALUES
# =========================================================
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    results["Predicted"],
    results["Error"],
    alpha=0.65,
)

ax.axhline(0, linestyle="--")
ax.set_xlabel("Predicted Delivery Time (minutes)")
ax.set_ylabel("Residual Error (minutes)")
ax.set_title("Residuals vs Predicted Values")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "residuals_vs_predicted.png",
    dpi=150,
)
plt.close(fig)

# =========================================================
# 7. COMPLETION SUMMARY
# =========================================================
print("\nEvaluation completed successfully.")
print("\nSaved evaluation files:")

for filename in [
    "model_evaluation.csv",
    "evaluation_metrics.csv",
    "actual_vs_predicted.png",
    "residual_distribution.png",
    "residuals_vs_predicted.png",
]:
    print(f"- outputs/{filename}")

print("\nThe original Kaggle dataset was not modified.")
