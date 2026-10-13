
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. PATHS - KAGGLE DATASET ONLY
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "delivery_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Kaggle dataset not found: {DATA_PATH}"
    )


# =========================================================
# 2. LOAD AND VALIDATE DATA
# =========================================================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required_columns = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

# Target must be numeric and cannot be missing.
df["Delivery_Time_min"] = pd.to_numeric(
    df["Delivery_Time_min"], errors="coerce"
)

df = df.dropna(subset=["Delivery_Time_min"]).copy()

# Remove records with impossible target values.
df = df[df["Delivery_Time_min"] > 0].copy()

if len(df) < 10:
    raise ValueError("Not enough valid records to train the model.")

# Convert numerical input columns safely.
numeric_features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Treat blank categorical values as missing.
categorical_features = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

for col in categorical_features:
    df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

# Remove impossible negative input values by marking them missing.
for col in numeric_features:
    df.loc[df[col] < 0, col] = np.nan

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["Delivery_Time_min"].copy()

# Split before fitting imputers and encoders.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("=" * 60)
print("SMARTLOGIX - MODEL TRAINING")
print("=" * 60)
print("Dataset: data/delivery_data.csv")
print("Valid target records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Target: Delivery_Time_min")
print("Missing values will be imputed within each training pipeline.")


# =========================================================
# 3. PREPROCESSING PIPELINE
# =========================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


# =========================================================
# 4. COMPARE MODELS
# =========================================================
models = {
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

for model_name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append(
        {
            "Model": model_name,
            "MAE (minutes)": round(mae, 3),
            "RMSE (minutes)": round(rmse, 3),
            "R2 Score": round(r2, 3),
        }
    )

    trained_models[model_name] = pipeline


# =========================================================
# 5. SELECT BEST MODEL
# =========================================================
results_df = pd.DataFrame(results).sort_values(
    by="MAE (minutes)",
    ascending=True,
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

model_path = MODEL_DIR / "delivery_time_model.pkl"
joblib.dump(best_model, model_path)

print("\nBest model:", best_name)
print("Saved complete pipeline to:", model_path)


# =========================================================
# 6. SAVE TEST PREDICTIONS AND METRICS
# =========================================================
test_predictions = best_model.predict(X_test)

predictions_df = pd.DataFrame(
    {
        "Actual_Delivery_Time_min": y_test.to_numpy(),
        "Predicted_Delivery_Time_min": test_predictions,
        "Prediction_Error_min": (
            y_test.to_numpy() - test_predictions
        ),
    }
)

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

metrics = {
    "Dataset": "Kaggle delivery-time dataset",
    "Selected_Model": best_name,
    "Training_Rows": len(X_train),
    "Testing_Rows": len(X_test),
    "MAE_minutes": mean_absolute_error(y_test, test_predictions),
    "RMSE_minutes": np.sqrt(
        mean_squared_error(y_test, test_predictions)
    ),
    "R2_Score": r2_score(y_test, test_predictions),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False,
)

print("\nFINAL TEST METRICS")
for key, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nTraining completed successfully.")
print("The original Kaggle dataset was not modified.")
print("The saved model includes preprocessing and imputation.")
