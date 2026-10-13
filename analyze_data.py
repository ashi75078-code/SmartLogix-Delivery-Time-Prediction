
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# =========================================================
# 1. FILE PATHS - KAGGLE DATASET ONLY
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "delivery_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Kaggle dataset not found: {DATA_PATH}\n"
        "Place delivery_data.csv inside the data folder."
    )

# =========================================================
# 2. LOAD DATASET
# =========================================================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required_columns = [
    "Order_ID",
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
        f"Missing required columns in Kaggle dataset: {missing_columns}"
    )

numeric_columns = [
    "Order_ID",
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_columns = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

for col in categorical_columns:
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("SMARTLOGIX - KAGGLE DATA ANALYSIS")
print("=" * 60)

# =========================================================
# 3. DATASET OVERVIEW
# =========================================================
print("\n1. DATASET OVERVIEW")
print("Dataset:", DATA_PATH.name)
print("Dataset shape:", df.shape)
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head().to_string(index=False))

print("\nData types:")
df.info()

# =========================================================
# 4. DATA QUALITY AND MISSING VALUES
# =========================================================
print("\n2. DATA QUALITY")

missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().sum() / max(len(df), 1) * 100
    ).round(2),
})

print("\nMissing Value Report:")
print(missing_report.to_string())

missing_rows = int(df.isnull().any(axis=1).sum())
complete_rows = int(df.notnull().all(axis=1).sum())

print("\nTotal rows:", len(df))
print("Rows with any missing value:", missing_rows)
print("Rows with no missing values:", complete_rows)
print("Duplicate complete rows:", int(df.duplicated().sum()))

print(
    "Duplicate Order_ID values:",
    int(df["Order_ID"].dropna().duplicated().sum()),
)

# Save the missing-value report
missing_report.to_csv(
    OUTPUT_DIR / "missing_value_report.csv"
)

# Visualize missing values
plt.figure(figsize=(10, 5))
missing_percent = (
    df.isnull().mean().mul(100).sort_values(ascending=False)
)

sns.barplot(
    x=missing_percent.values,
    y=missing_percent.index,
)

plt.title("Missing Values by Column")
plt.xlabel("Missing Values (%)")
plt.ylabel("Column")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "missing_values.png",
    dpi=150,
)
plt.close()

# =========================================================
# 5. INVALID VALUE CHECKS
# =========================================================
print("\n3. INVALID VALUE CHECKS")

range_rules = {
    "Distance_km": 0,
    "Preparation_Time_min": 0,
    "Courier_Experience_yrs": 0,
    "Delivery_Time_min": 0,
}

invalid_counts = {}

for col, minimum in range_rules.items():
    invalid_counts[col] = int(
        (df[col].notna() & (df[col] < minimum)).sum()
    )

for col, count in invalid_counts.items():
    print(f"{col} below {range_rules[col]}: {count}")

print("\nCategorical value counts:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

# =========================================================
# 6. PREPARE ANALYSIS DATA
# =========================================================
analysis_columns = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

analysis_df = df.dropna(
    subset=analysis_columns
).copy()

# Remove impossible negative measurements.
# Zero preparation time and zero courier experience are allowed.
for col in [
    "Distance_km",
    "Delivery_Time_min",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]:
    analysis_df = analysis_df[analysis_df[col] >= 0]

# A distance or delivery duration of zero is not useful here.
analysis_df = analysis_df[
    (analysis_df["Distance_km"] > 0)
    & (analysis_df["Delivery_Time_min"] > 0)
]

print("\n4. ANALYSIS DATA SUMMARY")
print("Original records:", len(df))
print("Records used for analysis:", len(analysis_df))
print("Records excluded:", len(df) - len(analysis_df))

if analysis_df.empty:
    raise ValueError(
        "No valid records remain after data-quality checks."
    )

# =========================================================
# 7. STATISTICAL SUMMARY
# =========================================================
print("\n5. NUMERICAL SUMMARY")
print(
    analysis_df.select_dtypes(include="number")
    .describe()
    .round(2)
    .to_string()
)

print("\n6. AVERAGE DELIVERY TIME BY TRAFFIC")
print(
    analysis_df.groupby("Traffic_Level")["Delivery_Time_min"]
    .agg(["count", "mean", "median"])
    .round(2)
    .to_string()
)

print("\n7. AVERAGE DELIVERY TIME BY WEATHER")
print(
    analysis_df.groupby("Weather")["Delivery_Time_min"]
    .agg(["count", "mean", "median"])
    .round(2)
    .to_string()
)

print("\n8. AVERAGE DELIVERY TIME BY VEHICLE")
print(
    analysis_df.groupby("Vehicle_Type")["Delivery_Time_min"]
    .agg(["count", "mean", "median"])
    .round(2)
    .to_string()
)

# =========================================================
# 9. CHART 1: DELIVERY TIME DISTRIBUTION
# =========================================================
plt.figure(figsize=(8, 5))

sns.histplot(
    data=analysis_df,
    x="Delivery_Time_min",
    bins=30,
    kde=True,
)

plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Deliveries")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "delivery_time_distribution.png",
    dpi=150,
)
plt.close()

# =========================================================
# 10. CHART 2: DELIVERY TIME BY TRAFFIC
# =========================================================
traffic_summary = (
    analysis_df.groupby("Traffic_Level", as_index=False)
    ["Delivery_Time_min"]
    .mean()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=traffic_summary,
    x="Traffic_Level",
    y="Delivery_Time_min",
    order=[
        level for level in ["Low", "Medium", "High"]
        if level in traffic_summary["Traffic_Level"].values
    ],
)

plt.title("Average Delivery Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "traffic_delivery_time.png",
    dpi=150,
)
plt.close()

# =========================================================
# 11. CHART 3: DELIVERY TIME BY WEATHER
# =========================================================
weather_summary = (
    analysis_df.groupby("Weather", as_index=False)
    ["Delivery_Time_min"]
    .mean()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=weather_summary,
    x="Weather",
    y="Delivery_Time_min",
)

plt.title("Average Delivery Time by Weather")
plt.xlabel("Weather Conditions")
plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "weather_delivery_time.png",
    dpi=150,
)
plt.close()

# =========================================================
# 12. CHART 4: DISTANCE VS DELIVERY TIME
# =========================================================
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=analysis_df,
    x="Distance_km",
    y="Delivery_Time_min",
    hue="Traffic_Level",
    alpha=0.7,
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "distance_delivery_time.png",
    dpi=150,
)
plt.close()

# =========================================================
# 13. CHART 5: DELIVERY TIME BY VEHICLE
# =========================================================
vehicle_summary = (
    analysis_df.groupby("Vehicle_Type", as_index=False)
    ["Delivery_Time_min"]
    .mean()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=vehicle_summary,
    x="Vehicle_Type",
    y="Delivery_Time_min",
)

plt.title("Average Delivery Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "vehicle_delivery_time.png",
    dpi=150,
)
plt.close()

# =========================================================
# 14. CHART 6: CORRELATION HEATMAP
# =========================================================
numeric_features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

numeric_df = analysis_df[numeric_features]

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
)

plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=150,
)
plt.close()

# =========================================================
# 15. SAVE ANALYSIS DATA
# =========================================================
analysis_df.to_csv(
    OUTPUT_DIR / "kaggle_delivery_analysis.csv",
    index=False,
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Dataset used: {DATA_PATH}")
print(f"Original records: {len(df)}")
print(f"Records analyzed: {len(analysis_df)}")
print(f"Output folder: {OUTPUT_DIR}")
print("\nGenerated files:")
print("- missing_value_report.csv")
print("- missing_values.png")
print("- delivery_time_distribution.png")
print("- traffic_delivery_time.png")
print("- weather_delivery_time.png")
print("- distance_delivery_time.png")
print("- vehicle_delivery_time.png")
print("- correlation_heatmap.png")
print("- kaggle_delivery_analysis.csv")
print("\nNo synthetic data was generated.")
