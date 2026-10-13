
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartLogix | Logistics Analytics",
    page_icon="🚚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "delivery_data.csv"
MODEL_PATH = BASE_DIR / "models" / "delivery_time_model.pkl"

TARGET = "Delivery_Time_min"

NUMERIC_FEATURES = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

CATEGORICAL_FEATURES = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]


# =========================================================
# 2. LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Convert target to numeric.
    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    # Convert numeric input features.
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        data[column] = data[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        data.loc[data[column] < 0, column] = np.nan

    # Normalize categorical values and use np.nan for missing data.
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].apply(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else np.nan
            )
        )

        data[column] = data[column].replace(
            "",
            np.nan,
        )

    # Match the target filtering used during training.
    data = data.dropna(subset=[TARGET]).copy()
    data = data[data[TARGET] > 0].copy()

    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()

except Exception as error:
    st.error(f"Unable to load SmartLogix: {error}")
    st.info(
        "Check that data/delivery_data.csv and "
        "models/delivery_time_model.pkl exist."
    )
    st.stop()


# Complete cases for descriptive charts only.
analytics_df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()


# =========================================================
# 3. SHARED CATEGORY OPTIONS
# =========================================================

def get_options(column, defaults):
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    options = sorted(
        value
        for value in values.unique().tolist()
        if value != ""
    )

    return options if options else defaults


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🚚 SmartLogix")
st.sidebar.caption("Smart Logistics Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Delivery Prediction",
        "Data Explorer",
        "Model Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.caption("AI-powered delivery time analysis")


# =========================================================
# 5. OVERVIEW
# =========================================================

if page == "Overview":

    st.title("SmartLogix Dashboard")

    st.write(
        "Analyze delivery patterns, explore logistics data, "
        "and predict delivery times using machine learning."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Valid Deliveries", f"{len(df):,}")
    col2.metric("Complete Records", f"{len(analytics_df):,}")

    col3.metric(
        "Average Delivery Time",
        f"{df[TARGET].mean():.2f} min",
    )

    col4.metric(
        "Average Distance",
        f"{df['Distance_km'].mean():.2f} km",
    )

    st.divider()

    if analytics_df.empty:
        st.warning(
            "No complete records are available for these charts."
        )

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Delivery Time Distribution")

            fig = px.histogram(
                analytics_df,
                x=TARGET,
                nbins=30,
                title="Distribution of Delivery Times",
                labels={
                    TARGET: "Delivery Time (minutes)"
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:
            st.subheader("Average Delivery Time by Traffic")

            traffic_summary = (
                analytics_df.groupby(
                    "Traffic_Level",
                    as_index=False,
                )[TARGET]
                .mean()
                .sort_values(TARGET)
            )

            fig = px.bar(
                traffic_summary,
                x="Traffic_Level",
                y=TARGET,
                title="Traffic Level vs Delivery Time",
                labels={
                    "Traffic_Level": "Traffic Level",
                    TARGET: "Average Delivery Time (minutes)",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Deliveries by Weather")

            weather_summary = (
                analytics_df["Weather"]
                .value_counts()
                .rename_axis("Weather")
                .reset_index(name="Number of Deliveries")
            )

            fig = px.pie(
                weather_summary,
                names="Weather",
                values="Number of Deliveries",
                title="Weather Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:
            st.subheader("Delivery Time vs Distance")

            fig = px.scatter(
                analytics_df,
                x="Distance_km",
                y=TARGET,
                color="Vehicle_Type",
                title="Distance and Delivery Time",
                labels={
                    "Distance_km": "Distance (km)",
                    TARGET: "Delivery Time (minutes)",
                    "Vehicle_Type": "Vehicle",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.caption(
        "Complete records exclude rows with missing feature values. "
        "The original Kaggle CSV is not modified."
    )


# =========================================================
# 6. DELIVERY PREDICTION
# =========================================================

elif page == "Delivery Prediction":

    st.title("Delivery Time Prediction")

    st.write(
        "Enter delivery details to estimate delivery time "
        "using the trained machine learning model."
    )

    weather_options = get_options(
        "Weather",
        ["Clear", "Rainy", "Foggy", "Stormy"],
    )

    traffic_options = get_options(
        "Traffic_Level",
        ["Low", "Medium", "High"],
    )

    time_options = get_options(
        "Time_of_Day",
        ["Morning", "Afternoon", "Evening", "Night"],
    )

    vehicle_options = get_options(
        "Vehicle_Type",
        ["Bike", "Scooter", "Car"],
    )

    with st.form("prediction_form"):

        st.subheader("Delivery Information")

        col1, col2 = st.columns(2)

        with col1:
            distance = st.number_input(
                "Distance (km)",
                min_value=0.1,
                max_value=1000.0,
                value=5.0,
                step=0.5,
            )

            weather = st.selectbox(
                "Weather",
                weather_options,
            )

            traffic = st.selectbox(
                "Traffic Level",
                traffic_options,
            )

            time_of_day = st.selectbox(
                "Time of Day",
                time_options,
            )

        with col2:
            vehicle = st.selectbox(
                "Vehicle Type",
                vehicle_options,
            )

            preparation_time = st.number_input(
                "Preparation Time (minutes)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=1.0,
            )

            experience = st.number_input(
                "Courier Experience (years)",
                min_value=0.0,
                max_value=80.0,
                value=2.0,
                step=0.5,
            )

        submitted = st.form_submit_button(
            "Predict Delivery Time",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        input_data = pd.DataFrame(
            [
                {
                    "Distance_km": distance,
                    "Weather": weather,
                    "Traffic_Level": traffic,
                    "Time_of_Day": time_of_day,
                    "Vehicle_Type": vehicle,
                    "Preparation_Time_min": preparation_time,
                    "Courier_Experience_yrs": experience,
                }
            ],
            columns=FEATURES,
        )

        try:
            prediction = float(
                model.predict(input_data)[0]
            )

            if not np.isfinite(prediction):
                st.error(
                    "The model returned an invalid prediction."
                )

            else:
                prediction = max(0.0, prediction)

                st.success("Prediction completed!")

                st.metric(
                    "Estimated Delivery Time",
                    f"{prediction:.2f} minutes",
                )

                if prediction >= 60:
                    st.info(
                        f"Estimated time: {prediction / 60:.2f} hours."
                    )
                else:
                    st.info(
                        "Estimated time: approximately "
                        f"{round(prediction)} minutes."
                    )

                st.caption(
                    "This is an estimate, not a guaranteed delivery time."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# 7. DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.title("Data Explorer")

    st.write(
        "Explore the Kaggle delivery dataset, filter records, "
        "and inspect missing values."
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{len(df.columns):,}")

    col3.metric(
        "Rows with Missing Values",
        f"{int(df.isna().any(axis=1).sum()):,}",
    )

    with st.expander("View missing-value report"):

        missing_report = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Values": df.isna().sum().values,
                "Missing Percentage": (
                    df.isna().mean().values * 100
                ).round(2),
            }
        )

        missing_report = missing_report.sort_values(
            "Missing Values",
            ascending=False,
        )

        st.dataframe(
            missing_report,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Filter Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        weather_options = get_options(
            "Weather",
            ["Clear", "Rainy", "Foggy", "Stormy"],
        )

        selected_weather = st.multiselect(
            "Weather",
            options=weather_options,
            default=weather_options,
        )

    with col2:
        traffic_options = get_options(
            "Traffic_Level",
            ["Low", "Medium", "High"],
        )

        selected_traffic = st.multiselect(
            "Traffic Level",
            options=traffic_options,
            default=traffic_options,
        )

    with col3:
        vehicle_options = get_options(
            "Vehicle_Type",
            ["Bike", "Scooter", "Car"],
        )

        selected_vehicle = st.multiselect(
            "Vehicle Type",
            options=vehicle_options,
            default=vehicle_options,
        )

    filtered_df = df.copy()

    # Include missing-category records when filtering.
    if selected_weather:
        filtered_df = filtered_df[
            filtered_df["Weather"].isna()
            | filtered_df["Weather"].isin(selected_weather)
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["Weather"].isna()
        ]

    if selected_traffic:
        filtered_df = filtered_df[
            filtered_df["Traffic_Level"].isna()
            | filtered_df["Traffic_Level"].isin(selected_traffic)
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["Traffic_Level"].isna()
        ]

    if selected_vehicle:
        filtered_df = filtered_df[
            filtered_df["Vehicle_Type"].isna()
            | filtered_df["Vehicle_Type"].isin(selected_vehicle)
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["Vehicle_Type"].isna()
        ]

    st.write(
        f"Showing **{len(filtered_df):,}** records."
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Filtered CSV",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="smartlogix_filtered_data.csv",
        mime="text/csv",
    )


# =========================================================
# 8. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    st.write(
        "Evaluate the saved model on the held-out test split."
    )

    # Prepare the evaluation inputs.
    # Missing input values remain missing for pipeline imputation.
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].apply(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else np.nan
            )
        )

        X[column] = X[column].replace(
            "",
            np.nan,
        )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    try:
        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{mae:.2f} min")
        col2.metric("RMSE", f"{rmse:.2f} min")
        col3.metric("R² Score", f"{r2:.3f}")

        st.caption(
            f"Training records: {len(X_train):,} | "
            f"Test records: {len(X_test):,}"
        )

        with st.expander("Understand the metrics"):

            st.markdown(
                """
                - **MAE:** Average absolute prediction error in minutes.
                  Lower is better.
                - **RMSE:** Penalizes larger errors more heavily.
                  Lower is better.
                - **R² Score:** Measures how well the model explains
                  variation in delivery time. A score closer to 1 is
                  generally better.
                """
            )

        results_df = pd.DataFrame(
            {
                "Actual Delivery Time": y_test.to_numpy(),
                "Predicted Delivery Time": predictions,
            }
        )

        results_df["Residual"] = (
            results_df["Actual Delivery Time"]
            - results_df["Predicted Delivery Time"]
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted")

            fig = px.scatter(
                results_df,
                x="Actual Delivery Time",
                y="Predicted Delivery Time",
                title="Actual vs Predicted Delivery Times",
                labels={
                    "Actual Delivery Time": "Actual (minutes)",
                    "Predicted Delivery Time": "Predicted (minutes)",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:
            st.subheader("Residual Distribution")

            fig = px.histogram(
                results_df,
                x="Residual",
                nbins=30,
                title="Distribution of Prediction Errors",
                labels={
                    "Residual": "Actual - Predicted (minutes)"
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader("Residuals vs Predicted Values")

        fig = px.scatter(
            results_df,
            x="Predicted Delivery Time",
            y="Residual",
            title="Residuals vs Predicted Delivery Time",
            labels={
                "Predicted Delivery Time": "Predicted (minutes)",
                "Residual": "Actual - Predicted (minutes)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Test Predictions")

        st.dataframe(
            results_df.round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download Test Predictions",
            data=results_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="smartlogix_test_predictions.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(
            f"Model evaluation failed: {error}"
        )

        st.info(
            "If the error persists, the preprocessing steps inside "
            "the saved model may differ from the current dataset "
            "preparation. Check train_model.py and retrain only "
            "if necessary."
        )
