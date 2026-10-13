
# SmartLogix 🚚
### Smart Logistics Analytics & Delivery Time Prediction

SmartLogix is a machine learning project that analyzes delivery data and predicts estimated delivery time based on distance, weather, traffic conditions, time of day, vehicle type, preparation time, and courier experience.

The project combines data analysis, machine learning, and an interactive Streamlit dashboard to explore delivery patterns and estimate delivery durations.

## Key Features

- **Delivery Time Prediction:** Predict estimated delivery duration using a trained machine learning model.
- **Interactive Dashboard:** View delivery statistics and visualizations.
- **Data Explorer:** Filter delivery records and download filtered data as a CSV file.
- **Model Evaluation:** Evaluate predictions using MAE, RMSE, and R².
- **Data Quality Analysis:** Inspect missing values and explore the dataset.
- **Visual Analytics:** Explore delivery-time distributions, traffic patterns, weather, and distance.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

## Machine Learning

The project uses a trained Random Forest regression model to estimate delivery time.

The model uses these input features:

- Distance in kilometres
- Weather conditions
- Traffic level
- Time of day
- Vehicle type
- Preparation time in minutes
- Courier experience in years

### Model Performance

Results from the current held-out test evaluation:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 6.66 minutes |
| Root Mean Squared Error (RMSE) | 9.56 minutes |
| R² Score | 0.796 |

**Interpretation:** The model's predictions have an average absolute error of approximately 6.66 minutes on the evaluated test data. The R² score indicates that the model explains approximately 79.6% of the variation in delivery time on that test split.

These results depend on the dataset, training configuration, and evaluation procedure.

## Project Structure

```text
SmartLogix/
├── app.py
├── train_model.py
├── analyze_data.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── data/
│   └── delivery_data.csv
├── models/
│   └── delivery_time_model.pkl
└── outputs/
    ├── model_metrics.csv
    ├── model_evaluation.csv
    ├── evaluation_metrics.csv
    ├── test_predictions.csv
    ├── actual_vs_predicted.png
    ├── residual_distribution.png
    └── residuals_vs_predicted.png
```

The files listed under `outputs/` may vary depending on which analysis and evaluation scripts have been run.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SmartLogix
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Training

To train the model using the Kaggle dataset:

```bash
python train_model.py
```

To run the data analysis:

```bash
python analyze_data.py
```

To evaluate the model separately:

```bash
python evaluate_model.py
```

Run the training script only when you intend to retrain the model.

## Dataset

The project uses the Kaggle delivery dataset stored at:

`data/delivery_data.csv`

The dataset contains delivery-related information, including distance, weather, traffic, vehicle type, preparation time, courier experience, and actual delivery time.

Ensure you have permission to redistribute the dataset before including it in a public repository. If redistribution is not permitted, document the original dataset source and provide instructions for obtaining it.

## Limitations

- Predictions are estimates and do not guarantee actual delivery times.
- Real-world results may differ due to road conditions, route changes, and other factors not included in the dataset.
- Performance on new delivery data may differ from the reported test results.
- The project is a predictive analytics prototype, not a live delivery-tracking system.

## Future Improvements

- Integrate real-time traffic and weather data.
- Add route optimization.
- Compare additional regression algorithms.
- Improve monitoring of prediction errors.
- Deploy the application for public access.

## Author

**Your Name**

BCA Student | Aspiring Data Science Professional

Skills demonstrated: Python, Data Analysis, Data Visualization, Machine Learning, and Streamlit.
