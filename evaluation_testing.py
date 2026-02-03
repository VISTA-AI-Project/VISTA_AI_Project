import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(file_path='training_set_predictions.csv'):
    """
    Loads prediction data and calculates MAE, MSE, and RMSE for Temperature,
    Precipitation, and WindSpeed.

    Args:
        file_path (str): The path to the CSV file containing the predictions.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Define the columns for evaluation: (Actual, Predicted)
    metrics_config = {
        'Temperature': ('Temperature', 'Temp_Pred'),
        'Precipitation': ('Precipitation', 'Precip_Pred'),
        'WindSpeed': ('WindSpeed', 'Wind_Pred')
    }

    results = {}

    # 2. Calculate metrics for each variable
    for var, (actual_col, pred_col) in metrics_config.items():
        actual = df[actual_col]
        predicted = df[pred_col]

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual, predicted)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(actual, predicted)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        results[var] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    # 3. Convert results to a DataFrame for clear presentation
    results_df = pd.DataFrame.from_dict(results, orient='index')

    # Use to_string() instead of to_markdown() to avoid the 'tabulate' dependency
    print("\n--- Model Evaluation Results (MAE, MSE, RMSE) ---")
    print(results_df.round(4).to_string())
    print("-" * 50)
    


if _name_ == '_main_':
    # Run the evaluation script
    evaluate_predictions()