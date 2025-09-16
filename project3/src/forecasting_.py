import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from statsmodels.tsa.arima.model import ARIMA

def train_and_forecast_arima(file_path, forecast_steps=30):
    """
    Trains an ARIMA model on historical stock price data and forecasts future prices.

    Args:
        file_path (str): The path to the cleaned financial data CSV.
        forecast_steps (int): The number of future days to forecast.

    Returns:
        tuple: A tuple containing the fitted model and the forecast results.
    """
    # 1. Load the cleaned data
    try:
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        print("Cleaned data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None, None
    
    # Use the ClosePrice for forecasting
    ts = df['ClosePrice']
    
    # 2. Fit the ARIMA model
    # Note: The (5,1,0) order is a simple example. In a real-world application,
    # you would use techniques like Auto-ARIMA or grid search to find the optimal parameters.
    try:
        model = ARIMA(ts, order=(5, 1, 0))
        fitted_model = model.fit()
        print("\nARIMA model has been successfully trained.")
        
    except Exception as e:
        print(f"An error occurred during model fitting: {e}")
        return None, None

    # 3. Forecast future values
    forecast_result = fitted_model.get_forecast(steps=forecast_steps)
    forecast_values = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int()
    
    print(f"\nForecast for the next {forecast_steps} steps generated.")

    # 4. Visualize the forecast
    plt.figure(figsize=(15, 7))
    plt.plot(ts, label='Historical Data', color='blue')
    plt.plot(forecast_values, label=f'Forecasted ({forecast_steps} days)', color='red')
    plt.fill_between(confidence_intervals.index,
                     confidence_intervals.iloc[:, 0],
                     confidence_intervals.iloc[:, 1],
                     color='pink', alpha=0.3, label='Confidence Interval')
    plt.title(f'GME Stock Price Forecast using ARIMA({forecast_steps} days)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Create the visualizations directory if it doesn't exist
    viz_dir = 'visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    plt.savefig(os.path.join(viz_dir, 'arima_forecast_plot.png'))
    plt.show()
    print(f"Saved forecast plot to the '{viz_dir}' folder.")

    # 5. Save the trained model as a .pkl file
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        
    model_filename = 'arima_model.pkl'
    model_path = os.path.join(models_dir, model_filename)
    
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(fitted_model, file)
        print(f"Successfully saved the trained model as '{model_filename}' in the '{models_dir}' folder.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        
    return fitted_model, forecast_result