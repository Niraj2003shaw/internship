"""
This script serves as the main application entry point (app.py)
to orchestrate the data cleaning, exploratory data analysis (EDA),
and time series forecasting pipeline.
"""

import os
import pandas as pd
from data_cleaning import clean_data
from exploratory_analysis import perform_eda
from forecasting_ import train_and_forecast_arima

# Define the absolute paths for the input and output files
RAW_DATA_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3", "GME_stock.csv")
CLEANED_DATA_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3\\src", "cleaned_dataSP.csv")
FORECAST_RESULTS_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3\\src", "forecast_results.csv")

if __name__ == "__main__":
    print("--- Starting the financial analysis pipeline ---")

    # Step 1: Data Cleaning
    print("\nStarting data cleaning process...")
    cleaned_df = clean_data(RAW_DATA_PATH)

    if cleaned_df is not None:
        print("Data cleaning successful!")
        
        # Save the cleaned DataFrame
        try:
            cleaned_df.to_csv(CLEANED_DATA_PATH)
            print(f"Successfully saved cleaned data to {CLEANED_DATA_PATH}")
            
            # Step 2: Exploratory Data Analysis (EDA)
            print("\nStarting Exploratory Data Analysis (EDA)...")
            perform_eda(CLEANED_DATA_PATH)
            
            # Step 3: Forecasting
            print("\nStarting time series forecasting...")
            fitted_model, forecast_result = train_and_forecast_arima(CLEANED_DATA_PATH)
            
            if forecast_result is not None:
                # Save the forecast results to a CSV file
                try:
                    forecast_df = forecast_result.summary_frame()
                    forecast_df.to_csv(FORECAST_RESULTS_PATH)
                    print(f"Successfully saved forecast results to {FORECAST_RESULTS_PATH}")
                except Exception as e:
                    print(f"An error occurred while saving forecast results: {e}")
            
            print("\n--- Financial analysis pipeline completed successfully ---")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

    else:
        print("\nData cleaning failed. Subsequent steps will not run.")