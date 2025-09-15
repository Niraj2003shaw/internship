"""
Main script to orchestrate the financial analysis and forecasting pipeline.

This script serves as the primary entry point for the project. It imports and executes functions
from various modules in the `src` directory to perform the following steps:
1.  **Data Cleaning**: Loads the raw financial data (e.g., GME_stock.csv) and cleans it.
2.  **Exploratory Data Analysis (EDA)**: Analyzes the cleaned data to identify trends and generates visualizations.
3.  **Forecasting**: Applies time series models to predict future stock prices or financial metrics.
4.  **Dashboard Creation**: Uses the processed data and insights to build an interactive dashboard.

The workflow is executed sequentially to ensure each step is completed before proceeding
to the next.
"""