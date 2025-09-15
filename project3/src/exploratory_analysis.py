import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path):
    """
    Performs Exploratory Data Analysis (EDA) on the cleaned financial data.

    This includes:
    - Displaying basic data information and descriptive statistics.
    - Analyzing trends in stock prices and trading volume over time.
    - Saving key visualizations to the 'visualizations' folder.

    Args:
        file_path (str): The path to the cleaned CSV file.

    Returns:
        None
    """
    # 1. Load the cleaned data
    try:
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        print("Data loaded successfully for EDA.")
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # 2. Display basic information and descriptive statistics
    print("\n--- Basic Data Information ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    # Create the visualizations directory if it doesn't exist
    viz_dir = 'visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"\nCreated directory: {viz_dir}")

    # 3. Trend Analysis and Visualization
    print("\n--- Generating Visualizations ---")

    # Trend 1: Stock Price Trend over time
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['ClosePrice'], label='Closing Price', color='b')
    plt.title('GME Stock Closing Price Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'stock_price_trend.png'))
    plt.show()
    print(f"Saved 'stock_price_trend.png' to the {viz_dir} folder.")

    # Trend 2: Trading Volume Analysis
    plt.figure(figsize=(14, 7))
    plt.bar(df.index, df['Volume'], color='g')
    plt.title('GME Daily Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'trading_volume.png'))
    plt.show()
    print(f"Saved 'trading_volume.png' to the {viz_dir} folder.")

    # 4. Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Financial Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'))
    plt.show()
    print(f"Saved 'correlation_matrix.png' to the {viz_dir} folder.")

    print("\nEDA complete. All visualizations have been generated and saved.")