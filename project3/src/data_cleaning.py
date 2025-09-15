import pandas as pd
import os

def clean_data(file_path):
    """
    Cleans the financial data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A cleaned and transformed DataFrame.
    """
    # Construct the full path using os.path.join for cross-platform compatibility
    full_path = os.path.join(file_path)

    # 1. Load the data
    try:
        df = pd.read_csv(full_path)
    except FileNotFoundError:
        print(f"Error: The file at {full_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # 2. Handle missing values
    # Dropping rows with any missing values as they could affect financial analysis
    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with missing values.")
    
    # 3. Correct data types and transform data
    # Convert 'date' column to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Set 'date' as the index for time series analysis
    df.set_index('date', inplace=True)

    # Correct erroneous data types
    # Ensure all numerical columns are of the float type
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove any potential duplicate entries based on the date index
    df = df[~df.index.duplicated(keep='first')]

    # 4. Optional: Rename columns for better readability and consistency
    df.columns = [col.replace('_', ' ').title().replace(' ', '') for col in df.columns]

    print("Data cleaning complete.")
    return df