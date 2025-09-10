import pandas as pd
from io import StringIO
from datetime import datetime



# Define the file path for the raw data
FILE_PATH = r"D:\Niraj\OneDrive\Desktop\internship\customer_segmentation_data.csv"

def load_data_from_file():
    """
    Loads data from the specified file path into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(FILE_PATH)
        print("Data loaded successfully from file.")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {FILE_PATH}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def standardize_column_names(df):
    """
    Standardizes column names by converting them to lowercase and replacing
    spaces with underscores.
    """
    if df is not None:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        print("Column names standardized.")
    return df

def handle_missing_values(df):
    """
    Fills missing values in string columns with 'Unknown' and in numeric
    columns with the mean.
    Note: This dataset has no missing values, but this is a good practice.
    """
    if df is not None:
        # Fill string columns with 'Unknown'
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Unknown')
        
        # Fill numeric columns with the mean
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].fillna(df[col].mean())
        print("Missing values handled.")
    return df

def convert_data_types(df):
    """
    Converts columns to appropriate data types.
    - 'interactions_with_customer_service' and 'purchase_history' to datetime.
    - 'customer_id' to string.
    """
    if df is not None:
        # Convert date columns to datetime objects
        date_columns = ['purchase_history', 'interactions_with_customer_service']
        for col in date_columns:
            # First, clean the column to ensure consistent date format
            df[col] = df[col].str.replace('(\d+)-(\d+)-(\d+)', r'\2/\1/\3', regex=True)
            
            # Now convert to datetime, with error handling
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                print(f"Error converting '{col}' to datetime: {e}")

        # Convert 'customer_id' to string to avoid mathematical operations
        df['customer_id'] = df['customer_id'].astype(str)
        print("Data types converted.")
    return df

def validate_data(df):
    """
    Performs basic data validation.
    - Checks for negative ages, incomes, coverage, and premium amounts.
    - If found, it prints a message and handles them.
    """
    if df is not None:
        negative_check_cols = ['age', 'income_level', 'coverage_amount', 'premium_amount']
        
        for col in negative_check_cols:
            if (df[col] < 0).any():
                print(f"Warning: Negative values found in '{col}'. Replacing with absolute values.")
                df[col] = df[col].abs()
    print("Data validation complete.")
    return df
    
def clean_data(df):
    """
    Main function to orchestrate the data cleaning process.
    It calls the other helper functions in sequence.
    """
    print("Starting data cleaning process...")
    df = standardize_column_names(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = validate_data(df)
    print("Data cleaning process finished.")
    return df
