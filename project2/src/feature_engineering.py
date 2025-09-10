import pandas as pd
import numpy as np
import os
from datetime import datetime

# Define the file path for the cleaned data
CLEANED_DATA_FILEPATH = os.path.join("internship", "project2", "cleaned_customer_data.csv")

def load_cleaned_data():
    """
    Loads the cleaned data from the specified file path.
    """
    try:
        df = pd.read_csv(CLEANED_DATA_FILEPATH)
        print("Cleaned data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {CLEANED_DATA_FILEPATH}")
        return None
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return None

def create_tenure(df):
    """
    Creates a new feature for customer tenure in months based on purchase history.
    """
    if df is not None and 'purchase_history' in df.columns:
        # Ensure the column is in datetime format
        df['purchase_history'] = pd.to_datetime(df['purchase_history'])
        
        # Calculate tenure in months
        current_date = datetime.now()
        df['tenure_months'] = (current_date - df['purchase_history']).dt.days // 30
        print("Feature 'tenure_months' created.")
    return df

def create_days_since_interaction(df):
    """
    Creates a new feature for days since last customer service interaction.
    """
    if df is not None and 'interactions_with_customer_service' in df.columns:
        # Ensure the column is in datetime format
        df['interactions_with_customer_service'] = pd.to_datetime(df['interactions_with_customer_service'])
        
        # Calculate days since last interaction
        current_date = datetime.now()
        df['days_since_last_interaction'] = (current_date - df['interactions_with_customer_service']).dt.days
        print("Feature 'days_since_last_interaction' created.")
    return df

def create_income_bracket(df):
    """
    Creates a new categorical feature for income bracket.
    """
    if df is not None and 'income_level' in df.columns:
        # Define income brackets
        bins = [0, 50000, 100000, np.inf]
        labels = ['Low', 'Medium', 'High']
        df['income_bracket'] = pd.cut(df['income_level'], bins=bins, labels=labels, right=False)
        print("Feature 'income_bracket' created.")
    return df

def create_customer_lifetime_value(df):
    """
    Calculates Customer Lifetime Value (CLV) based on premium amounts.
    """
    if df is not None and 'premium_amount' in df.columns and 'customer_id' in df.columns:
        clv_df = df.groupby('customer_id')['premium_amount'].sum().reset_index()
        clv_df.rename(columns={'premium_amount': 'customer_lifetime_value'}, inplace=True)
        df = df.merge(clv_df, on='customer_id', how='left')
        print("Feature 'customer_lifetime_value' created.")
    return df

def create_average_order_value(df):
    """
    Calculates Average Order Value (AOV) based on premium amounts.
    """
    if df is not None and 'premium_amount' in df.columns and 'customer_id' in df.columns:
        aov_df = df.groupby('customer_id')['premium_amount'].mean().reset_index()
        aov_df.rename(columns={'premium_amount': 'average_order_value'}, inplace=True)
        df = df.merge(aov_df, on='customer_id', how='left')
        print("Feature 'average_order_value' created.")
    return df

def create_purchase_frequency(df):
    """
    Calculates Purchase Frequency by counting transactions per customer.
    """
    if df is not None and 'customer_id' in df.columns:
        frequency_df = df.groupby('customer_id').size().reset_index(name='purchase_frequency')
        df = df.merge(frequency_df, on='customer_id', how='left')
        print("Feature 'purchase_frequency' created.")
    return df

def one_hot_encode_categorical(df):
    """
    Applies one-hot encoding to specified categorical features.
    """
    if df is not None:
        categorical_cols = ['gender', 'marital_status']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print("Categorical features one-hot encoded.")
    return df

def feature_engineer_data(df):
    """
    Main function to orchestrate the feature engineering process.
    """
    print("Starting feature engineering process...")
    df = create_tenure(df)
    df = create_days_since_interaction(df)
    df = create_income_bracket(df)
    df = create_customer_lifetime_value(df)
    df = create_average_order_value(df)
    df = create_purchase_frequency(df)
    df = one_hot_encode_categorical(df)
    print("Feature engineering process finished.")
    return df
