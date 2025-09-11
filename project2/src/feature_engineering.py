import pandas as pd
import numpy as np
import datetime as dt

def feature_engineer_data(df):
    """
    Performs feature engineering on the cleaned DataFrame to create new,
    more meaningful features for customer segmentation.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new engineered features.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot perform feature engineering.")
        return pd.DataFrame()

    print("Starting feature engineering...")

    # Step 1: Standardize column names for consistency
    # This dictionary maps potential inconsistent names to standardized names
    standard_names = {
        'customerid': 'customer_id',
        'purchasehistory': 'purchase_history',
        'premiumamount': 'premium_amount',
        'interactionswithcustomerservice': 'interactions_with_customer_service',
        'policytype': 'policy_type'
    }
    
    # Create a mapping from current column names (normalized) to the standardized names
    col_map = {col.strip().lower().replace(' ', ''): col for col in df.columns}
    df.rename(columns={col_map[key]: value for key, value in standard_names.items() if key in col_map}, inplace=True)
    
    required_cols = ['customer_id', 'purchase_history', 'premium_amount', 'interactions_with_customer_service', 'policy_type']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing one or more required features for segmentation. Missing: {missing_cols}")
        return pd.DataFrame()

    # Step 2: Ensure correct data types for calculations
    df['premium_amount'] = pd.to_numeric(df['premium_amount'], errors='coerce')
    df['purchase_history'] = pd.to_datetime(df['purchase_history'], errors='coerce', dayfirst=True)
    df['interactions_with_customer_service'] = pd.to_datetime(df['interactions_with_customer_service'], errors='coerce', dayfirst=True)

    # Step 3: Impute missing values
    current_date = dt.datetime.now()
    df['premium_amount'] = df['premium_amount'].fillna(df['premium_amount'].mean())
    df['purchase_history'] = df['purchase_history'].fillna(current_date)
    df['interactions_with_customer_service'] = df['interactions_with_customer_service'].fillna(current_date)
    
    # Step 4: Create new features
    try:
        # Create a new feature for customer lifetime value (CLV)
        clv_df = df.groupby('customer_id')['premium_amount'].sum().reset_index()
        clv_df.rename(columns={'premium_amount': 'customer_lifetime_value'}, inplace=True)
        df = pd.merge(df, clv_df, on='customer_id', how='left')

        # Create a new feature for average order value (AOV)
        aov_df = df.groupby('customer_id')['premium_amount'].mean().reset_index()
        aov_df.rename(columns={'premium_amount': 'average_order_value'}, inplace=True)
        df = pd.merge(df, aov_df, on='customer_id', how='left')

        # Create a new feature for frequency of purchase
        freq_df = df.groupby('customer_id')['policy_type'].count().reset_index()
        freq_df.rename(columns={'policy_type': 'purchase_frequency'}, inplace=True)
        df = pd.merge(df, freq_df, on='customer_id', how='left')

        # Additional feature: Tenure in days
        df['tenure_days'] = (current_date - df['purchase_history']).dt.days

        # Additional feature: Days since last interaction
        df['days_since_last_interaction'] = (current_date - df['interactions_with_customer_service']).dt.days
        
        # Drop duplicates before returning, as new features are based on customer_id
        df.drop_duplicates(subset='customer_id', keep='first', inplace=True)

        print("Feature engineering successful.")
        return df

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return pd.DataFrame()
