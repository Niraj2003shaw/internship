import pandas as pd
import os
from data_cleaning import load_data_from_file, clean_data
from feature_engineering import feature_engineer_data
from segmentation import segment_customers
from prediction import build_predictive_model

def main():
    """
    Main function to run the entire customer segmentation and predictive modeling pipeline.
    """
    # Define file paths
    raw_data_path = os.path.join("D:\\", "Niraj", "OneDrive", "Desktop", "internship", "customer_segmentation_data.csv")
    cleaned_data_path = os.path.join("D:\\", "Niraj", "OneDrive", "Desktop", "internship", "project2", "src","internship","project2", "cleaned_customer_data.csv")
    segmented_data_path = os.path.join("D:\\", "Niraj", "OneDrive", "Desktop", "internship", "project2", "src", "segmented_customer_data.csv")
    
    # Step 1: Data Loading and Cleaning
    df = pd.DataFrame()
    if os.path.exists(cleaned_data_path):
        print("Cleaned data found. Loading directly for feature engineering...")
        df = pd.read_csv(cleaned_data_path)
    else:
        print("Cleaned data not found. Running full cleaning pipeline...")
        df = load_data_from_file(raw_data_path)
        if not df.empty:
            df = clean_data(df)
            if not df.empty:
                df.to_csv(cleaned_data_path, index=False)
                print(f"Cleaned data saved to {cleaned_data_path}")
            else:
                print("Data cleaning resulted in an empty DataFrame. Exiting.")
                return
        else:
            print("Failed to load data. Exiting.")
            return

    # Step 2: Feature Engineering
    engineered_df = feature_engineer_data(df.copy())
    if engineered_df is None or engineered_df.empty:
        print("Feature engineering failed. Exiting.")
        return

    # Step 3: Customer Segmentation
    segmented_df, kmeans_model = segment_customers(engineered_df.copy())
    if segmented_df is None or segmented_df.empty or kmeans_model is None:
        print("Customer segmentation failed. Exiting.")
        return
    
    segmented_df.to_csv(segmented_data_path, index=False)
    print(f"\nFinal segmented data saved to {segmented_data_path}")
    
    # Step 4: Predictive Modeling
    print("\n----------------------------------")
    print("Beginning Predictive Modeling Step...")
    print("----------------------------------")
    predictive_model = build_predictive_model(segmented_df.copy())
    if predictive_model is None:
        print("Predictive model building failed. Exiting.")
        return
    
    print("\n--- Pipeline Complete ---")
    print("The pipeline has successfully:")
    print("1. Loaded and cleaned the raw data.")
    print("2. Engineered new features like CLV and purchase frequency.")
    print("3. Segmented customers into distinct groups using K-Means.")
    print("4. Trained and saved a predictive model to forecast high-value customers.")
    print("\nYou can now use 'segmented_customer_data.csv' for further analysis and insights.")

if __name__ == "__main__":
    main()
