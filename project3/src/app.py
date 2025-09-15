import os
from data_cleaning import clean_data
from exploratory_analysis import perform_eda

# Define the absolute path to your raw data file
DATA_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3", "GME_stock.csv")

# Define the path where the cleaned data will be saved
CLEANED_DATA_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3\\src", "cleaned_dataSP.csv")

if __name__ == "__main__":
    print("Starting the financial analysis pipeline...")

    # Step 1: Data Cleaning
    print("\n--- Step 1: Starting data cleaning process ---")
    cleaned_df = clean_data(DATA_PATH)

    if cleaned_df is not None:
        print("\nData cleaning was successful!")
        
        # Step 2: Save the cleaned DataFrame
        try:
            cleaned_df.to_csv(CLEANED_DATA_PATH)
            print(f"\nSuccessfully saved the cleaned data to {CLEANED_DATA_PATH}")
            
            # Step 3: Exploratory Data Analysis (EDA)
            print("\n--- Step 2: Starting Exploratory Data Analysis (EDA) ---")
            perform_eda(CLEANED_DATA_PATH)
            print("\nFinancial analysis pipeline completed successfully.")

        except Exception as e:
            print(f"\nAn error occurred while saving the file: {e}")

    else:
        print("\nData cleaning failed. EDA process will not run.")