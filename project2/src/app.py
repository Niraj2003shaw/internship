import os
from data_cleaning import load_data_from_file, clean_data

def main():
    """
    Main function to run the application.
    It loads the raw data, cleans it, and saves the result to a CSV file.
    """
    # 1. Load the data
    df = load_data_from_file()

    if df is not None:
        # 2. Clean the data
        cleaned_df = clean_data(df)
        
        # 3. Display the cleaned data info
        print("\n--- Cleaned Data Info ---")
        cleaned_df.info()
        
        # 4. Display the first few rows of the cleaned data
        print("\n--- Cleaned Data Head ---")
        print(cleaned_df.head())
        
        # 5. Define the output file path based on the user's request
        output_dir = os.path.join("internship", "project2")
        output_filepath = os.path.join(output_dir, "cleaned_customer_data.csv")
        
        # 6. Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 7. Save the cleaned DataFrame to a CSV file
        cleaned_df.to_csv(output_filepath, index=False)
        print(f"\n--- Success! ---")
        print(f"Cleaned data has been saved to: {output_filepath}")

if __name__ == "__main__":
    main()
