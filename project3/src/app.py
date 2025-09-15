import os
from data_cleaning import clean_data

# Define the absolute path to your data file
# Make sure to replace this path if your file location changes
DATA_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3", "GME_stock.csv")

# Define the path where the cleaned data will be saved
CLEANED_DATA_PATH = os.path.join("D:\\Niraj\\OneDrive\\Desktop\\internship\\project3\\src", "cleaned_dataSP.csv")

if __name__ == "__main__":
    print("Starting the data cleaning process...")

    # Call the clean_data method from the data_cleaning module
    cleaned_df = clean_data(DATA_PATH)

    if cleaned_df is not None:
        print("\nData cleaning was successful!")
        print("\nFirst 5 rows of the cleaned data:")
        print(cleaned_df.head())

        # Save the cleaned DataFrame to a new CSV file
        try:
            cleaned_df.to_csv(CLEANED_DATA_PATH)
            print(f"\nSuccessfully saved the cleaned data to {CLEANED_DATA_PATH}")
        except Exception as e:
            print(f"\nAn error occurred while saving the file: {e}")

    else:
        print("\nData cleaning failed. Please check the file path and the `data_cleaning.py` script.")