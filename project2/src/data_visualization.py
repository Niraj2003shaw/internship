import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_segments(file_path):
    """
    Loads segmented customer data and generates visualizations.
    
    Args:
        file_path (str): The path to the segmented customer data CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty. Please ensure the data pipeline ran correctly.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    print("Data loaded successfully. Generating visualizations...")
    
    # Set the style for the plots
    sns.set_style("whitegrid")
    
    # --- Visualization 1: Customer Count by Segment ---
    plt.figure(figsize=(10, 6))
    sns.countplot(x='segment', data=df, palette='viridis')
    plt.title('Customer Count by Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Customers')
    plt.savefig('customer_count_plot.png')
    print("Saved customer_count_plot.png")
    
    # --- Visualization 2: Income vs. Age Scatter Plot (with Segmentation) ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='income_level', y='age', hue='segment', data=df, palette='tab10', s=100, alpha=0.7)
    plt.title('Age vs. Income Level by Customer Segment')
    plt.xlabel('Income Level')
    plt.ylabel('Age')
    plt.legend(title='Segment')
    plt.savefig('income_age_scatter.png')
    print("Saved income_age_scatter.png")

    # --- Visualization 3: Average Income by Segment ---
    avg_income = df.groupby('segment')['income_level'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    # Corrected line to address the FutureWarning
    sns.barplot(x='segment', y='income_level', hue='segment', data=avg_income, palette='plasma', legend=False)
    plt.title('Average Income by Customer Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Average Income Level')
    plt.savefig('average_income_bar.png')
    print("Saved average_income_bar.png")

    print("All visualizations have been successfully generated and saved.")

if __name__ == "__main__":
    file_path = "segmented_customer_data.csv"
    visualize_segments(file_path)