import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def segment_customers(df, output_dir='.'):
    """
    Segments customers using K-Means clustering on the feature-engineered data.

    Args:
        df (pd.DataFrame): The feature-engineered DataFrame.
        output_dir (str): The directory to save the output files.

    Returns:
        tuple: A tuple containing the segmented DataFrame and the trained KMeans model.
    """
    if df.empty:
        print("DataFrame is empty. Cannot perform customer segmentation.")
        return pd.DataFrame(), None

    print("\nStarting customer segmentation...")

    # Step 1: Select features for clustering
    # We will use the new, engineered features for segmentation
    features = ['customer_lifetime_value', 'average_order_value', 'purchase_frequency', 'tenure_days', 'days_since_last_interaction']
    
    # Ensure all required features are in the DataFrame
    if not all(col in df.columns for col in features):
        missing_features = [col for col in features if col not in df.columns]
        print(f"Error: Missing one or more required features for segmentation. Missing: {missing_features}")
        return pd.DataFrame(), None

    # Create a copy to avoid SettingWithCopyWarning
    features_df = df[features].copy()
    
    # Step 2: Handle any remaining NaN or infinite values
    try:
        if features_df.isnull().values.any() or np.isinf(features_df.values).any():
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            features_df.fillna(features_df.mean(), inplace=True)
            print("Missing or infinite values imputed with the mean.")
    except Exception as e:
        print(f"Error handling NaN/inf values: {e}")
        return pd.DataFrame(), None

    # Step 3: Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    print("Features scaled using StandardScaler.")

    # Step 4: Find the optimal number of clusters using the Elbow Method
    print("Finding optimal number of clusters using Elbow Method...")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Method results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Within-cluster sum of squares (WCSS)')
    plt.grid(True)
    
    elbow_plot_path = os.path.join(output_dir, 'elbow_method_plot.png')
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow Method plot saved to {elbow_plot_path}")
    

    # Step 5: Perform K-Means clustering with optimal K (we'll assume 4 based on the plot)
    optimal_k = 4
    kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_model.fit(scaled_features)
    df['segment'] = kmeans_model.labels_
    print(f"Customer segmentation complete. Found {optimal_k} segments.")
    
    # Step 6: Save the trained K-Means model
    model_path = os.path.join(output_dir, 'kmeans_model.pkl')
    joblib.dump(kmeans_model, model_path)
    print(f"K-Means model saved to {model_path}")
    
    # Step 7: Analyze the segments
    print("\n--- Segment Analysis ---")
    segment_summary = df.groupby('segment')[features].mean()
    print(segment_summary)
    
    # Step 8: Return the segmented DataFrame and the model
    return df, kmeans_model
