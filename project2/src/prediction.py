import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

def build_predictive_model(df, output_dir='.'):
    """
    Builds and trains a predictive model to forecast future purchases.

    Args:
        df (pd.DataFrame): The segmented DataFrame.
        output_dir (str): The directory to save the output files.

    Returns:
        The trained predictive model.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot build predictive model.")
        return None

    print("\nStarting predictive model building...")
    
    # We will predict a binary target: "Is this a high-value customer?"
    # Define "high-value" as having a CLV in the top 25% of the data
    try:
        df['is_high_value'] = (df['customer_lifetime_value'] > df['customer_lifetime_value'].quantile(0.75)).astype(int)
        
        # Features (X) and Target (y)
        features = ['age', 'income_level', 'purchase_frequency', 'tenure_days', 'days_since_last_interaction']
        target = 'is_high_value'
        
        # Ensure all required features are in the DataFrame
        if not all(col in df.columns for col in features):
            missing_features = [col for col in features if col not in df.columns]
            print(f"Error: Missing one or more required features for prediction. Missing: {missing_features}")
            return None
        
        X = df[features]
        y = df[target]

        # Handle any NaN values in the features
        if X.isnull().values.any():
            X.fillna(X.mean(), inplace=True)
            print("Missing values in features imputed with the mean.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")
        
        # Initialize and train the Decision Tree Classifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Predictive model trained successfully.")

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save the trained model
        model_path = os.path.join(output_dir, 'predictive_model.pkl')
        joblib.dump(model, model_path)
        print(f"\nPredictive model saved to {model_path}")

        return model

    except Exception as e:
        print(f"An error occurred during predictive model building: {e}")
        return None
