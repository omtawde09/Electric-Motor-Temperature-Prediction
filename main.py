import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- 1. Data Loading and Pre-processing ---
try:
    df = pd.read_csv('measures_v2.csv')
    df = df.drop(columns=['profile_id', 'torque'])
    print("Data loaded and pre-processed.")

    # --- 2. Feature Engineering and Scaling ---
    X = df.drop('pm', axis=1)
    y = df['pm']

    # We will fit the scaler on the FULL dataset's features.
    # This is a common practice before saving it for deployment.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features have been scaled.")

    # Save the scaler object to a file
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Scaler has been saved to 'scaler.pkl'")

    # Split data for final model training
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Data prepared for final training.")

    # --- 3. Final Model Training ---
    print("\nTraining the final Random Forest model on the full training data...")
    # Initialize the best model with the same settings
    final_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Train the model
    final_model.fit(X_train, y_train)
    print("Final model training complete.")

    # --- 4. Saving the Final Model ---
    # Save the trained model to a file
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(final_model, model_file)
    print("Trained model has been saved to 'model.pkl'")

    print("\nProcess complete. You now have 'model.pkl' and 'scaler.pkl' in your project directory.")

except FileNotFoundError:
    print("Error: 'measures_v2.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
