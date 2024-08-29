# train the dataset (.csv file), & save the extracted features into model.pkl & scaler.pkl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Path to the dataset
dataset_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\features_with_amplitude.csv" # Path to your .csv file
# Path to save the model and scaler
model_save_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\model.pkl" # Save location of your model.pkl
scaler_save_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\scaler.pkl" # Save location of your scaler.pkl

print("Loading dataset...")
data = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")

# Prepare features and target
print("Preparing features and target...")
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split completed.")

# Create a pipeline with scaling and classification
print("Creating model pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
print("Pipeline created.")

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training completed.")

# Save the model and scaler
print("Saving the model and scaler...")
joblib.dump(pipeline, model_save_path)
print(f"Model saved to {model_save_path}.")

# Optionally, you can save the scaler separately if needed
scaler = pipeline.named_steps['scaler']
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to {scaler_save_path}.")

print("All operations completed successfully.")
