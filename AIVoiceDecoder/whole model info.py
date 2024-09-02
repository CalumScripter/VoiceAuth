import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the saved model and scaler
model_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\model.pkl"
scaler_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load your complete dataset
data_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\features_with_amplitude.csv"
data = pd.read_csv(data_path)

# Define the features and the target
X = data.drop('label', axis=1)
y = data['label']

# Scale the features
X_scaled = scaler.transform(X)

# Predict the labels
y_pred = model.predict(X_scaled)

# Generate and print the confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("\nAccuracy:", accuracy)

# Calculate and print classification report
report = classification_report(y, y_pred, target_names=['Real Voice', 'AI-Generated Voice'])
print("\nClassification Report:")
print(report)
