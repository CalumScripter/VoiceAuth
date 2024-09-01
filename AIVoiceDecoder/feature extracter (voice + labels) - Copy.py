import pandas as pd
from sklearn.model_selection import train_test_split

# Load your complete dataset
data_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\features_with_amplitude.csv"
data = pd.read_csv(data_path)

# Define the features and the target
X = data.drop('label', axis=1)  # Assuming 'label' is the column with the labels
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% of the data is used for testing

# Save the testing data with more descriptive filenames
X_test.to_csv(r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\voice_features_test.csv", index=False)
y_test.to_csv(r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\voice_labels_test.csv", index=False)

print("Testing files saved successfully.")

