import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import librosa

# Load model and scaler
model_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\model.pkl"
scaler_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Model and scalar\scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    avg_amplitude = np.mean(np.abs(y))
    features = {'avg_amplitude': avg_amplitude}
    for idx, mfcc in enumerate(mfccs):
        features[f'mfcc{idx}'] = np.mean(mfcc)
    column_order = ['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
                    'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'avg_amplitude']
    features_df = pd.DataFrame([features])[column_order]
    return features_df

def predict_voice():
    filepath = filedialog.askopenfilename(title="Select an Audio File", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
    if filepath:
        features_df = extract_features(filepath)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        result = "AI-generated" if prediction[0] == 1 else "Real"
        messagebox.showinfo("Prediction Result", f"The voice is predicted as: {result}")
        evaluate_model()

# You can find the 'voice_features_test.csv' + 'voice_labels_test.csv' in "AIVoiceDecoder\data\Dataset Features\"

def evaluate_model():
    # Load test data with explicit feature names setting
    X_test = pd.read_csv(r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\voice_features_test.csv")
    y_test = pd.read_csv(r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\voice_labels_test.csv")
    y_test = y_test.squeeze()  # Ensure y_test is a Series if it's a single column

    # Ensure the feature names are set correctly for the test data
    feature_names = [f'mfcc{idx}' for idx in range(13)] + ['avg_amplitude']
    X_test.columns = feature_names
    X_test_scaled = scaler.transform(X_test)

    # Evaluating model with test data
    predictions = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    # Display results
    result_text = f"Confusion Matrix:\n{cm}\n\nClassification Report:\nAccuracy: {report['accuracy']}\n"
    result_text += f"Macro Avg - Precision: {report['macro avg']['precision']}, Recall: {report['macro avg']['recall']}, F1-score: {report['macro avg']['f1-score']}\n"
    result_text += f"Weighted Avg - Precision: {report['weighted avg']['precision']}, Recall: {report['weighted avg']['recall']}, F1-score: {report['weighted avg']['f1-score']}"
    messagebox.showinfo("Model Evaluation", result_text)

root = tk.Tk()
root.title("Voice Classification Test")

btn_predict = tk.Button(root, text="Select and Classify Audio", command=predict_voice)
btn_predict.pack(pady=20)

root.mainloop()
