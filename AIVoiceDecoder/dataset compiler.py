# Make sure your datasets of both AI voices & Real voices are ready (and they're in .wav)
# Give a saving directory for your model and scalar at Ln: 57

import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# Paths to the directories containing audio files
ai_voices_dir = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Voices\AI"  # Path to your AI Voices
real_voices_dir = r"C:\Users\Paradox\Documents\Project Resources\Real Voices\Dataset"  # Path to your Real Voices

# Function to extract features from an audio file
def extract_features(file_path, label):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        avg_amplitude = np.mean(np.abs(y))  # Calculate average amplitude
        features = np.mean(mfccs, axis=1)
        features = np.hstack((features, avg_amplitude, label))  # Append avg_amplitude and label
        return features
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

# Prepare the dataset
def prepare_dataset(directory, label):
    features_list = []
    files = os.listdir(directory)
    for filename in tqdm(files, desc=f"Processing {('AI' if label == 1 else 'Real')} Voices", total=len(files)):
        file_path = os.path.join(directory, filename)
        if file_path.endswith(".wav"):
            features = extract_features(file_path, label)
            if features is not None:
                features_list.append(features)
    return features_list

# Extract features from AI and real voices
print("Extracting features from AI voices...")
ai_features = prepare_dataset(ai_voices_dir, 1)  # Label 1 for AI-generated voices
print("AI voices processed.")

print("Extracting features from real voices...")
real_features = prepare_dataset(real_voices_dir, 0)  # Label 0 for real human voices
print("Real voices processed.")

# Combine the datasets
print("Combining features from AI and real voices...")
total_features = ai_features + real_features
feature_columns = [f'mfcc{i}' for i in range(13)] + ['avg_amplitude', 'label']
data_frame = pd.DataFrame(total_features, columns=feature_columns)
print("Features combined.")

# Save the combined dataset to a CSV file
output_path = r"C:\Users\Paradox\Documents\Project Resources\Dataset\Features csv\features_with_amplitude.csv"
data_frame.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
