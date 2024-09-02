from pydub import AudioSegment
import os

# Directory containing the .flac files
source_folder = r'C:\Users\Paradox\Documents\Testing Dataset for VoiceAuth\Real Voices'
# Directory to save the .wav files
destination_folder = r'C:\Users\Paradox\Documents\Testing Dataset for VoiceAuth\Real Voices .wav'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through all the .flac files in the source directory
for flac_file in os.listdir(source_folder):
    if flac_file.endswith('.flac'):
        # Full path to your .flac file
        flac_path = os.path.join(source_folder, flac_file)
        # Load the .flac file
        audio = AudioSegment.from_file(flac_path, format="flac")
        
        # Define the output path for the .wav file
        wav_path = os.path.join(destination_folder, flac_file[:-5] + ".wav")
        
        # Export the audio in .wav format
        audio.export(wav_path, format="wav")
        print(f'Converted {flac_file} to WAV.')

print("All files have been converted.")
