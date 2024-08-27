import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_audio(file_path, sr=None):
    """Load an audio file and apply dynamic range compression."""
    audio, sr = librosa.load(file_path, sr=sr)
    # Apply dynamic range compression
    audio = librosa.effects.percussive(audio)
    return audio, sr

# Define file path for AI-generated voice
ai_voice_path = r'C:\Users\Paradox\Documents\Project Resources\AI Voices\Audrey\audrey.wav'

# Load and process the AI voice audio file
ai_voice, sr_ai = load_audio(ai_voice_path)

# Plot settings
plt.figure(figsize=(14, 9))

# Plot Waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(ai_voice, sr=sr_ai, color='r', label='AI Voice')
plt.title('AI Voice Waveform')
plt.legend(loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot Spectrogram
plt.subplot(2, 1, 2)
D_ai = librosa.amplitude_to_db(librosa.stft(ai_voice), ref=np.max)
librosa.display.specshow(D_ai, sr=sr_ai, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('AI-Generated Voice Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()
