"""
Visualization example.
"""

import matplotlib.pyplot as plt

from audiofeatures.utils import load_audio
from audiofeatures.visualization import plot_waveform, plot_spectrogram

# Load audio
signal, sr = load_audio("cough.wav", sr=16000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_waveform(signal, sr, title="Waveform", ax=ax1)
plot_spectrogram(signal, sr, title="Spectrogram", ax=ax2)

plt.tight_layout()
plt.show()
