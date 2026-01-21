"""
Augmentation example.
"""

from audiofeatures.utils import load_audio
from audiofeatures.augmentation import time_stretch, pitch_shift, add_noise

signal, sr = load_audio("cough.wav", sr=16000)

augmented_signals = {
    "time_stretched": [
        time_stretch(signal, sr=sr, rate=0.9),
        time_stretch(signal, sr=sr, rate=1.1)
    ],
    "pitch_shifted": [
        pitch_shift(signal, sr=sr, n_steps=-2),
        pitch_shift(signal, sr=sr, n_steps=2)
    ],
    "noisy": [
        add_noise(signal, noise_level=0.005),
        add_noise(signal, noise_level=0.01)
    ]
}

print({key: len(value) for key, value in augmented_signals.items()})
