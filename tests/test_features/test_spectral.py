import unittest
import numpy as np

from audiofeatures.features import mfcc, delta_mfcc, mel_spectrogram


class TestSpectralFeatures(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1, self.sr, endpoint=False)
        self.signal = np.sin(2 * np.pi * 440 * t)

    def test_mfcc_and_delta(self):
        mfccs = mfcc(self.signal, sr=self.sr, n_mfcc=13, n_fft=512, hop_length=256)
        self.assertEqual(mfccs.shape[1], 13)
        self.assertGreater(mfccs.shape[0], 0)
        delta = delta_mfcc(mfccs, order=1)
        self.assertEqual(delta.shape, mfccs.shape)

    def test_mel_spectrogram(self):
        mel = mel_spectrogram(self.signal, sr=self.sr, n_fft=512, hop_length=256, n_mels=40)
        self.assertEqual(mel.shape[1], 40)
        self.assertGreater(mel.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
