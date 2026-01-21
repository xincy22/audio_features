import unittest

from audiofeatures.utils import (
    hz_to_mel,
    mel_to_hz,
    hz_to_note,
    note_to_hz,
    seconds_to_samples,
    samples_to_seconds
)


class TestConversionUtils(unittest.TestCase):
    def test_hz_mel_roundtrip(self):
        hz = 1000.0
        mel = hz_to_mel(hz)
        self.assertAlmostEqual(mel_to_hz(mel), hz, places=1)

    def test_note_conversion(self):
        self.assertEqual(hz_to_note(440.0), "A4")
        self.assertAlmostEqual(note_to_hz("A4"), 440.0, places=2)
        self.assertAlmostEqual(note_to_hz("C#4"), 277.18, places=1)

    def test_time_sample_conversion(self):
        samples = seconds_to_samples(1.5, sr=16000)
        self.assertEqual(samples, 24000)
        self.assertAlmostEqual(samples_to_seconds(samples, sr=16000), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
