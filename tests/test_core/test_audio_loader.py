import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audiofeatures.core.audio_loader import load_audio, get_audio_info

class TestAudioLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.assets_dir = self.temp_dir_obj.name
        self.wav_file = os.path.join(self.assets_dir, "test.wav")
        self.src_sr = 22050
        self.duration = 0.25
        t = np.linspace(0.0, self.duration, int(self.src_sr * self.duration), endpoint=False)
        audio = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(self.wav_file, audio, self.src_sr)

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_load_audio_(self):
        """测试音频加载"""
        # wav 文件
        wav_audio, wav_sr = load_audio(self.wav_file)
        self.assertIsInstance(wav_audio, np.ndarray)
        self.assertGreater(wav_audio.size, 0)
        self.assertEqual(wav_sr, self.src_sr)

        # 采样率
        audio_16k, sr_16k = load_audio(self.wav_file, sr=16000)
        self.assertEqual(sr_16k, 16000)

    def test_get_audio_info(self):
        """测试获取音频信息"""
        # wav 文件
        wav_info = get_audio_info(self.wav_file)
        self.assertIsInstance(wav_info, dict)
        self.assertEqual(wav_info["format"], "wav")
        self.assertEqual(wav_info["sr"], self.src_sr)
        self.assertEqual(wav_info["channels"], 1)
        self.assertGreater(wav_info["duration"], 0)

    def test_error_handling(self):
        """测试错误处理"""
        non_exist_file = os.path.join(self.assets_dir, "non_exist_file.wav")
        with self.assertRaises(Exception):
            load_audio(non_exist_file)

if __name__ == "__main__":
    unittest.main()
