import unittest
import numpy as np
import audiofeatures.preprocessing as prep

class TestFiltering(unittest.TestCase):
    def setUp(self):
        # 创建测试信号：1kHz正弦波
        self.sr = 16000
        t = np.linspace(0, 1, self.sr)
        self.signal = np.sin(2 * np.pi * 1000 * t)
        
    def test_low_pass_filter(self):
        # 测试低通滤波器是否能通过低频信号
        filtered = prep.low_pass_filter(self.signal, self.sr, cutoff_freq=2000)
        self.assertEqual(len(filtered), len(self.signal))
        
    def test_high_pass_filter(self):
        # 测试高通滤波器是否能通过高频信号
        filtered = prep.high_pass_filter(self.signal, self.sr, cutoff_freq=500)
        self.assertEqual(len(filtered), len(self.signal))
        
    def test_band_pass_filter(self):
        # 测试带通滤波器
        filtered = prep.band_pass_filter(self.signal, self.sr, 500, 2000)
        self.assertEqual(len(filtered), len(self.signal))
        
    def test_median_filter(self):
        # 测试中值滤波器去除脉冲噪声
        noisy_signal = self.signal.copy()
        noisy_signal[100] = 10  # 添加脉冲噪声
        filtered = prep.median_filter(noisy_signal, kernel_size=3)
        self.assertTrue(np.abs(filtered[100]) < 10)

if __name__ == '__main__':
    unittest.main()
