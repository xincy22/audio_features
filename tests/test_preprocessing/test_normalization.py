import unittest
import numpy as np
import warnings
import audiofeatures.preprocessing as prep

class TestNormalization(unittest.TestCase):
    def setUp(self):
        # 创建测试信号
        self.signal = np.random.randn(1000) * 0.1
        
    def test_normalize_amplitude(self):
        # 测试dBFS归一化
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normalized = prep.normalize_amplitude(self.signal, target_dBFS=-20)
            self.assertEqual(len(normalized), len(self.signal))
            
    def test_peak_normalize(self):
        # 测试峰值归一化
        target_peak = 0.95
        normalized = prep.peak_normalize(self.signal, target_peak)
        self.assertAlmostEqual(np.max(np.abs(normalized)), target_peak, places=6)
        
    def test_z_normalize(self):
        # 测试Z-score标准化
        normalized = prep.z_normalize(self.signal)
        self.assertAlmostEqual(np.mean(normalized), 0, places=6)
        self.assertAlmostEqual(np.std(normalized), 1, places=6)
        
    def test_min_max_normalize(self):
        # 测试最小-最大归一化
        normalized = prep.min_max_normalize(self.signal, 0, 1)
        self.assertGreaterEqual(np.min(normalized), 0)
        self.assertLessEqual(np.max(normalized), 1)

if __name__ == '__main__':
    unittest.main()
