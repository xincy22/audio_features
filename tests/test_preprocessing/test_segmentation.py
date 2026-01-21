import unittest
import numpy as np
import audiofeatures.preprocessing as prep

class TestSegmentation(unittest.TestCase):
    def setUp(self):
        # 创建测试信号：包含两个有声片段，使用更大的幅度确保能被检测到
        self.sr = 16000
        t = np.linspace(0, 1, self.sr)
        self.signal = np.zeros_like(t)
        # 使用更大的幅度 (1.0)，确保能被能量检测到
        self.signal[4000:6000] = 1.0 * np.sin(2 * np.pi * 1000 * t[4000:6000])
        self.signal[8000:10000] = 1.0 * np.sin(2 * np.pi * 500 * t[8000:10000])
        
    def test_segment_by_energy(self):
        # 测试基于能量的分割，使用更低的阈值
        # 添加打印语句来调试
        energy = np.sum(self.signal**2) / len(self.signal)
        print(f"Signal energy: {energy}")
        print(f"Max amplitude: {np.max(np.abs(self.signal))}")
        
        segments = prep.segment_by_energy(
            self.signal, 
            self.sr, 
            threshold=0.01,  # 使用非常低的阈值
            min_length=0.05  # 减少最小片段长度
        )
        print(f"Segments found: {segments}")
        self.assertTrue(len(segments) > 0)
        self.assertTrue(all(isinstance(seg, tuple) for seg in segments))
        
    def test_segment_by_zcr(self):
        # 测试基于过零率的分割
        segments = prep.segment_by_zcr(
            self.signal, self.sr, 
            threshold=0.1, 
            min_length=0.1
        )
        self.assertTrue(len(segments) > 0)
        self.assertTrue(all(isinstance(seg, tuple) for seg in segments))

if __name__ == '__main__':
    unittest.main()
