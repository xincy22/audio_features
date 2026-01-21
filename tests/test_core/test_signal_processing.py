import unittest
import numpy as np
from audiofeatures.core.signal_processing import frame_signal, apply_window

class TestSignalProcessing(unittest.TestCase):
    def setUp(self):
        self.test_signal = np.sin(np.linspace(0, 4*np.pi, 100))
        self.frame_length = 25
        self.hop_length = 10

    def test_frame_signal(self):
        """测试信号分帧"""

        with self.subTest(msg="测试没有居中的情况"):
            frames = frame_signal(
                self.test_signal,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                center=False
            )

            # 检查输出类型和维度
            self.assertIsInstance(frames, np.ndarray)
            self.assertEqual(frames.ndim, 2)

            # 验证分帧后的形状
            expected_frames = (len(self.test_signal) - self.frame_length) // self.hop_length + 1
            self.assertEqual(frames.shape, (expected_frames, self.frame_length))

        with self.subTest(msg="测试居中的情况"):
            frames = frame_signal(
                self.test_signal,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            ) # center默认为True

            # 检查输出类型和维度
            self.assertIsInstance(frames, np.ndarray)
            self.assertEqual(frames.ndim, 2)

            # 验证分帧后的形状
            padded_length = len(self.test_signal) + self.frame_length - 1
            expected_frames = (padded_length - self.frame_length) // self.hop_length + 1
            self.assertEqual(frames.shape, (expected_frames, self.frame_length))

    def test_frame_signal_error(self):
        """测试信号分帧错误处理"""
        # 测试多维数组输入
        invalid_signal = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            frame_signal(invalid_signal, self.frame_length, self.hop_length)

    def test_apply_window(self):
        """测试窗函数应用"""
        frames = frame_signal(
            self.test_signal,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        windowed_frames = apply_window(frames, window_type='hann')

        # 检查输出类型
        self.assertIsInstance(windowed_frames, np.ndarray)

        # 检查不支持的窗函数类型
        with self.assertRaises(ValueError):
            apply_window(frames, window_type='unsupported_window')

if __name__ == "__main__":
    unittest.main()
