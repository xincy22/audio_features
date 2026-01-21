import unittest
import numpy as np

from audiofeatures.utils.contract import ensure_float32, to_feature_matrix


class TestContractUtils(unittest.TestCase):
    def test_ensure_float32(self):
        values = [0, 1, -1]
        arr = ensure_float32(values)
        self.assertEqual(arr.dtype, np.float32)

    def test_ensure_float32_clip(self):
        values = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        arr = ensure_float32(values, clip=True)
        np.testing.assert_allclose(arr, np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    def test_to_feature_matrix_1d(self):
        values = np.array([1.0, 2.0, 3.0])
        mat = to_feature_matrix(values)
        self.assertEqual(mat.shape, (3, 1))

    def test_to_feature_matrix_transpose(self):
        values = np.random.randn(5, 10)
        mat = to_feature_matrix(values, frame_axis=1)
        self.assertEqual(mat.shape, (10, 5))

    def test_to_feature_matrix_invalid_axis(self):
        with self.assertRaises(ValueError):
            to_feature_matrix(np.zeros((2, 2)), frame_axis=2)


if __name__ == "__main__":
    unittest.main()
