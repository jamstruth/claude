import claude_low_level_library as low_level
import unittest
import numpy as np

class LowLevelLibraryTest(unittest.TestCase):

    def test_scalar_gradient_x_all_zeroes(self):
        # Arrange
        a = np.zeros((3,3,3))
        nlon = 3
        i = 0
        j = 0
        k = 0
        dx = np.array([1, 2, 1])

        expected_result = 0
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_middle_value(self):
        # Arrange
        a = np.zeros((3,3,3))
        nlon = 3
        i = 1
        j = 1
        k = 1
        a[1,0,1] = 0
        a[1,1,1] = 2
        a[1,2,1] = 4
        dx = np.array([1, 2, 1])

        expected_result = 2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)