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
        nlon = 4
        i = 1
        j = 1
        k = 1
        a[1,0,1] = 0
        a[1,1,1] = 2
        a[1,2,1] = 4
        dx = np.array([1, 1, 1, 1])

        expected_result = 4
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_middle_value_larger_distance(self):
        # Arrange
        a = np.zeros((3,3,3))
        nlon = 4
        i = 1
        j = 1
        k = 1
        a[1,0,1] = 0
        a[1,1,1] = 2
        a[1,2,1] = 4
        dx = np.array([1, 2, 1, 1])

        expected_result = 2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)
    
    def test_scalar_gradient_x_middle_value_smaller_distance(self):
        # Arrange
        a = np.zeros((3,3,3))
        nlon = 4
        i = 1
        j = 1
        k = 1
        a[1,0,1] = 0
        a[1,1,1] = 2
        a[1,2,1] = 4
        dx = np.array([1, 0.5, 1, 1])

        expected_result = 8
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_far_left_wraparound(self):
        # Arrange
        a = np.zeros((4,4,4))
        nlon = 4
        i = 0
        j = 0
        k = 0
        a[0,0,0] = 0
        a[0,1,0] = 2
        a[0,3,0] = 4
        dx = np.array([1, 1, 1, 1])  
        expected_result = -2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)
    
    def test_scalar_gradient_x_far_right_wraparound(self):
        # Arrange
        a = np.zeros((4,4,4))
        nlon = 3
        i = 3
        j = 3
        k = 3
        a[3,0,3] = 0
        a[3,2,3] = 2
        a[3,3,3] = 4
        dx = np.array([1, 1, 1, 1])
        expected_result = -2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)