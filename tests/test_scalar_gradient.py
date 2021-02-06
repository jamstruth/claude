import claude_low_level_library as low_level
import unittest
import numpy as np


class ScalarGradientTests(unittest.TestCase):

    def test_scalar_gradient_x_all_zeroes(self):
        # Arrange
        a = np.zeros((3, 3, 3))
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
        a = np.zeros((3, 3, 3))
        nlon = 4
        i = 1
        j = 1
        k = 1
        a[1, 0, 1] = 0
        a[1, 1, 1] = 2
        a[1, 2, 1] = 4
        dx = np.array([1, 1, 1, 1])

        expected_result = 4
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_middle_value_larger_distance(self):
        # Arrange
        a = np.zeros((3, 3, 3))
        nlon = 4
        i = 1
        j = 1
        k = 1
        a[1, 0, 1] = 0
        a[1, 1, 1] = 2
        a[1, 2, 1] = 4
        dx = np.array([1, 2, 1, 1])

        expected_result = 2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_middle_value_smaller_distance(self):
        # Arrange
        a = np.zeros((3, 3, 3))
        nlon = 4
        i = 1
        j = 1
        k = 1
        a[1, 0, 1] = 0
        a[1, 1, 1] = 2
        a[1, 2, 1] = 4
        dx = np.array([1, 0.5, 1, 1])

        expected_result = 8
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_far_left_wraparound(self):
        # Arrange
        a = np.zeros((4, 4, 4))
        nlon = 4
        i = 0
        j = 0
        k = 0
        a[0, 0, 0] = 0
        a[0, 1, 0] = 2
        a[0, 3, 0] = 4
        dx = np.array([1, 1, 1, 1])
        expected_result = -2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_far_right_wraparound(self):
        # Arrange
        a = np.zeros((4, 4, 4))
        nlon = 3
        i = 3
        j = 3
        k = 3
        a[3, 0, 3] = 0
        a[3, 2, 3] = 2
        a[3, 3, 3] = 4
        dx = np.array([1, 1, 1, 1])
        expected_result = -2
        # Act
        result = low_level.scalar_gradient_x(a, dx, nlon, i, j, k)
        # Assert
        self.assertEqual(expected_result, result)

    def test_scalar_gradient_x_matrix(self):
        a = np.array([[[3, 0], [4, 0], [7, 0]],
                      [[7, 0], [6, 0], [2, 0]],
                      [[8, 0], [8, 0], [8, 0]]])
        dx = np.array([1, 0.5, 2])

        expected_results = np.array([[[-3, 0], [4, 0], [-1, 0]],
                                     [[8, 0], [-10, 0], [2, 0]],
                                     [[0, 0], [0, 0], [0, 0]]])

        results = low_level.scalar_gradient_x_matrix(a, dx)

        self.assertTrue(np.array_equal(expected_results, results),
                        msg=f"expected: {expected_results}, actual: {results}")

    def test_scalar_gradient_x_matrix_primitive(self):
        a = np.array([[[3.0, 0], [4, 0], [7, 0]],
                      [[7, 0], [6, 0], [2, 0]],
                      [[3, 0], [4, 0], [7, 0]],
                      [[8, 0], [8, 0], [8, 0]]])
        dx = np.array([1, 0.5, 2, 1])

        expected_results = np.array([[[0, 0], [0, 0], [0, 0]],
                                     [[8, 0], [-10, 0], [2, 0]],
                                     [[-1.5, 0], [2, 0], [-0.5, 0]],
                                     [[0, 0], [0, 0], [0, 0]]])

        results = low_level.scalar_gradient_x_matrix_primitive(a, dx)

        self.assertTrue(np.array_equal(expected_results, results),
                        msg=f"expected: {expected_results}, actual: {results}")

    def test_scalar_gradient_y_matrix_dy_one(self):
        dy = 1.0
        a = np.array([[[3.0, 0], [4, 0], [7, 0]],
                      [[7, 0], [6, 0], [2, 0]],
                      [[3, 0], [4, 0], [7, 0]],
                      [[8, 0], [8, 0], [8, 0]]])
        
        expected_results = np.array([[[8.0, 0], [4, 0], [-10, 0]],
                                     [[0, 0], [0, 0], [0, 0]],
                                     [[1, 0], [2, 0], [6, 0]],
                                     [[10, 0], [8, 0], [2, 0]]])
        results = low_level.scalar_gradient_y_matrix(a, dy)
        self.assertTrue(np.array_equal(expected_results, results),
                        msg=f"expected: {expected_results}, actual: {results}")

    def test_scalar_gradient_y_matrix_dy_two(self):
        dy = 2.0
        a = np.array([[[3.0, 0], [4, 0], [7, 0]],
                      [[7, 0], [6, 0], [2, 0]],
                      [[3, 0], [4, 0], [7, 0]],
                      [[8, 0], [8, 0], [8, 0]]])
        
        expected_results = np.array([[[4.0, 0], [2, 0], [-5, 0]],
                                     [[0, 0], [0, 0], [0, 0]],
                                     [[0.5, 0], [1, 0], [3, 0]],
                                     [[5, 0], [4, 0], [1, 0]]])
        results = low_level.scalar_gradient_y_matrix(a, dy)
        self.assertTrue(np.array_equal(expected_results, results),
                        msg=f"expected: {expected_results}, actual: {results}")

    def test_scalar_gradient_y_matrix_primitive_dy_one(self):
        dy = 1.0
        a = np.array([[[3.0, 0], [4, 0], [7, 0]],
                      [[7, 0], [6, 0], [2, 0]],
                      [[3, 0], [4, 0], [7, 0]],
                      [[8, 0], [8, 0], [8, 0]]])
        
        expected_results = np.array([[[8.0, 0], [4, 0], [-10, 0]],
                                     [[0, 0], [0, 0], [0, 0]],
                                     [[1, 0], [2, 0], [6, 0]],
                                     [[10, 0], [8, 0], [2, 0]]])
        results = low_level.scalar_gradient_y_matrix_primitive(a, dy)
        self.assertTrue(np.array_equal(expected_results, results),
                        msg=f"expected: {expected_results}, actual: {results}")

    def test_scalar_gradient_y_matrix_primitive_dy_two(self):
        dy = 2.0
        a = np.array([[[3.0, 0], [4, 0], [7, 0]],
                      [[7, 0], [6, 0], [2, 0]],
                      [[3, 0], [4, 0], [7, 0]],
                      [[8, 0], [8, 0], [8, 0]]])
        
        expected_results = np.array([[[4.0, 0], [2, 0], [-5, 0]],
                                     [[0, 0], [0, 0], [0, 0]],
                                     [[0.5, 0], [1, 0], [3, 0]],
                                     [[5, 0], [4, 0], [1, 0]]])
        results = low_level.scalar_gradient_y_matrix_primitive(a, dy)
        self.assertTrue(np.array_equal(expected_results, results),
                        msg=f"expected: {expected_results}, actual: {results}")

    def test_scalar_gradient_y_2D_dy_one(self):
        dy = 1.0
        a = np.array([[3.0, 4, 7],
                      [7, 6, 2],
                      [3, 4, 7],
                      [8, 8, 8]])

        coord_one = [0,0]
        coord_two = [1,1]
        coord_three = [3,3]

        expected_result_one = 8
        expected_result_two = 0
        expected_result_three = 2

        result_one = low_level.scalar_gradient_y_2D(a, dy, a.shape[0], coord_one[0], coord_one[1])
        result_two = low_level.scalar_gradient_y_2D(a, dy, a.shape[0], coord_two[0], coord_two[1])
        result_three = low_level.scalar_gradient_y_2D(a, dy, a.shape[0], coord_three[0], coord_three[1])


        self.assertEqual(expected_result_one, result_one)
        self.assertEqual(expected_result_two, result_two)
        self.assertEqual(expected_result_three, result_three)

    def test_scalar_gradient_y_2D_dy_two(self):
        dy = 2.0
        a = np.array([[3.0, 4, 7],
                      [7, 6, 2],
                      [3, 4, 7],
                      [8, 8, 8]])

        coord_one = [0,0]
        coord_two = [1,1]
        coord_three = [3,3]

        expected_result_one = 4
        expected_result_two = 0
        expected_result_three = 1
        
        result_one = low_level.scalar_gradient_y_2D(a, dy, a.shape[0], coord_one[0], coord_one[1])
        result_two = low_level.scalar_gradient_y_2D(a, dy, a.shape[0], coord_two[0], coord_two[1])
        result_three = low_level.scalar_gradient_y_2D(a, dy, a.shape[0], coord_three[0], coord_three[1])


        self.assertEqual(expected_result_one, result_one)
        self.assertEqual(expected_result_two, result_two)
        self.assertEqual(expected_result_three, result_three)

