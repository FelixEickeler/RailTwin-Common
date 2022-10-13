from algorithms.quadindex import QuadIndex
import unittest
import pandas
import numpy as np


# data = []
# for x in np.arange(0, 3, 0.1):
#     for y in np.arange(0, 3, 0.1):
#         data.append([x, y])
#         if x >= 2 and y > 2:
#             del [x, y]


df = pandas.DataFrame(np.array([[0.1, 0.2], [0.3, 1.3], [0.3, 1.6], [0.2, 2.3], [1.3, 1.3], [1.2, 0.3], [2.5, 0.8]]),
                      columns=['x', 'y'])

# [1, 3, 4]
# [5, 6, 6]
# [7, 7, 7]


class TestQuadindex(unittest.TestCase):

    # def test_get(self):
    #     points = QuadIndex.get_xy(input_path)
    #     self.assertTrue(points.shape[0] == 181)

    def test_index(self):
        points = df
        distance = 1

        quad = QuadIndex.create_quadindex(points, distance)
        all_points_in_cell = quad.get(x=0, y=1)
        all_points_in_cell_1 = quad.get(x=0, y=0)
        all_points_in_cell_2 = quad.get(x=1, y=0)
        self.assertTrue(2, len(all_points_in_cell))
        self.assertAlmostEqual(all_points_in_cell.iloc[0]["x"], 0.3)
        self.assertAlmostEqual(all_points_in_cell.iloc[0]["y"], 1.3)
        self.assertAlmostEqual(all_points_in_cell_1.iloc[0]["x"], 0.1)
        self.assertAlmostEqual(all_points_in_cell_1.iloc[0]["y"], 0.2)
        self.assertAlmostEqual(all_points_in_cell_2.iloc[0]["x"], 1.2)
        self.assertAlmostEqual(all_points_in_cell_2.iloc[0]["y"], 0.3)


if __name__ == '__main__':
    unittest.main()
