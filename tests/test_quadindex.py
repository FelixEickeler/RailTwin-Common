from railtwin_common.algorithms.quadindex import QuadIndex
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
        points = df.copy()
        distance = 1

        quad = QuadIndex.create_quadindex(points, distance)
        all_points_in_cell = quad.get_ro(x=0, y=1)
        all_points_in_cell_1 = quad.get_ro(x=0, y=0)
        all_points_in_cell_2 = quad.get_ro(x=1, y=0)
        self.assertEqual(2, len(all_points_in_cell))
        self.assertAlmostEqual(all_points_in_cell.iloc[0]["x"], 0.3)
        self.assertAlmostEqual(all_points_in_cell.iloc[0]["y"], 1.3)
        self.assertAlmostEqual(all_points_in_cell_1.iloc[0]["x"], 0.1)
        self.assertAlmostEqual(all_points_in_cell_1.iloc[0]["y"], 0.2)
        self.assertAlmostEqual(all_points_in_cell_2.iloc[0]["x"], 1.2)
        self.assertAlmostEqual(all_points_in_cell_2.iloc[0]["y"], 0.3)

    def test_cells_for_georeferenced_and_negative_coordinates(self):
        rng = np.random.default_rng(0)
        for x0, y0 in ((3420000.0, 5330000.0), (-5.0, -5.0), (0.0, 0.0)):
            points = pandas.DataFrame({"x": x0 + rng.uniform(0, 10, 2000), "y": y0 + rng.uniform(0, 10, 2000),
                                   "z": rng.uniform(0, 1, 2000)})
            quad = QuadIndex.create_quadindex(points, distance=1)
            seen = 0
            for row, col, cells in quad:
                cell = quad._sorted_points_ref[cells]
                seen += len(cell)
                # pandas.cut cells are right-closed, the first one includes its lower bound
                lo_x, lo_y = quad.min[0] + row, quad.min[1] + col
                self.assertTrue(((cell["x"] >= lo_x - 1e-9) & (cell["x"] <= lo_x + 1 + 1e-9)).all(), (x0, row, col))
                self.assertTrue(((cell["y"] >= lo_y - 1e-9) & (cell["y"] <= lo_y + 1 + 1e-9)).all(), (y0, row, col))
            self.assertEqual(seen, 2000)


if __name__ == '__main__':
    unittest.main()
