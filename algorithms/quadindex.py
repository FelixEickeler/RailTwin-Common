import math
import numpy as np
import pandas


class QuadIndex:

    def __init__(self):
        self._index_grid = []
        # Should be moved to association of top
        self._sorted_points_ref = None
        self.min = None
        self.max = None
        self.spacing = 0
        self.cells = 0

    def __iter__(self):
        for row in range(self._index_grid.shape[0]):
            for col in range(self._index_grid.shape[1]):
                begin, end = self.get_begin_end(row, col)
                yield row, col, np.s_[begin:end]

    def get_begin_end(self, x, y):
        end = self._index_grid[x][y]
        if y == 0:
            if x > 0:
                begin = self._index_grid[x - 1][-1]
            else:
                begin = 0
        else:
            begin = self._index_grid[x][y - 1]

        return begin, end

    def get_ro(self, x, y):
        begin, end = self.get_begin_end(x, y)
        return self._sorted_points_ref[:][begin:end]

    @staticmethod
    def create_quadindex(points, distance=1, debug=False):

        # 1. get the number of cells needed for this distance:
        x_min = min(points["x"])
        x_max = max(points["x"])
        y_min = min(points["y"])
        y_max = max(points["y"])

        # span_x = x_max - x_min
        # span_y = y_max - y_min

        # 2. create bins => include lowest
        def create_bins(start, stop, spacing):
            start = spacing * np.floor(start / spacing)
            stop = spacing * np.ceil(stop / spacing)
            upper_bounds = np.arange(start=start, stop=stop + distance, step=distance)
            return upper_bounds, start, stop

        x_bins_bounds, ix_0, ix_n = create_bins(start=x_min, stop=x_max, spacing=distance)
        y_bins_bounds, iy_0, iy_n = create_bins(start=y_min, stop=y_max, spacing=distance)

        # 3. put stuff in the bins
        x_labels = range(0, len(x_bins_bounds) - 1)
        y_labels = range(0, len(y_bins_bounds) - 1)
        cut_x = pandas.cut(x=points["x"], bins=x_bins_bounds, labels=x_labels, include_lowest=True)
        cut_y = pandas.cut(x=points["y"], bins=y_bins_bounds, labels=y_labels, include_lowest=True)

        # 5 create hashes, sort by them, and group again
        points["coord_hash"] = cut_x.to_numpy() * 1e12 + cut_y.to_numpy() * 1e6 + (1e6 + points["x"].to_numpy())
        points.sort_values(by=["coord_hash"], inplace=True, ignore_index=True)
        hash_bins = [x * 1e12 + y * 1e6 + 1e6 for x in x_labels for y in y_labels]
        qi = QuadIndex()
        qi._sorted_points_ref = points
        qi._index_grid = np.searchsorted(points["coord_hash"], hash_bins, side="left").reshape([len(x_labels), len(y_labels)])
        qi.cells = qi._index_grid.size
        iz_0 = distance * np.floor(points.iloc[0]["z"] / distance)
        iz_n = distance * np.ceil(points.iloc[-1]["z"] / distance)
        qi.min = np.array([ix_0, iy_0, iz_0])
        qi.max = np.array([ix_n, iy_n, iz_n])
        qi.spacing = distance
        if not debug:
            points.drop(["coord_hash"], axis=1)

        return qi

        # grid[x_i][y_i] = p_i

    @staticmethod
    def create_quadindex_slow(points, distance=1):
        points.sort_values(by=['x'], inplace=True, ignore_index=True)

        # applied fix
        n_x = int(np.abs(math.ceil(max(points["x"])) - math.floor(min(points["x"])) / distance))
        n_y = int(np.abs(math.ceil(max(points["y"])) - math.floor(min(points["y"])) / distance))

        grid = np.zeros([n_x, n_y], dtype=np.int32)
        end = 0
        for x_i in range(0, n_x):
            threshold_x = distance * (x_i + 1)
            for point_x in points["x"][end:]:
                if point_x > threshold_x:
                    grid[x_i][-1] = end
                    break
                end += 1
        grid[-1][-1] = end

        _begin = 0
        # für jede X-Zelle
        for x_i in range(0, n_x):
            _end = grid[x_i][-1]
            points[:][_begin:_end] = points[:][_begin:_end].sort_values(by=['y'], ignore_index=True)
            p_i = _begin
            # für jede Y-Zelle
            for y_i in range(0, n_y - 1):
                threshold_y = distance * (y_i + 1)
                # für jeden Punkt in Zelle
                for point_y in points["y"][p_i:_end]:
                    if point_y > threshold_y:
                        grid[x_i][y_i] = p_i
                        break
                    p_i += 1
                grid[x_i][y_i] = p_i
            _begin = _end

        qi = QuadIndex()
        qi._sorted_points_ref = points
        qi._index_grid = grid
        return qi
