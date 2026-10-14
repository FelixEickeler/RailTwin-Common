import json
import unittest
import numpy as np
from railtwin_common.common.io import read
from pathlib import Path

from files.cube import cube_test
from helpers import balance_pointcloud

test_folder = Path(__file__).parent / "files"


class TestReader(unittest.TestCase):

    # def test_default_registration(self):
    #     self.assertIn("ply", read._)

    def test_ply_ascii(self):
        data = read(test_folder / "cube_ascii.ply")
        diff = data.compare(cube_test())
        self.assertTrue(diff.empty)

    def test_ply_binary(self):
        data = read(test_folder / "cube_bin.ply")
        diff = data.compare(cube_test())
        self.assertTrue(diff.empty)

    def test_ply_metadata(self):
        point_cloud = read(test_folder / "cube_ascii.ply")
        metadata = {
            "test_string": "string",
            "test_int": 1,
            "test_float": 1.0,
            "test_numpy": [1, 2, 3],
            "test_list": [1, 2, 3],
            "test_dict": {},
        }
        for mname, mdata in metadata.items():
            self.assertTrue(mname in point_cloud.attrs["metadata"])
            self.assertEqual(mdata, point_cloud.attrs["metadata"][mname])

    def test_ply_category(self):
        with open(test_folder / "example_categories.json") as json_file:
            categories = json.load(json_file)
        point_cloud = read(test_folder / "category_test.ply")
        for mname, mdata in categories.items():
            self.assertTrue(mname in point_cloud.attrs["category_mapping"])
            self.assertEqual(mdata, point_cloud.attrs["category_mapping"][mname])

    def test_feather(self):
        data = read(test_folder / "cube.feather")
        diff = data.compare(cube_test())
        self.assertTrue(diff.empty)

    # def test_feather_metadata(self):
    #     data = read(test_folder / "cube.feather")
    #     diff = data.compare(cube_test())
    #     self.assertTrue(diff.empty)

    def test_las0(self):
        data = read(test_folder / "cube.las")
        val = cube_test()
        balance_pointcloud(val)
        diff = data.compare(val)
        self.assertTrue(diff.empty)


def _write_las(path, n, **fields):
    import laspy
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [3420000.0, 5330000.0, 500.0]
    las = laspy.LasData(header)
    rng = np.random.default_rng(0)
    las.x = 3420000.0 + rng.uniform(0, 10, n)
    las.y = 5330000.0 + rng.uniform(0, 10, n)
    las.z = 500.0 + rng.uniform(0, 1, n)
    for name, value in fields.items():
        las[name] = np.full(n, value)
    las.write(str(path))


def test_las_keeps_constant_fields_at_non_default_values(tmp_path):
    from railtwin_common.common.io.reader.laz import read_laz
    _write_las(tmp_path / "a.las", 50, intensity=500, point_source_id=7, user_data=3, withheld=1, scan_angle_rank=-5)
    cloud = read_laz(tmp_path / "a.las")
    assert (cloud["point_source_id"] == 7).all() and (cloud["user_data"] == 3).all()
    assert (cloud["intensity"] == 1.0).all() and cloud["withheld"].all() and (cloud["scan_angle_rank"] == -5).all()
    # Fields at their default are left out
    assert "class" not in cloud and "key_point" not in cloud


def test_las_without_points(tmp_path):
    from railtwin_common.common.io.reader.laz import read_laz
    _write_las(tmp_path / "empty.las", 0)
    assert len(read_laz(tmp_path / "empty.las")) == 0
