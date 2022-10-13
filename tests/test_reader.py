import json
import unittest
import numpy as np
from common.io import read
from pathlib import Path

from tests.files.cube import cube_test
from tests.helpers import balance_pointcloud

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
            self.assertTrue(mname in point_cloud.attrs["categories"])
            self.assertEqual(mdata, point_cloud.attrs["categories"][mname])

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
