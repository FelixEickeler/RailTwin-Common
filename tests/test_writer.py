import unittest
import json
import pandas
import numpy as np
import os

from pathlib import Path

from common.io.header import HeaderGenerator
from common.io.io_options import IOOptions
from tests.helpers import cmpfile_by_lines, cmpfile_by_binary
from common.io import read
from common.io.writer.ply import ply_writer
from common.io.writer.feather import feather_writer

test_folder = Path(__file__).parent / "files"
tmp_path = Path(__file__).parent / "_tmp"


class TestWriter(unittest.TestCase):
    point_cloud = None

    @classmethod
    def setUpClass(cls):
        cls.point_cloud = read(test_folder / "cube_ascii.ply")
        files_in_folder = tmp_path.glob("*.*")
        for f in files_in_folder:
            os.remove(f)
        tmp_path.mkdir(exist_ok=True, parents=True)

    # @classmethod
    # def tearDownClass(cls):
    #     files_in_folder = tmp_path.glob("*.*")
    #     for f in files_in_folder:
    #         os.remove(f)
    #     os.rmdir(tmp_path)

    def test_ply_xyz_balanced(self):
        write_options = IOOptions(binary=False, local=True, xyz_accuracy="<f4", intensity=None, reduce_to=None, filetype=None, custom_index=False)
        ply_writer(tmp_path / "cube_ascii_balanced.ply", self.point_cloud, options=write_options)
        self.assertTrue(cmpfile_by_lines(tmp_path / "cube_ascii_balanced.ply", test_folder / "cube_ascii_balanced.ply"))

    # def test_offset_output_write_ply(self):
    #     header = HeaderGenerator.create_xyzic()
    #     tmp_pointcloud = header.create_dataframe(
    #         np.array([(1, 3, 5, 0, 1), (1, 2, 3, 1, 48)], dtype=header.numpy_dtypes())
    #     )
    #     # local system does center the coordinate system
    #     ply_writer(tmp_path / "offset_test.ply", tmp_pointcloud, options={"local_system"})
    #     self.assertTrue(cmpfile_by_lines(tmp_path / "offset_test.ply", test_folder / "offset_test.ply"))

    def test_ply_metadata(self):
        header = HeaderGenerator.create_xyzic()
        tmp_pointcloud = header.create_dataframe(np.array([(0, 0, 0, 0, 48)], dtype=header.numpy_dtypes()))
        metadata = {
            "test_string": "string",
            "test_int": 1,
            "test_float": 1.0,
            "test_numpy": np.array([1, 2, 3]),
            "test_list": [1, 2, 3],
            "test_dict": {},
        }
        tmp_pointcloud.attrs["metadata"] = metadata
        do_nothing = IOOptions.do_nothing()
        ply_writer(tmp_path / "metadata_test.ply", tmp_pointcloud, options=do_nothing)
        self.assertTrue(cmpfile_by_lines(tmp_path / "metadata_test.ply", test_folder / "metadata_test.ply"))

    def test_ply_appended(self):
        with open(test_folder / "example_categories.json") as json_file:
            categories = json.load(json_file)

        tmp_pointcloud = HeaderGenerator.create_xyzic().create_dataframe(np.zeros(len(categories),
                                                                                  dtype=[("x", "f4"), ("y", "f4"),
                                                                                         ("z", "f4"),
                                                                                         ("intensity", "f4"),
                                                                                         ("class", "i4")]))
        tmp_pointcloud["class"] = pandas.Series(np.arange(len(categories)), dtype=np.int32)
        tmp_pointcloud["class"] = tmp_pointcloud["class"].astype("category")
        tmp_pointcloud.attrs["category_mapping"] = categories
        tmp_pointcloud["class"].cat.rename_categories(categories)
        do_nothing = IOOptions.do_nothing()
        ply_writer(tmp_path / "category_test.ply", tmp_pointcloud, options=do_nothing)

    def test_feather(self):
        write_options = IOOptions.do_nothing()
        feather_writer(tmp_path / "cube.feather", self.point_cloud, options=write_options)
        self.assertTrue(cmpfile_by_binary(tmp_path / "cube.feather", test_folder / "cube.feather"))
