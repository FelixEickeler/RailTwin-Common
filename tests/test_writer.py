import json
import logging

import numpy as np
import pandas
import pytest
from pathlib import Path

from railtwin_common.common.io import read, write
from railtwin_common.common.io.augmentations import TA, Normalize
from railtwin_common.common.io.header import HeaderGenerator
from railtwin_common.common.io.io_options import IOOptions
from railtwin_common.common.io.read_router import scandir_supported
from railtwin_common.common.io.reader.ply import read_ply
from railtwin_common.common.io.writer.feather import feather_writer
from railtwin_common.common.io.writer.ply import ply_writer
from railtwin_common.common.logger import RailTwinLogger

test_folder = Path(__file__).parent / "files"

# Georeferenced coordinates (UTM): float32 would round them to about 0.1 m
UTM = np.array([691234.567, 5334123.456, 520.123])


@pytest.fixture
def cube():
    return read(test_folder / "cube_ascii.ply")


def utm_cloud(n=100, seed=0):
    rng = np.random.default_rng(seed)
    xyz = UTM + rng.uniform(0, 50, (n, 3))
    return pandas.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                             "class": rng.integers(0, 5, n).astype(np.uint32)})


# ----------------------------------------------------------------------------------------------------------------------
# PLY
# ----------------------------------------------------------------------------------------------------------------------

def test_ply_local_system_round_trip(tmp_path, cube):
    options = IOOptions(binary=False, local=True, xyz_accuracy="<f4", intensity=None, reduce_to=None, filetype=None,
                        custom_index=False)
    ply_writer(tmp_path / "balanced.ply", cube, options=options)
    back = read(tmp_path / "balanced.ply")
    offset = back.attrs["metadata"]["offset"].view("3f8").reshape(3)
    # Centred, and centre + offset gives the original coordinates
    xyz = back[["x", "y", "z"]].to_numpy(np.float64)
    np.testing.assert_allclose(xyz.min(axis=0), -xyz.max(axis=0), atol=1e-6)
    np.testing.assert_allclose(xyz + offset, cube[["x", "y", "z"]].to_numpy(), atol=1e-5)


def test_ply_do_nothing_keeps_float64_coordinates(tmp_path):
    cloud = utm_cloud()
    ply_writer(tmp_path / "utm.ply", cloud, options=IOOptions.do_nothing())
    back = read(tmp_path / "utm.ply")
    assert back["x"].dtype == np.float64
    np.testing.assert_array_equal(back[["x", "y", "z"]].to_numpy(), cloud[["x", "y", "z"]].to_numpy())


def test_writers_do_not_change_the_callers_dataframe(tmp_path):
    cloud = utm_cloud()
    cloud.attrs["metadata"] = {"source": "test"}
    before = cloud.copy()
    for options in (IOOptions.do_nothing(), IOOptions.default(), IOOptions.only_localization()):
        ply_writer(tmp_path / "a.ply", cloud, options=options)
        feather_writer(tmp_path / "a.feather", cloud, options=options)
    pandas.testing.assert_frame_equal(cloud, before)
    assert cloud.attrs["metadata"] == {"source": "test"}


def test_ply_metadata_round_trip(tmp_path):
    header = HeaderGenerator.create_xyzic()
    cloud = header.create_dataframe(np.array([(0, 0, 0, 0, 48)], dtype=header.numpy_dtypes()))
    cloud.attrs["metadata"] = {"test_string": "string", "test_int": 1, "test_float": 1.0,
                               "test_numpy": np.array([1, 2, 3]), "test_list": [1, 2, 3], "test_dict": {}}
    ply_writer(tmp_path / "metadata.ply", cloud, options=IOOptions.do_nothing())
    metadata = read(tmp_path / "metadata.ply").attrs["metadata"]
    assert metadata == {"test_string": "string", "test_int": 1, "test_float": 1.0,
                        "test_numpy": [1, 2, 3], "test_list": [1, 2, 3], "test_dict": {}}


def test_ply_category_mapping_round_trip(tmp_path):
    with open(test_folder / "example_categories.json") as json_file:
        categories = json.load(json_file)
    cloud = utm_cloud(len(categories))
    cloud.attrs["category_mapping"] = categories
    ply_writer(tmp_path / "category.ply", cloud, options=IOOptions.do_nothing())
    assert read(tmp_path / "category.ply").attrs["category_mapping"] == categories


def test_ply_keeps_unknown_columns(tmp_path):
    cloud = utm_cloud()
    cloud["label"] = np.arange(len(cloud), dtype=np.int32)
    cloud["my_score"] = np.linspace(0, 1, len(cloud), dtype=np.float32)
    ply_writer(tmp_path / "extra.ply", cloud, options=IOOptions.do_nothing())
    back = read_ply(tmp_path / "extra.ply")
    np.testing.assert_array_equal(back["label"], cloud["label"])
    np.testing.assert_array_equal(back["my_score"], cloud["my_score"])


def test_ply_64_bit_integers(tmp_path):
    cloud = utm_cloud()
    cloud["object_id"] = np.arange(len(cloud), dtype=np.int64)
    ply_writer(tmp_path / "int.ply", cloud, options=IOOptions.do_nothing())
    np.testing.assert_array_equal(read(tmp_path / "int.ply")["object_id"], cloud["object_id"])

    cloud["object_id"] = np.int64(2 ** 33)
    with pytest.raises(OverflowError):
        ply_writer(tmp_path / "overflow.ply", cloud, options=IOOptions.do_nothing())


def test_ply_float16_intensity(tmp_path):
    cloud = utm_cloud()
    cloud["intensity"] = np.linspace(0, 1, len(cloud)).astype(np.float16)
    ply_writer(tmp_path / "f16.ply", cloud, options=IOOptions.do_nothing())
    np.testing.assert_allclose(read(tmp_path / "f16.ply")["intensity"], cloud["intensity"].astype(np.float32))


def test_ply_without_vertices(tmp_path):
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(np.zeros(1, dtype=[("a", "f4")]), "faces")], text=True).write(tmp_path / "x.ply")
    with pytest.raises(ValueError):
        read_ply(tmp_path / "x.ply")


# ----------------------------------------------------------------------------------------------------------------------
# Feather
# ----------------------------------------------------------------------------------------------------------------------

def test_feather_round_trip(tmp_path, cube):
    cube["class"] = np.arange(len(cube), dtype=np.uint32)
    cube.attrs["category_mapping"] = {"rail": 1}
    feather_writer(tmp_path / "cube.feather", cube, options=IOOptions.do_nothing())
    back = read(tmp_path / "cube.feather")
    pandas.testing.assert_frame_equal(back, cube, check_like=True)
    assert back.attrs["category_mapping"] == {"rail": 1}
    assert "metadata" in back.attrs


def test_feather_custom_index(tmp_path):
    cloud = utm_cloud(10)
    cloud.index = np.arange(100, 110)
    options = IOOptions.do_nothing()
    options.custom_index = True
    feather_writer(tmp_path / "index.feather", cloud, options=options)
    back = read(tmp_path / "index.feather")
    np.testing.assert_array_equal(back.index, cloud.index)
    np.testing.assert_array_equal(back["x"], cloud["x"])


def test_feather_reads_the_old_class_mapping_key(tmp_path):
    import pyarrow
    from pyarrow import feather
    table = pyarrow.Table.from_pandas(utm_cloud(5), preserve_index=False)
    table = table.replace_schema_metadata({b"class_mapping": json.dumps({"rail": 1}).encode()})
    feather.write_feather(table, (tmp_path / "old.feather").as_posix())
    assert read(tmp_path / "old.feather").attrs["category_mapping"] == {"rail": 1}


# ----------------------------------------------------------------------------------------------------------------------
# Router, augmentations, logger
# ----------------------------------------------------------------------------------------------------------------------

def test_write_router_and_scandir_ignore_case(tmp_path):
    cloud = utm_cloud(5)
    write(tmp_path / "a.ply", cloud, options=IOOptions.do_nothing())
    (tmp_path / "sub").mkdir()
    write(tmp_path / "sub" / "b.feather", cloud, options=IOOptions.do_nothing())
    (tmp_path / "a.ply").rename(tmp_path / "A.PLY")
    (tmp_path / "notes.txt").write_text("no point cloud")
    found = sorted(p.name for p in scandir_supported(tmp_path))
    assert found == ["A.PLY", "b.feather"]


def test_normalize_does_not_keep_state():
    augmentation = TA(np.float32, Normalize())
    data = np.array([0.0, 50.0, 100.0])
    first = augmentation.augment(data)
    second = augmentation.augment(data)
    np.testing.assert_array_equal(first, second)
    assert first.max() == 1.0


def test_pandas_options_are_not_changed(tmp_path, cube):
    before = pandas.options.mode.chained_assignment
    ply_writer(tmp_path / "a.ply", cube, options=IOOptions.default())
    assert pandas.options.mode.chained_assignment == before


def test_logger_file(tmp_path):
    logger = RailTwinLogger.create()
    assert isinstance(logger, logging.Logger) and logger is RailTwinLogger.create()
    log_file = RailTwinLogger.register_file_logger(tmp_path)
    try:
        logger.warning("hello file")
        for handler in logger.handlers:
            handler.flush()
        assert "hello file" in log_file.read_text()
    finally:
        for handler in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
            logger.removeHandler(handler)
            handler.close()
