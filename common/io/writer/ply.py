from pathlib import Path

import numpy as np
import pandas
from plyfile import PlyElement, PlyData

from ..data_shaping import data_augmentations
from ..header import as_offset
from ..writer._common import extract_metadata
from ...logger import RailTwinLogger

logger = RailTwinLogger.create()

# PLY has no 64 bit integers and no float16: (numpy kind, item size) -> type written
_PLY_TYPES = {("i", 8): np.int32, ("u", 8): np.uint32, ("f", 2): np.float32}


def _ply_dtype(name, column):
    """Type a column is written with. 64 bit integers become 32 bit, if their values fit."""
    dtype = column.dtype
    target = _PLY_TYPES.get((dtype.kind, dtype.itemsize))
    if target is None:
        return dtype
    if dtype.kind in "iu" and len(column) and \
            (column.min() < np.iinfo(target).min or column.max() > np.iinfo(target).max):
        raise OverflowError(f"Column '{name}' does not fit into {np.dtype(target).name}, which PLY needs")
    return np.dtype(target)


def ply_writer(output_path: Path, point_cloud: pandas.DataFrame, options):
    """Write a point cloud as .ply. The DataFrame of the caller is not changed."""
    point_cloud = point_cloud.copy()
    point_cloud.attrs = {key: (value.copy() if isinstance(value, dict) else value)
                         for key, value in point_cloud.attrs.items()}
    data_augmentations(point_cloud, options)

    # PLY has no categorical type: write the codes, the categories go into the metadata ("class": category_mapping)
    for name in [n for n in point_cloud.columns if isinstance(point_cloud[n].dtype, pandas.CategoricalDtype)]:
        categories = [str(c) for c in point_cloud[name].cat.categories]
        if name == "class" and "category_mapping" not in point_cloud.attrs:
            point_cloud.attrs["category_mapping"] = {c: i for i, c in enumerate(categories)}
        else:
            point_cloud.attrs["metadata"].setdefault("categories", {})[name] = categories
        point_cloud[name] = point_cloud[name].cat.codes.astype(np.int32)

    data = point_cloud.to_records(index=False)
    data = data.astype([(name, _ply_dtype(name, data[name])) for name in data.dtype.names])

    output = []
    categories, meta_comment = extract_metadata(point_cloud)

    # The offset is written as element "offset" (and in the metadata comment)
    if "offset" in point_cloud.attrs["metadata"]:
        point_cloud.attrs["metadata"]["offset"] = as_offset(point_cloud.attrs["metadata"]["offset"])
        output.append(PlyElement.describe(point_cloud.attrs["metadata"]["offset"], 'offset'))

    # vertices
    output.append(PlyElement.describe(data, 'vertices'))

    comment = ["dialect=tum_ply",
               "metadata:" + meta_comment,
               "categories:" + categories]
    PlyData(output, text=not options.binary, comments=comment).write(output_path)
