from pathlib import Path

import numpy as np
import pandas
from plyfile import PlyElement, PlyData

from ..data_shaping import data_augmentations
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

    data = point_cloud.to_records(index=False)
    data = data.astype([(name, _ply_dtype(name, data[name])) for name in data.dtype.names])

    output = []
    categories, meta_comment = extract_metadata(point_cloud)

    # offset and vertices instance saved into output property called offset
    if "offset" in point_cloud.attrs["metadata"]:
        if isinstance(point_cloud.attrs["metadata"]["offset"], np.ndarray):
            output.append(PlyElement.describe(point_cloud.attrs["metadata"]["offset"], 'offset'))
        else:
            logger.warning("Offset type must be ndarray, it is only written into the metadata comment")

    # vertices
    output.append(PlyElement.describe(data, 'vertices'))

    comment = ["dialect=tum_ply",
               "metadata:" + meta_comment,
               "categories:" + categories]
    PlyData(output, text=not options.binary, comments=comment).write(output_path)
