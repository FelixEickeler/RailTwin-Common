from ..header import HeaderGenerator, as_offset
from plyfile import PlyData
from pathlib import Path
import numpy as np
import json
import pandas
from ...logger import RailTwinLogger

# connect to logging system
logger = RailTwinLogger.create()

_VERTEX_NAMES = ["vertex", "points", "point", "vertices"]


def read_ply(input_path: Path):
    """
    Read a .ply point cloud into a DataFrame. Known properties get their internal name and type (header.py), other
    properties are kept with their own name and type. Offset / metadata / categories are restored into attrs.
    """
    point_header = HeaderGenerator()
    point_cloud_ply = PlyData.read(input_path)
    metadata = {}
    category_mapping = {}

    vertices = None
    for ele in point_cloud_ply.elements:
        if ele.name in _VERTEX_NAMES:
            vertices = ele.data
            for col_name in vertices.dtype.names:
                if col_name.lower() in point_header.possible_properties():
                    point_header.add(col_name)
        if ele.name in ["shift", "offset"]:
            metadata["offset"] = as_offset(ele.data)
            metadata["original_offset_name"] = ele.name
    if vertices is None:
        raise ValueError(f"{input_path} has no vertex element (one of {_VERTEX_NAMES})")

    if any([c.find("dialect=tum_ply") > -1 for c in point_cloud_ply.comments]):
        for comment in point_cloud_ply.comments:
            if comment.startswith("metadata:"):
                metadata_update = json.loads(comment[9:])
                if "offset" in metadata_update:
                    metadata_update["offset"] = as_offset(metadata_update["offset"])
                    if "offset" in metadata:
                        # The offset element takes priority over the copy in the json comment
                        difference = metadata["offset"].view("3f8") - metadata_update["offset"].view("3f8")
                        if np.linalg.norm(difference) < 0.01:
                            logger.info("Metadata was defined twice. Discarding json offset")
                        else:
                            logger.warning(f"Two different offsets, the offset element is used: {metadata_update['offset']}")
                            metadata_update["I0"] = metadata_update["offset"]
                        del metadata_update["offset"]

                metadata.update(metadata_update)
            elif comment.startswith("categories:"):
                category_mapping = json.loads(comment[11:])

    # Known properties with their internal name and type, the others as they are
    series = {}
    for ply_name in vertices.dtype.names:
        if ply_name in point_header.mapping:
            internal_name = point_header.mapping[ply_name]
            series[internal_name] = pandas.Series(vertices[ply_name], dtype=point_header.header[internal_name].type)
        else:
            series[ply_name] = pandas.Series(vertices[ply_name])
    point_cloud = pandas.DataFrame(series)
    del series

    if metadata:
        point_cloud.attrs["metadata"] = metadata
    if category_mapping:
        point_cloud.attrs["category_mapping"] = category_mapping
    return point_cloud
