from pyarrow import feather
from ..header import as_offset
from ...logger import RailTwinLogger
from pathlib import Path
import json

# connect to logging system
logger = RailTwinLogger.create()


def read_feather(input_path: Path):
    arrow_table = feather.read_table(input_path)
    point_cloud = arrow_table.to_pandas()
    schema_metadata = arrow_table.schema.metadata or {}

    # restore metadata ("class_mapping" was written by older versions)
    for key in (b"category_mapping", b"class_mapping"):
        if key in schema_metadata:
            point_cloud.attrs["category_mapping"] = json.loads(schema_metadata[key])
            break
    metadata = json.loads(schema_metadata[b"metadata"]) if b"metadata" in schema_metadata else {}
    if b"offset" in schema_metadata and "offset" not in metadata:
        metadata["offset"] = json.loads(schema_metadata[b"offset"])  # written by older versions
    if "offset" in metadata:
        metadata["offset"] = as_offset(metadata["offset"])
    if metadata:
        point_cloud.attrs["metadata"] = metadata

    if "index" in point_cloud:
        point_cloud = point_cloud.set_index("index")
        point_cloud.index.name = None

    return point_cloud
