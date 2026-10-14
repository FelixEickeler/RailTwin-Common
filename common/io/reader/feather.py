from pyarrow import feather
from ..header import HeaderGenerator
from ...logger import RailTwinLogger
from pathlib import Path
import json
import numpy as np

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
    if b"metadata" in schema_metadata:
        point_cloud.attrs["metadata"] = json.loads(schema_metadata[b"metadata"])
    if b"offset" in schema_metadata:
        offset = np.array(json.loads(schema_metadata[b"offset"]), dtype="f8").reshape(1, 3)
        point_cloud.attrs["offset"] = offset.view(HeaderGenerator.offset()).reshape(1)

    if "index" in point_cloud:
        point_cloud = point_cloud.set_index("index")
        point_cloud.index.name = None

    return point_cloud
