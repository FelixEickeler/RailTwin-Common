from pyarrow import feather
from ..header import HeaderGenerator
from ...logger import RailTwinLogger as logger, RailTwinLogger
from pathlib import Path
import json
# connect to logging system
logger = RailTwinLogger.create()



def read_feather(input_path: Path):
    arrow_table = feather.read_table(input_path)
    point_cloud = arrow_table.to_pandas()
    # restore metadata
    if "category_mapping".encode() in arrow_table.schema.metadata:
        point_cloud.attrs["category_mapping"] = json.loads(arrow_table.schema.metadata["category_mapping".encode()])
    if "metadata".encode() in arrow_table.schema.metadata:
        point_cloud.attrs["metadata"] = json.loads(arrow_table.schema.metadata["metadata".encode()])
    if "offset".encode() in arrow_table.schema.metadata:
        point_cloud.attrs["offset"] = np.array(json.loads(arrow_table.schema.metadata["offset".encode()])).view(HeaderGenerator.offset())

    if "index" in point_cloud:
        point_cloud.index = point_cloud["index"]
        point_cloud.drop("index")

    return point_cloud