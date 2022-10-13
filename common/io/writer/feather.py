import json
import pandas
import pyarrow
from pyarrow import feather
from pathlib import Path

from ..io_options import IOOptions
from ..writer._common import extract_metadata
from ...logger import RailTwinLogger

logger = RailTwinLogger.create()


def feather_writer(output_path: Path, point_cloud: pandas.DataFrame, options: IOOptions):
    if options.custom_index:
        point_cloud.reset_index(inplace=True)
    else:
        point_cloud.reset_index(drop=True, inplace=True)
    if not options.binary:
        logger.info("Feather files are always binary, please check your configuration")

    category_mapping, metadata = extract_metadata(point_cloud)
    new_metadata = {}
    if category_mapping != "{}":
        new_metadata["class_mapping".encode()] = category_mapping.encode()
    if metadata != "{}":
        new_metadata["metadata".encode()] = metadata.encode()
    if "offset" in point_cloud.attrs:
        new_metadata["offset".encode()] = (json.dumps(point_cloud.attrs["offset"].tolist()).encode())

    arrow_table = pyarrow.Table.from_pandas(point_cloud)
    arrow_metadata = arrow_table.schema.metadata
    new_metadata.update(arrow_metadata)
    arrow_table = arrow_table.replace_schema_metadata(new_metadata)
    feather.write_feather(df=arrow_table, dest=output_path.as_posix(), compression=options.compression)

    if options.custom_index:
        point_cloud.set_index("index", drop=True)
