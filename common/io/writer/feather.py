import pandas
import pyarrow
from pyarrow import feather
from pathlib import Path

from ..io_options import IOOptions
from ..writer._common import extract_metadata
from ...logger import RailTwinLogger

logger = RailTwinLogger.create()


def feather_writer(output_path: Path, point_cloud: pandas.DataFrame, options: IOOptions):
    """Write a point cloud as .feather. The DataFrame of the caller is not changed."""
    # custom_index: keep the index as column "index", otherwise it is dropped
    point_cloud = point_cloud.reset_index(drop=not options.custom_index)
    if options.custom_index and "index" not in point_cloud:
        point_cloud = point_cloud.rename(columns={point_cloud.columns[0]: "index"})
    if not options.binary:
        logger.info("Feather files are always binary, please check your configuration")

    category_mapping, metadata = extract_metadata(point_cloud)
    new_metadata = {}
    if category_mapping != "{}":
        new_metadata[b"category_mapping"] = category_mapping.encode()
    if metadata != "{}":
        new_metadata[b"metadata"] = metadata.encode()

    arrow_table = pyarrow.Table.from_pandas(point_cloud)
    new_metadata.update(arrow_table.schema.metadata or {})
    arrow_table = arrow_table.replace_schema_metadata(new_metadata)
    feather.write_feather(df=arrow_table, dest=output_path.as_posix(), compression=options.compression)
