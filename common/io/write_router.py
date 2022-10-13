from pathlib import Path
import pandas
from .io_options import IOOptions
from .writer.feather import feather_writer
from .writer.ply import ply_writer
from ..logger import RailTwinLogger

logger = RailTwinLogger.create()

registered_writer = {".ply": ply_writer, ".PLY": ply_writer, ".feather": feather_writer}


def write(output_path: Path, point_cloud: pandas.DataFrame, options=IOOptions.default()):
    filetype = output_path.suffix.lower()
    if filetype in registered_writer:
        xyz = registered_writer[filetype](output_path, point_cloud, options)
    else:
        raise NotImplementedError("This filetype is not supported !")
    return xyz


