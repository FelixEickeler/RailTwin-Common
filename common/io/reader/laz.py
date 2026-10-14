from ..augmentations import TA, UnpackBinary, Normalize
from ..data_shaping import data_augmentations
from ..io_options import IOOptions
from ...logger import RailTwinLogger
from pathlib import Path
import numpy as np
from pandas import DataFrame
import laspy

# connect to logging system
logger = RailTwinLogger.create()


def read_laz(input_path: Path):
    """
    Read a .las / .laz file (point formats 0-5) into a DataFrame with the internal column names (see header.py).
    The points are centred (local system), the full offset to world coordinates is in attrs["metadata"]["offset"].
    Fields that are constant over the whole cloud at their default value are left out.
    """
    logger.info("Reading points from {}".format(input_path))
    las = laspy.read(input_path)
    if las.point_format.id >= 6:
        raise NotImplementedError("Only LAS point formats 0-5 are supported")

    # World coordinates (laspy applies scale and offset), then centred: float32 is not precise enough for them
    point_cloud = DataFrame({"x": np.asarray(las.x, dtype=np.float64),
                             "y": np.asarray(las.y, dtype=np.float64),
                             "z": np.asarray(las.z, dtype=np.float64)})
    base_options = IOOptions.only_localization()
    base_options.xyz_accuracy = np.float32
    data_augmentations(point_cloud, base_options)

    # Fill in scalar fields
    if np.unique(las.intensity).size > 1:
        point_cloud["intensity"] = TA(np.float32, Normalize()).augment(np.asarray(las.intensity))

    # Bit fields: (las name, internal name, dtype, default value)
    for las_name, name, dtype, default in [("return_number", "returnNumber", np.uint8, 1),
                                           ("number_of_returns", "numberOfReturns", np.uint8, 1),
                                           ("scan_direction_flag", "scan_direction_flag", bool, 0),
                                           ("edge_of_flight_line", "edge_of_flight_line", bool, 0),
                                           ("classification", "class", np.uint8, 0),
                                           ("synthetic", "synthetic", np.uint8, 0),
                                           ("key_point", "key_point", np.uint8, 0)]:
        field = las[las_name]
        values = TA(dtype, UnpackBinary(field.bit_mask)).augment(field.array)
        if np.unique(values).size > 1 or values[0] != default:
            point_cloud[name] = values

    for name, dtype in [("user_data", np.uint8), ("point_source_id", np.uint16)]:
        if np.unique(las[name]).size > 1:
            point_cloud[name] = np.asarray(las[name], dtype=dtype)

    # Formats 1, 3, 4, 5: GPS time; 2, 3, 5: colour
    dimensions = set(las.point_format.dimension_names)
    if "gps_time" in dimensions:
        point_cloud["gps_time"] = np.asarray(las.gps_time, dtype=np.float64)
    if {"red", "green", "blue"} <= dimensions:
        # LAS stores 16 bit colour, the internal format is 8 bit (some writers store 8 bit values in the 16 bits)
        shift = 8 if max(int(np.max(las[c])) for c in ("red", "green", "blue")) > 255 else 0
        for colour in ("red", "green", "blue"):
            point_cloud[colour] = (np.asarray(las[colour]) >> shift).astype(np.uint8)

    logger.info("{} points where added.".format(point_cloud.shape[0]))
    return point_cloud
