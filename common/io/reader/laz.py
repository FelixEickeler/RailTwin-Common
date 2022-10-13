from ..augmentations import TA, UnpackBinary, TrueValue
from ..data_shaping import data_augmentations
from ..header import HeaderGenerator, _allowed_headers
from ..io_options import IOOptions
from ...logger import RailTwinLogger
from pathlib import Path
import numpy as np
from pandas import DataFrame
import laspy

# connect to logging system
logger = RailTwinLogger.create()


def read_laz(input_path: Path):
    logger.info("Reading points from {}".format(input_path))
    las = laspy.read(input_path)
    header = HeaderGenerator()
    ignore_default = False

    #TODO check how to fix this

    if las.point_format.id < 6:
        header.add("x").add("y").add("z")
        # reconstruct points from las format scale(f8) * val(i4) + offset(f8)
        xyz = np.array([las.X, las.Y, las.Z]).transpose() * las.header.scale
        point_cloud = DataFrame.from_records(xyz, columns=header.header.keys())

        # preset offset
        # offset = records.fromarrays(las.header.offset, dtype=HeaderGenerator.offset())
        offset = las.header.offset.view(HeaderGenerator.offset())
        point_cloud.attrs["offset"] = offset

        # This will also update the offset and center the point  cloud so that min_n == -max_n
        base_options = IOOptions.only_localization()
        base_options.xyz_accuracy = np.float32
        data_augmentations(point_cloud, base_options)

        # Fill in scalar fields
        if np.unique(las.intensity).size > 1:
            point_cloud["intensity"] = _allowed_headers["intensity"].augment(las.intensity)

        # BitField 1
        tmp = TA(np.uint8, UnpackBinary(las.return_number.bit_mask)).augment(las.return_number.array)
        if np.unique(tmp).size > 1 or tmp[0] != 1 or ignore_default:
            point_cloud["return_number"] = tmp
        tmp = TA(np.uint8, UnpackBinary(las.number_of_returns.bit_mask)).augment(las.number_of_returns.array)
        if np.unique(tmp).size > 1 or tmp[0] != 1 or ignore_default:
            point_cloud["number_of_returns"] = tmp
        tmp = TA(bool, UnpackBinary(las.scan_direction_flag.bit_mask)).augment(las.scan_direction_flag.array)
        if np.unique(tmp).size > 1 or tmp[0] != 0 or ignore_default:
            point_cloud["scan_direction_flag"] = tmp
        tmp = TA(bool, UnpackBinary(las.edge_of_flight_line.bit_mask)).augment(las.edge_of_flight_line.array)
        if np.unique(tmp).size > 1 or tmp[0] != 0 or ignore_default:
            point_cloud["edge_of_flight_line"] = tmp

        # BitField 2
        tmp = TA(np.uint8, UnpackBinary(las.classification.bit_mask)).augment(las.classification.array)
        if np.unique(tmp).size > 1 or tmp[0] != 0 or ignore_default:
            point_cloud["classification"] = tmp
        tmp = TA(np.uint8, UnpackBinary(las.synthetic.bit_mask)).augment(las.synthetic.array)
        if np.unique(tmp).size > 1 or tmp[0] != 0 or ignore_default:
            point_cloud["synthetic"] = tmp
        tmp = TA(np.uint8, UnpackBinary(las.key_point.bit_mask)).augment(las.key_point.array)
        if np.unique(tmp).size > 1 or tmp[0] != 0 or ignore_default:
            point_cloud["key_point"] = tmp

        # additional f6
        if np.unique(las.user_data).size > 1:
            point_cloud["user_data"] = TA(np.float16, TrueValue()).augment(las.user_data)
        if np.unique(las.point_source_id).size > 1:
            point_cloud["point_source_id"] = TA(np.float16, TrueValue()).augment(las.point_source_id)

        if las.point_format.id >= 1:
            raise NotImplementedError("This format is present in laspy, but not implemented in this tool !")

        elif las.point_format.id > 0:
            logger.warn(f"Las format {las.point_format.id} is only supported by base attributes.")

    else:
        raise NotImplementedError("Only 0-Based las formats are supported")
    logger.info("{} points where added.".format(point_cloud.shape[0]))
    return point_cloud
