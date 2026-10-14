import numpy as np
from pandas import DataFrame
from .augmentations import apply_augmentations, TypeAugmentation, LocalSystem, TrueValue, GobalOffset
from .io_options import IOOptions
from .header import HeaderGenerator
from ..logger import RailTwinLogger

logger = RailTwinLogger.create()


def _shift_xyz(dataframe: DataFrame, xyz_dtype, make_shift):
    """Shift x, y, z (make_shift() gives one augmentation per axis) and add the shift to the metadata offset."""
    augmentations = {axis: TypeAugmentation(xyz_dtype, make_shift(i)) for i, axis in enumerate("xyz")}
    apply_augmentations(dataframe, augmentations)

    offset = np.array([tuple(augmentations[axis].augmentation.offset for axis in "xyz")],
                      dtype=HeaderGenerator.offset())
    if not np.any(offset.view("3f8")):
        return

    metadata = dataframe.attrs["metadata"]
    if "offset" not in metadata:
        logger.info("Point cloud was shifted by {} (offset)".format(offset))
        metadata["offset"] = offset
    else:
        # The offset is stored as structured array [(x, y, z)] (or a list from older metadata)
        existing = np.asarray(metadata["offset"])
        existing = existing.view("3f8").reshape(3) if existing.dtype.names else existing.astype("f8").reshape(3)
        metadata["offset"] = np.array([tuple(existing + offset.view("3f8").reshape(3))],
                                      dtype=HeaderGenerator.offset())


def data_augmentations(dataframe: DataFrame, options: IOOptions):
    """
    Shape the point cloud in place according to the options:
        intensity == "normalize"  intensity as float32
        local_system              x, y, z centred, the shift is added to attrs["metadata"]["offset"]
        shift_system              x, y, z shifted by the given offset (also added to the metadata offset)
        xyz_accuracy              dtype of x, y, z, e.g. "<f4". None keeps the current dtype (float64 stays float64,
                                  which georeferenced coordinates need).
    """
    if "metadata" not in dataframe.attrs:
        dataframe.attrs["metadata"] = {}

    xyz_dtype = np.dtype(options.xyz_accuracy) if options.xyz_accuracy else dataframe["x"].dtype

    if "intensity" in dataframe and options.intensity in ("normalize", "normalized"):
        apply_augmentations(dataframe, {"intensity": TypeAugmentation(np.float32, TrueValue())})

    if options.local_system:
        _shift_xyz(dataframe, xyz_dtype, lambda i: LocalSystem())

    if options.shift_system:
        _shift_xyz(dataframe, xyz_dtype, lambda i: GobalOffset(options.shift_system[i]))

    if dataframe["x"].dtype != xyz_dtype:
        apply_augmentations(dataframe, {axis: TypeAugmentation(xyz_dtype, TrueValue()) for axis in "xyz"})
