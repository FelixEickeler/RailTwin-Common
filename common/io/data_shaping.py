import numpy as np
from pandas import DataFrame
from .augmentations import apply_augmentations, TypeAugmentation, LocalSystem, TrueValue, GobalOffset
from .io_options import IOOptions
from .header import *
from ..logger import RailTwinLogger

logger = RailTwinLogger.create()


def data_augmentations(dataframe: DataFrame, options: IOOptions):
    if "metadata" not in dataframe.attrs:
        dataframe.attrs["metadata"] = {}

    if options.xyz_accuracy:
        _xyz_dtype = np.dtype(options.xyz_accuracy)
    else:
        _xyz_dtype = np.float32
    # Augmentations: define datatype of x,y,z,intensity -> with TA(type,augment)
    if "intensity" in dataframe and options.intensity == "normalize":
        augmentations = {
            "intensity": TypeAugmentation(np.float32, TrueValue())
        }
        apply_augmentations(dataframe, augmentations)

    if options.local_system:
        augmentations = ({
            "x": TypeAugmentation(_xyz_dtype, LocalSystem()),
            "y": TypeAugmentation(_xyz_dtype, LocalSystem()),
            "z": TypeAugmentation(_xyz_dtype, LocalSystem()),
        })
        apply_augmentations(dataframe, augmentations)

        offset = np.array([(augmentations["x"].augmentation.offset,
                            augmentations["y"].augmentation.offset,
                            augmentations["z"].augmentation.offset)],
                          dtype=HeaderGenerator.offset())
        # if offset x^2+y^2+z^2>0 then add offset value to pointcloud.attrs
        if offset["x"] ** 2 + offset["y"] ** 2 + offset["z"] ** 2 > 0:
            try:
                _existing_offset = dataframe.attrs["metadata"]["offset"].view("3f8")
                _existing_offset += offset.view("3f8")

            except KeyError:
                logger.info("Point cloud was shifted by {} (offset)".format(offset))
                dataframe.attrs["metadata"]["offset"] = offset
            except AttributeError:
                print(type(dataframe.attrs["metadata"]["offset"]))
                print(dataframe.attrs["metadata"]["offset"])
                _existing_offset = np.array(dataframe.attrs["metadata"]["offset"][0], dtype="3f8")
            except Exception as e:
                # dataframe.attscandir_supportedrs["offset"] = np.core.records.fromarrays(offset, HeaderGenerator.offset())
                # print("whyscandir_supported is this such hard thing")
                _offset = dataframe.attrs["metadata"]["offset"]
                logger.warn("Offset existed but did not comply to the standard of 3 vector. It is set to 0 !")
                raise e

    if options.shift_system:
        augmentations = ({
            "x": TypeAugmentation(_xyz_dtype, GobalOffset(options.shift_system[0])),
            "y": TypeAugmentation(_xyz_dtype, GobalOffset(options.shift_system[1])),
            "z": TypeAugmentation(_xyz_dtype, GobalOffset(options.shift_system[2])),
        })
        apply_augmentations(dataframe, augmentations)

        offset = np.array([(augmentations["x"].augmentation.offset,
                            augmentations["y"].augmentation.offset,
                            augmentations["z"].augmentation.offset)],
                          dtype=HeaderGenerator.offset())
        # if offset x^2+y^2+z^2>0 then add offset value to pointcloud.attrs
        if offset["x"] ** 2 + offset["y"] ** 2 + offset["z"] ** 2 > 0:
            try:
                _existing_offset = dataframe.attrs["metadata"]["offset"].view("3f8")
                _existing_offset += offset.view("3f8")
            except KeyError:
                logger.info("Point cloud was shifted by {} (offset)".format(offset))
                dataframe.attrs["metadata"]["offset"] = offset
            except Exception as e:
                # dataframe.attrs["offset"] = np.core.records.fromarrays(offset, HeaderGenerator.offset())
                # print("why is this such hard thing")

                _offset = dataframe.attrs["metadata"]["offset"]
                logger.warn("Offset existed but did not comply to the standard of 3 vector. It is set to 0 !")
                raise e

    elif dataframe["x"].dtype != _xyz_dtype:
        # cast to appropriate datatype
        augmentations = ({
            "x": TypeAugmentation(_xyz_dtype, TrueValue()),
            "y": TypeAugmentation(_xyz_dtype, TrueValue()),
            "z": TypeAugmentation(_xyz_dtype, TrueValue()),
        })
        apply_augmentations(dataframe, augmentations)
