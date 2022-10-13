"""
This file contains alternative non-dynamic formats. These should only be needed if the input format may have ambiguous information about the header.
An Example for this would be the XYZ format an non defined number of columns !
"""

import numpy as np
import pandas
from ..logger.railtwinlogger import RailTwinLogger

logger = RailTwinLogger.create()


class PointCloudFormat:
    format = []

    @classmethod
    def cols(cls):
        return [i[0] for i in cls.format]

    @classmethod
    def dtype(cls):
        return {i[0]: i[1] for i in cls.format}


class Transformer():
    def valid(self, dtypes):
        raise NotImplementedError()

    def apply(self, df: pandas.DataFrame):
        raise NotImplementedError()


class HeliosFormat(PointCloudFormat, Transformer):
    """ This is a generic version of a helios XYZ format. Since Helios does not include the metadata, this can to register a xyz-reader
    """
    format = [("x", np.float64),
              ("y", np.float64),
              ("z", np.float64),
              ("intensity", np.float64),
              ("echoWidth", np.float64),
              ("returnNumber", np.int32),
              ("numberOfReturns", np.int32),
              ("fullwaveIndex", np.int64),
              ("hitObjectId", np.int64),
              ("class", np.int64),
              ("time", np.float32)]

    def apply(self, df: pandas.DataFrame):
        pass

    def valid(self, dtypes):
        return True


class ModifiedHeliosFormat(PointCloudFormat, Transformer):
    """
    This is a very specific format featuring a intermediate output of the railtwin virtualizer ai package. Normally the mapping is inputted to apply transformation
    during the tooling.
    """
    format = [("x", np.float64),
              ("y", np.float64),
              ("z", np.float64),
              ("intensity", np.float64),
              ("object_id", np.int64),
              ("gpsTime", np.float32),
              ("scanner_id", np.int16),
              ("alignment_id", np.int16)]

    def __init__(self, object_mapping):
        self.object_mapping = object_mapping

    def valid(self, dtypes):
        return True

    def apply(self, df: pandas.DataFrame):
        try:
            df["class"] = df["object_id"].apply(lambda uid: self.object_mapping[uid]["class_idx"])
            # ply only supports this as a hex encoding, which does not make much sense, i guess ? However we can apply the data
            # TODO Apply metadata
            # df["uid"] = df["object_id"].apply(lambda uid: self.object_mapping[uid]["uid"])
        except Exception as e:
            raise e
            if self.object_mapping:
                raise LookupError("Error due to the transformation, reraise1")


class ProjectPTS(PointCloudFormat, Transformer):
    """
    This is a very specific format featuring from a recording in Project:DATA is pts xyzirgb
    """
    format = [("x", np.float64),
              ("y", np.float64),
              ("z", np.float64),
              ("intensity", np.float64),
              ("red", np.uint8),
              ("green", np.uint8),
              ("blue", np.uint8)]

    def apply(self, df: pandas.DataFrame):
        pass

    def valid(self, dtypes):
        return True


class CloudCompareExport(PointCloudFormat, Transformer):
    """
    This is a very specific format featuring from a recording in Project:DATA is pts xyzirgb
    """
    format = [("x", np.float64),
              ("y", np.float64),
              ("z", np.float64),
              ("red", np.uint8),
              ("green", np.uint8),
              ("blue", np.uint8),
              ("intensity", np.float32),
              ("scalar1", np.float32),
              ("scalar2", np.float32)]

    def apply(self, df: pandas.DataFrame):
        pass

    def valid(self, dtypes):
        return True

# class PlyObject(PointCloudFormat, Transformer):
#     format = [("x", np.float64),
#               ("y", np.float64),
#               ("z", np.float64),
#               ("red", np.uint8),
#               ("green", np.uint8),
#               ("blue", np.uint8),
#               ("intensity", np.float32),
#               ("object_id", np.float32),
#               ("scalar2", np.float32)]
#
#     def valid(self, dtypes):
#         return True
#
#     def apply(self, df: pandas.DataFrame):
#         try:
#             df["class"] = df["object_id"].apply(lambda uid: self.object_mapping[uid]["class_idx"])
#             # ply only supports this as a hex encoding, which does not make much sense, i guess ? However we can apply the data
#             # TODO Apply metadata
#             # df["uid"] = df["object_id"].apply(lambda uid: self.object_mapping[uid]["uid"])
#         except Exception:
#             if self.object_mapping:
#                 raise LookupError("Error due to the transformation, reraise1")