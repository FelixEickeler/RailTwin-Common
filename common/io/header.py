import numpy as np
import pandas
from .augmentations import TA, TrueValue, Normalize

_allowed_headers = {"x": TA(np.float64, TrueValue()),
                    "y": TA(np.float64, TrueValue()),
                    "z": TA(np.float64, TrueValue()),
                    "red": TA(np.uint8, TrueValue()),
                    "green": TA(np.uint8, TrueValue()),
                    "blue": TA(np.uint8, TrueValue()),
                    "intensity": TA(np.float32, Normalize()),
                    "echoWidth": TA(np.float64, Normalize()),
                    "returnNumber": TA(np.uint8, TrueValue()),
                    "numberOfReturns": TA(np.uint8, TrueValue()),
                    # "fullwaveIndex": TA(np.int, TrueValue()), #higher laz or helios, not implemented right now
                    # "hitObjectId": TA(np.int32, TrueValue()),
                    "class": TA(np.uint32, TrueValue()),
                    "gps_time": TA(np.float64, TrueValue()),
                    # => las
                    "scan_direction_flag": TA(np.bool, TrueValue()),
                    "edge_of_flight_line": TA(np.bool, TrueValue()),
                    "synthetic": TA(np.bool, TrueValue()),
                    "key_point": TA(np.bool, TrueValue()),
                    "withheld": TA(np.bool, TrueValue()),
                    "scan_angle_rank": TA(np.int8, TrueValue()),
                    "user_data": TA(np.uint8, TrueValue()),
                    "point_source_id": TA(np.uint8, TrueValue()),
                    "object_id": TA(np.uint32, TrueValue()),
                    "scanner_id": TA(np.uint8, TrueValue()),
                    "above_ground": TA(np.float32, TrueValue()),
                    "DTMAnalysis": TA(np.uint8, TrueValue()),
                    "component": TA(np.uint32, TrueValue()),
                    "system": TA(np.uint32, TrueValue()),
                    "preds_system": TA(np.uint32, TrueValue()),
                    "preds_component": TA(np.uint32, TrueValue()),
                    }

# these alternative mappings are from testfile
_header_mapping = {
    "return_number": "returnNumber",
    "number_of_returns": "numberOfReturns",
    "classification": "class",
    "time": "gps_time",
    "gpstime": "gps_time",
    "gps_time": "gps_time",
    "scalar_intensity": "intensity",
    "scalar_object_id": "object_id",
    "scalar_gpsTime" : "gps_time",
    "scalar_scanner_id" : "scanner_id",
    "scalar_classification": "class",
    "scalar_component" : "component",
    "scalar_system" : "system"
}

for key in _allowed_headers:
    _header_mapping[key.lower()] = key


class HeaderGenerator:

    def __init__(self):
        self.header = {}
        self.mapping = {}

    @staticmethod
    def determine_supertype(nptype):
        if nptype == (np.float64 or np.float32 or np.float16): return "float"
        if nptype == (np.int16 or np.int32 or np.int64):       return "int"
        if nptype == (np.uint16 or np.uint32 or np.uint64):    return "uint"
        return "nan"

    def numpy_dtypes(self):
        _dtypes = []
        for i, ta in self.header.items():
            if hasattr(ta.type, 'name') and ta.type.name == "category":
                _dtypes.append((i, "i4"))
            else:
                _dtypes.append((i, ta.type))
        return _dtypes

    def dtypes(self):
        return [(i, t.type) for i, t in self.header.items()]

    def create_dataframe(self, data):
        series = {}
        if data.dtype.names:
            for ext_name in data.dtype.names:
                if ext_name in self.mapping:
                    internal_name = self.mapping[ext_name]
                    internal_type = self.header[internal_name].type
                    series[internal_name] = pandas.Series(data[ext_name], dtype=internal_type)
        return pandas.DataFrame(series)

    def add(self, _type):
        if _type.lower() in _header_mapping:
            true_name = _header_mapping[_type.lower()]
            self.mapping[_type] = true_name
            self.header[true_name] = _allowed_headers[true_name]
        else:
            raise "This property is not in the set, if you need it add it to mappings or allowed headers !"
        return self

    @staticmethod
    def possible_properties():
        return list(_header_mapping.keys())

    @staticmethod
    def create_xyzic():
        hg = HeaderGenerator()
        return hg.add("x").add("y").add("z").add("intensity").add("class")

    @staticmethod
    def create_xyz():
        hg = HeaderGenerator()
        return hg.add("x").add("y").add("z")

    @staticmethod
    def offset():
        return [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
