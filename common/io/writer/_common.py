import copy
import json
from json import JSONEncoder

import numpy as np

from ..data_shaping import data_augmentations


class RailTwinEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


def shaped_copy(point_cloud, options):
    """
    Copy of the point cloud shaped by the options (local system, shift, xyz dtype, intensity; see data_augmentations).
    Every writer writes such a copy, so the DataFrame of the caller is not changed.
    """
    point_cloud = point_cloud.copy()
    point_cloud.attrs = {key: (copy.deepcopy(value) if isinstance(value, dict) else value)
                         for key, value in point_cloud.attrs.items()}
    data_augmentations(point_cloud, options)
    return point_cloud


def extract_metadata(point_cloud):
    meta_comment = "{}"
    if "metadata" in point_cloud.attrs:
        _meta = copy.deepcopy(point_cloud.attrs["metadata"])
        meta_comment = json.dumps(_meta, cls=RailTwinEncoder)
        del _meta

    # categories to json
    category_mapping = "{}"
    if "class" in point_cloud and "category_mapping" in point_cloud.attrs:
        category_mapping = json.dumps(point_cloud.attrs["category_mapping"])

    return category_mapping, meta_comment,
