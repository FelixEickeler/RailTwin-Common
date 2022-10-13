import copy
import json
from json import JSONEncoder

import numpy as np


class RailTwinEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


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
