from pathlib import Path

import numpy as np
import pandas
from plyfile import PlyElement, PlyData

from ..augmentations import TA
from ..data_shaping import data_augmentations
from ..writer._common import extract_metadata


def ply_writer(output_path: Path, point_cloud: pandas.DataFrame, options):
    # TODO This will augment the original dataframe => this needs to be the numpy array ? then this needs to be moved to the front i guess
    integer_intensity = True
    data_augmentations(point_cloud, options)
    data = point_cloud.to_records(index=False)
    _dtypes = []
    # checks if dtype is <i4 and sets it to <i4 if not
    for name, _type in data.dtype.descr:
        if _type in ["<i8", "i8"]:
            _dtypes.append((name, "<i4"))
        elif _type in ["<f2", "f2"]:
            if name == "intensity" and integer_intensity:
                _dtypes.append((name, "<i2"))
                point_cloud["intensity"] = TA(np.uint16).augment(point_cloud["intensity"])
            else:
                _dtypes.append((name, "f4"))
        else:
            _dtypes.append((name, _type))
    data = data.astype(_dtypes)
    del _dtypes

    output = []
    categories, meta_comment = extract_metadata(point_cloud)

    # offset and vertices instance saved into output property called offset
    if "metadata" in point_cloud.attrs and "offset" in point_cloud.attrs["metadata"]:
        # offset_name fix if needed
        if isinstance(point_cloud.attrs["metadata"]["offset"], np.ndarray):
            output.append(PlyElement.describe(point_cloud.attrs["metadata"]["offset"], 'offset'))
        else:
            print("Offset type must be ndarray")

    # vertices
    output.append(PlyElement.describe(data, 'vertices'))

    # TODO implement objects similar to faces
    if "objects" in point_cloud and "category_mapping" in point_cloud.attrs:
        pass

    comment = ["dialect=tum_ply",
               "metadata:" + meta_comment,
               "categories:" + categories]
    PlyData(output, text=not options.binary, comments=comment).write(output_path)