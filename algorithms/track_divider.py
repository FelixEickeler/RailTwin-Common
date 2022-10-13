from pathlib import Path
import numpy as np
import pandas
from .pca import svd
from ..common.io.data_shaping import data_augmentations
from ..common.io.io_options import IOOptions
from ..common.io import read
from ..common.io import write
from ..common.logger import RailTwinLogger
from scipy import linalg as LA

logger = RailTwinLogger.create()


def chunkify_1D(xyz_set: pandas.DataFrame, divide_length=100, v0=None, vn=None, eigenvalue_threshold=.90, tracker="#"):
    U, s, Vh = LA.svd(xyz_set, full_matrices=False)
    if v0 is None:
        v0 = Vh[0]
    # Stop if eigenvalues if < threshold (default = 95%)
    validation = np.cumsum(s) / np.sum(s)
    xyz_set.attrs["pca_validation"] = validation
    if validation[1] < eigenvalue_threshold:
        logger.info(f"An eigenvalue threshold was triggerd on part {tracker} ")
        Vh = vn

    v_diff = v0.dot(Vh[0])
    tr = np.dot(xyz_set, np.sign(v_diff) * Vh[0])
    tr_min = np.min(tr)

    length = np.max(tr) - tr_min
    if length < divide_length:
        if "metadata" not in xyz_set.attrs:
            xyz_set.attrs["metadata"] = {}
            xyz_set.attrs["metadata"].update({"pca": {
                "directional_error": 1 - v_diff,
                "projection_quality": validation,
                "pca_fit": np.sign(v_diff) * Vh,
                "tracker_id": tracker,
                "length": length}})
        return [xyz_set]
    set1_idx = tr < (length / 2 + tr_min)
    set2_idx = np.ones(len(xyz_set), dtype=bool)
    set2_idx[set1_idx] = 0
    return chunkify_1D(xyz_set[set1_idx], divide_length, v0=v0, vn=Vh, tracker=tracker + "0") + \
           chunkify_1D(xyz_set[set2_idx], divide_length, v0=v0, vn=Vh, tracker=tracker + "1")


def _track_divider(data, length=100):
    # find pca score of the point cloud
    xyz_extract = data[["x", "y", "z"]].copy()
    data_augmentations(dataframe=xyz_extract, options=IOOptions.do_nothing())
    xyz_extract.drop("z",axis=1, inplace=True)

    chunks = chunkify_1D(xyz_extract, length)
    # chunks are the projected points, now extract based on chunks
    divided = []
    for chunk in chunks:
        divided.append(data.iloc[chunk.index])
        if "metadata" not in divided[-1].attrs:
            divided[-1].attrs["metadata"] = {}
        divided[-1].attrs["metadata"].update(chunk.attrs["metadata"])
        # divided[-1].attrs["pca_fit"] = chunk.attrs["pca_fit"]  # + data.attrs["offset"]
    return divided


def track_divider(src, dst, length, out_options: IOOptions, in_options: IOOptions, label="chunk", transformer=None):
    '''
    This function is very memory hungry. If you hit memory limits try to modify the recursive function to dump tmp files for evey split in _track_divider
    :param src:
    :param dst:
    :param length:
    :param out_options:
    :param label:
    :param transformer:
    :return:
    '''
    dst = Path(dst).expanduser()
    if isinstance(src, Path) or isinstance(src, str):
        src = Path(src).expanduser()
        data = read(src)
    elif isinstance(src, pandas.DataFrame):
        data = src
    else:
        raise

    if in_options.intensity == "normalize" and hasattr(data, 'intensity'):
        # debug = data.intensity
        data.intensity = np.interp(data.intensity, (data.intensity.min(), data.intensity.max()), (-1364, 2034)).astype(np.int16)  # normalize to 16 bit e.g. laz specification + rail-vmx

    if not dst.is_file():
        dst.mkdir(exist_ok=True, parents=True)
        base_name = src.stem
        if not out_options.filetype:
            out_options = src.suffix
    else:
        base_name = dst.stem
        out_options.filetype = dst.suffix
        dst = dst.parent

    # file_type = ".ply"
    logger.info(f"Dividing:  {src.name}")
    divided_pcs = _track_divider(data, length=length)
    pad_len = len(str(len(divided_pcs)))
    for i, div in enumerate(divided_pcs):
        out_path = dst / f"{base_name}_{label}{i:03}{out_options.filetype}"
        if transformer and transformer.valid(div.dtypes):
            transformer.apply(div)
        logger.info(f"Writing part {i}: \t {out_path}")
        write(out_path, div, options=out_options)
