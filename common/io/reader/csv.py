import pandas
from pathlib import Path
from ..pcs_formats import ModifiedHeliosFormat


def create_csv_reader(format=ModifiedHeliosFormat, row_skip=0):
    def read_xyz(src: Path):
        data = pandas.read_csv(src, header=None,
                               names=format.cols(),
                               dtype=format.dtype(),
                               # converters={"time": lambda x: np.int64(float(x))},
                               delimiter=r"\s+",
                               skiprows=row_skip

                               )
        return data

    return read_xyz
