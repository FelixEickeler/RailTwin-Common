import matplotlib
# from matplotlib import pyplot as plt
import pandas
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_z_plane(data, multi=False, chunk_coloring=False, cmap=cm.cividis, sample_factor=0.01, label="chunks"):
    plt.figure()
    _data = [data] if not multi else data
    _c = np.arange(0, int(len(_data)) / 2, dtype=int)
    col_sel = np.concatenate([_c, _c])
    col_sel[1::2] += _c.shape[0]
    nr_samples = int(0.01 * len(_data[0]))
    normalize = Normalize(vmin=np.min(col_sel), vmax=np.max(col_sel))
    cmap = cm.get_cmap('viridis', len(_data[0]))

    for i, dat in enumerate(_data):
        if isinstance(dat, pandas.DataFrame):
            _dat = dat.sample(nr_samples)
            x = _dat["x"]
            y = _dat["y"]
            c = _dat["z"]

        elif isinstance(dat, np.array) and len(dat.shape[1]) > 2:
            _dat = np.random.choice(dat, nr_samples)
            x = _dat[:, 0]
            y = _dat[:, 1]
            c = _dat[:, 2]
        else:
            raise NotImplementedError("No suitable numpy array or dataframe")
        if chunk_coloring:
            c = np.ones_like(c) * i  # col_sel[i]

        plt.scatter(x=x, y=y, c=c, cmap=cmap, norm=normalize, label="Projection to z")
    ticks = np.arange(0, len(_data) + 1)
    plt.colorbar(ticks=ticks).set_label(label, rotation=270)
    plt.axis('equal')
    # plt.savefig("test.svg")
    plt.show()

