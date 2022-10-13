# 28.06.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler
#              felix.eickeler@tum.de
# ----------------------------------------------------------------------------------------------------------------
#
#

import pandas
import numpy as np
from functools import partial

# TODO unclear stuff here
from sklearn.neighbors import KDTree


def KnnVoting(in_cloud: pandas.DataFrame, out_cloud, colname, search_tree: KDTree = None, knn=5):
    if not isinstance(search_tree, KDTree):
        search_tree = KDTree(in_cloud[["x", "y", "z"]].to_numpy())

    dist, ind = search_tree.query(out_cloud[["x", "y", "z"]].to_numpy(), k=knn)
    dist += 0.001
    probabilities = 1 / (dist / (np.min(dist, axis=1)[:, np.newaxis]))
    preds = in_cloud[colname].to_numpy()
    bins = np.zeros(np.max(preds) + 1)

    def _KnnVoting(out_row: pandas.Series):
        bins.fill(0)
        nn_preds = preds[ind[out_row.name]]
        nn_probs = probabilities[out_row.name]
        np.add.at(bins, nn_preds, nn_probs)
        return np.argmax(bins)

    return _KnnVoting

# optimization
# baseline:          1510683 := 575.000s
# baseline:           100000 :=  19.176s
# no .to_numpy       1510683 :=  27.757s
# no .to_numpy        100000 :=   0.917s
# 27.75
