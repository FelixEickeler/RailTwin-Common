# 28.06.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler
#              felix.eickeler@tum.de
# ----------------------------------------------------------------------------------------------------------------
#
#
import unittest

import numpy as np
import pandas
from ..algorithms.voting import KnnVoting


def create_testframe():
    return pandas.DataFrame([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 2],
        [1, 0, 1, 2],
        [0, 1, 1, 2],
        [1, 1, 1, 2]
    ], columns=["x", "y", "z", "label"])


class TestVoting(unittest.TestCase):

    def test_dfknn5_ones(self):
        dfin = create_testframe()

        dfout = pandas.DataFrame([
            [0.5, 0.5, 0.5 - 0.01],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5 + 0.01]
        ], columns=["x", "y", "z"])

        # dfout = pandas.DataFrame(np.nan, index=range(0, len(dfin)), columns=['more'], dtype='float')
        voting_process = KnnVoting(dfin, dfout, "label", knn=8)
        dfout["label"] = dfout.apply(voting_process, axis=1, raw=False)

        self.assertEqual([1, 1, 2], [i for i in dfout["label"]])
