import unittest
# from algorithms.track_divider import pca_cov
from common.io.reader.laz import read_laz
from pathlib import Path
from timeit import default_timer as timer

# config
# las_path = Path(r"data/test - Cloud.las")


class TestPca(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass
        # self.repeat = 10
        # self.data = read_laz(las_path)

    def correct_split(self):
        self.assertTrue(False)
        # start = timer()
        # for i in range(self.repeat):
        #     u, e, v = pca_cov(self.data)
        # time = timer() - start
        # print('Cpv time: %.3fms' % (time * 1000 / self.repeat))

