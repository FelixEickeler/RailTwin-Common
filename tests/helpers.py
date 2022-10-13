import numpy as np
import filecmp


def cmpfile_by_lines(path_1, path_2):
    l1 = l2 = True
    line_counter = 0
    with open(path_1, 'r') as f1, open(path_2, 'r') as f2:
        while l1 and l2:
            line_counter += 1
            l1 = f1.readline()
            l2 = f2.readline()
            if l1 != l2:
                print(f"Not the same on line {line_counter}")
                return False
    return True


def cmpfile_by_binary(path_1, path_2):
    return filecmp.cmp(path_1, path_2)


def balance_pointcloud(pcs):
    pcs["x"] = pcs["x"] - np.mean(pcs["x"])
    pcs["y"] = pcs["y"] - np.mean(pcs["y"])
    pcs["z"] = pcs["z"] - np.mean(pcs["z"])
    return pcs
