# 05.04.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler 
#              felix.eickeler@tum.de       
# ----------------------------------------------------------------------------------------------------------------
#   This algorithms finds the minimal distance, angle, and sagittal distance of a point cloud to a polyline.
#
import json
import numpy as np
import pandas
from scipy import interpolate
from sklearn.neighbors import KDTree
from python.common.shared.common.io import read

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=DeprecationWarning)
    from numpy.core.umath_tests import inner1d


def point2alignment_properties(point_cloud: pandas.DataFrame, refined_alignment: pandas.DataFrame, inplace=False):
    hasattr(refined_alignment, 'tangential_x'), 'This is not an refined_alignment'
    # nearest-alighment-search
    tree = KDTree(refined_alignment[["x", "y", "z"]], leaf_size=2)
    dist, ind = tree.query(point_cloud[["x", "y", "z"]], k=1)
    del tree

    ind = ind[:, 0]
    selected = refined_alignment.iloc[ind]
    p2a = selected[["x", "y", "z"]] - point_cloud[["x", "y", "z"]].to_numpy()
    plummet = np.repeat(np.array([0, 0, 1])[:, np.newaxis], p2a.shape[0], axis=1).T
    cosine_angle = inner1d(p2a, plummet) / (np.linalg.norm(p2a, axis=1) * np.linalg.norm(plummet, axis=1))
    angles = np.arccos(cosine_angle)
    plane_normals = np.cross(selected[["tangential_x", "tangential_x", "tangential_x"]].to_numpy()[: None, :], plummet[None, :, :])[0]
    plane_normals = plane_normals / np.sqrt(inner1d(plane_normals, plane_normals)[:, None])
    distance_to_plane = inner1d(plane_normals, p2a)
    attack_angle = np.degrees(angles) * np.sign(distance_to_plane)

    if inplace:
        point_cloud["recording_distance"] = dist
        point_cloud["sagittal_distance"] = distance_to_plane
        point_cloud["scan_angle"] = attack_angle
        # tangentials ?
        return point_cloud

    return pandas.DataFrame({"recording_distance": dist[:, 0],
                             "sagittal_distance": distance_to_plane,
                             "scan_angle": attack_angle})


def refine_alignment(alignment: pandas.DataFrame, samples_m=100):
    # create spline out of points
    # 1. make sure they are sorted & get overall length
    alignment.sort_values(by=["horizontal_distance"], inplace=True, ignore_index=True)
    end_length = alignment.iloc[-1]["horizontal_distance"]

    # 2. define the max resolution
    num_true_pts = int(np.ceil(end_length)) * samples_m
    nr_samples = np.linspace(0, 1, num_true_pts, endpoint=True)

    # 3. Build Spline from original and create high_res points
    tck, u = interpolate.splprep([alignment["x"], alignment["y"], alignment["z"]], s=3)
    x_fine, y_fine, z_fine = interpolate.splev(nr_samples, tck)

    # 4. Recalculate distances
    original = np.array([x_fine, y_fine, z_fine])
    shift = np.concatenate([original[:, :1], original], axis=1)
    diffs = original - shift[:, :-1]
    # repeat last direction and remove first (n+1) => n
    tangents = np.concatenate([diffs[:, 1:], diffs[:, -1:]], axis=1)
    point_distances = np.linalg.norm(diffs, axis=0)
    horizontal_distance = np.cumsum(point_distances)
    refined_alignment = pandas.DataFrame({"x": x_fine, "y": y_fine, "z": z_fine,
                                          "horizontal_distance": horizontal_distance,
                                          "tangential_x": tangents[0],
                                          "tangential_y": tangents[1],
                                          "tangential_z": tangents[2], })
    return refined_alignment
