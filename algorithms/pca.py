import numpy as np
from scipy import linalg as LA
from pandas import DataFrame


def pca_cov(pointcloud: DataFrame, components=None):
    data = pointcloud[["x", "y", "z"]].to_numpy(copy=True)
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    m_cov = np.cov(data, rowvar=False)
    e, v = LA.eigh(m_cov)
    idx = np.argsort(e)[::-1][:components]
    e = e[idx]
    v = v[:, idx]
    u = np.dot(data, v)  # used to be dot(V.T, data.T).T
    return u, e, v


def svd(pointcloud: DataFrame, components=2):
    data = pointcloud[["x", "y", "z"]].to_numpy(copy=True)
    U, s, Vh = LA.svd(data, full_matrices=False)
    print("somethings")
    print("more_comp")
    # sig = np.zeros((data.shape[0], data.shape[1]))
    # sig[:data.shape[0], :data.shape[0]] = np.diag(s)
    # select
    # n_elements = 2
    # sig = sig[:, :n_elements]
    # VT = VT[:n_elements, :]
    # reconstruct
    # B = U.dot(sig.dot(VT))
    # print(B)
    # # transform
    # T = U.dot(Sigma)
    # print(T)
    # T = A.dot(VT.T)
    # print(T)
    # X_r = pca.fit_transform(pointcloud)
    # print(pca.components_)
    #
