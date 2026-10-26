import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


def get_effective_coordination_number(
    distance, order=1, bandwidth=0.1, cluster_number=0
):
    rij = distance[distance != 0]
    rij /= np.amin(rij)
    clusters = one_dimensional_clustering_with_kernel_estimation(
        rij, bandwidth
    )
    rij = rij[clusters == cluster_number]

    r_av = np.average(rij)
    # r_min = np.amin(rij)
    # r_exp = np.exp(1.-np.power(rij/r_min, order))
    # r_av = np.sum(rij * r_exp) / np.sum(r_exp)

    bond_weight = np.exp(1.0 - np.power(rij / r_av, order))
    ECON = np.sum(bond_weight)

    CON_id = np.arange(np.size(distance))[distance.flatten() != 0][
        clusters == cluster_number
    ]
    return ECON, CON_id

def one_dimensional_clustering_with_kernel_estimation(data, bandwidth=0.10):
    minval = np.amin(data)
    maxval = np.amax(data)
    gridx = np.linspace(minval, maxval, 1024)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        data.reshape(-1, 1)
    )
    score = kde.score_samples(gridx.reshape(-1, 1))
    minpoint = argrelextrema(score, np.less)[0]

    num_minpoint = len(gridx[minpoint])
    cluster_index = np.ones_like(data) * num_minpoint

    for i in range(0, num_minpoint):
        cluster_i = np.where((data >= minval) & (data <= gridx[minpoint][i]))
        cluster_index[cluster_i] = i
        minval = gridx[minpoint][i]

    return cluster_index

