import numpy as np


def expansion(phase, matrix=np.ones(3)):
    xx, yy, zz = np.meshgrid(
        np.arange(matrix[0]), np.arange(matrix[1]), np.arange(matrix[2])
    )
    translational_vectors = np.concatenate(
        [xx.reshape(-1, 1).T, yy.reshape(-1, 1).T, zz.reshape(-1, 1).T]
    ).T

    base_sites = phase.lattice_sites
    phase.model = phase.make_translated_model(
        base_sites, translational_vectors
    )
    phase.model["xyz"] = phase.model["xyz"] / matrix

    lattice = phase.par["lattice"]
    lattice[:3] *= matrix
    phase.model["lattice"] = lattice
    phase.model["pbc"] = phase.par["pbc"]
    phase.model["lattice_vector"] = phase.par.get_lattice_vector(
        phase.model["lattice"]
    )

    return phase
