import numpy as np


def pattern_stacking(phase, stack_sequence, inplane_vector, stacking_axis=2):
    number_of_stacks = np.size(stack_sequence)
    stacking_vectors = np.zeros(number_of_stacks * 2).reshape(-1, 2)

    for key in inplane_vector.keys():
        layer = np.where(stack_sequence == key)
        stacking_vectors[layer] = inplane_vector[key]

    stacking_vectors = stacking_vectors.reshape(-1, 3)
    stacking_vectors = np.insert(
        stacking_vectors, stacking_axis, np.arange(number_of_stacks), axis=-1
    )
    phase = phase.cell_stacking(stacking_vectors, stacking_axis=stacking_axis)

    return phase


def cell_stacking(phase, vectors, stacking_axis=2):
    number_of_stacks = np.shape(vectors)[0]
    translational_vectors = vectors

    base_sites = phase.lattice_sites
    phase.model = phase.make_translated_model(
        base_sites, translational_vectors
    )

    xyz = phase.model["xyz"]
    xyz[:, stacking_axis] /= number_of_stacks
    phase.model["xyz"] = xyz

    lattice = phase.model["lattice"]
    lattice[stacking_axis] *= number_of_stacks
    phase.model["lattice"] = lattice
    phase.model["pbc"] = phase.par["pbc"]
    phase.model["lattice_vector"] = phase.par.get_lattice_vector(
        phase.model["lattice"]
    )

    return phase
