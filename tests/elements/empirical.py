import numpy as np

# element types
element_types = {
    "transition_metal": np.arange(21, 30).tolist()
    + np.arange(39, 48).tolist()
    + np.arange(72, 80).tolist(),
    "lanthanoid": np.arange(57, 72).tolist(),  # La-Au
    "alkali": [3, 11, 19, 37, 55, 87],
    "alkali_earth": [4, 12, 20, 38, 56, 88],
    "halogen": [9, 17, 35, 53, 85, 117],
    "chalcogen": [8, 16, 34, 52, 84, 116],
    "rare_gas": [2, 10, 18, 36, 54, 86, 118],
    "metalloid": [5, 14, 32, 33, 51, 52],
}

typical_valence_coordination = {
    "Li": {"1": [4, 6]},
    "Be": {"2": [4, 3]},
    "B": {"3": [3, 4]},
    "C": {"4": [4]},
    "O": {"-2": [2, 4, 3], "0": [1, 2]},
    "Na": {"1": [6]},
    "Mg": {"2": [6]},
    "K": {"1": [6, 8]},
    "Ca": {"2": [8]},
    "Ce": {"4": [8]},
    "Ni":{"0":[12], "2":[6]}
}


center_distance_in_polyhedra_connection = {
    "6": {  # octahedron
        "3": 1.25,  # face sharing
        "2": 1.41,  # edge sharing
        "1": 2.0,  # corner sharing
        "0": 1.41,  # edge
    },
    "8": {"3": 1.15, "2": 1.63, "1": 2.0, "0": 1.15},  # hexahedron
    "4": {"2": 1.15, "1": 2.0, "0": 1.63},  # tetrahedron
}
