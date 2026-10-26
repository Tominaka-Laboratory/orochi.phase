import numpy as np

# def set_lattice_vector(self):
#     vector = get_lattice_vector(self, self["lattice"])
#     self["lattice_vector"] = vector

def set_lattice_from_vector(self, lattice_vector=None):
    lattice = get_lattice_from_vector(self, lattice_vector)
    self["lattice"] = lattice

def get_lattice_vector(self, lattice):
    a = lattice[0]
    b = lattice[1]
    c = lattice[2]
    alpha = np.radians(lattice[3])
    beta = np.radians(lattice[4])
    gamma = np.radians(lattice[5])

    cos_alpha_star = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (
        np.sin(beta) * np.sin(gamma)
    )
    safe_cos = np.clip(cos_alpha_star, -1.0, 1.0)
    sin_alpha_star = np.sqrt(1.0 - safe_cos**2.0)

    g_a = [a, b * np.cos(gamma), c * np.cos(beta)]
    g_b = [
        0.0,
        b * np.sin(gamma),
        -1.0 * c * np.sin(beta) * cos_alpha_star,
    ]
    g_c = [0.0, 0.0, c * np.sin(beta) * sin_alpha_star]
    g = np.array([g_a, g_b, g_c])

    g = np.round(g, 8)
    return g.T

def get_lattice_from_vector(self, lattice_vector):
    lv = np.array(lattice_vector)
    
    # Extract basis vectors
    a_v, b_v, c_v = lv[0], lv[1], lv[2]
    
    # Axis lengths
    a = np.linalg.norm(a_v)
    b = np.linalg.norm(b_v)
    c = np.linalg.norm(c_v)
    
    # Compute cosines from inner products and norms
    cos_alpha = np.clip(np.inner(b_v, c_v) / (b * c), -1.0, 1.0)
    cos_beta  = np.clip(np.inner(c_v, a_v) / (a * c), -1.0, 1.0)
    cos_gamma = np.clip(np.inner(a_v, b_v) / (a * b), -1.0, 1.0)
    
    # Convert angles from radians to degrees
    alpha = np.rad2deg(np.arccos(cos_alpha))
    beta  = np.rad2deg(np.arccos(cos_beta))
    gamma = np.rad2deg(np.arccos(cos_gamma))
    lattice = np.array([a, b, c, alpha, beta, gamma]).flatten()

    return lattice



# def set_lattice_system(self):  
#     if self["space_group", "number"] in range(1, 3):
#         self["lattice_system"] = "triclinic"

#     elif self["space_group", "number"] in range(3, 16):
#         self["lattice_system"] = "monoclinic"

#     elif self["space_group", "number"] in range(16, 75):
#         self["lattice_system"] = "orthorhombic"

#     elif self["space_group", "number"] in range(75, 143):
#         self["lattice_system"] = "tetragonal"

#     elif self["space_group", "number"] in range(143, 168):
#         self["lattice_system"] = "trigonal"

#     elif self["space_group", "number"] in range(168, 195):
#         self["lattice_system"] = "hexagonal"

#     elif self["space_group", "number"] in range(195, 231):
#         self["lattice_system"] = "cubic"
#     else:
#         self["lattice_system"] = "triclinic"

def lattice_transformation(
    self,
    rotation_matrix=np.diag(np.ones(3)),
    translation_vector=np.zeros(3),
):
    self["xyz"] = np.dot(self["xyz"], rotation_matrix)
    self["xyz"] = self["xyz"] + translation_vector

    g = self["lattice", "vector"]
    self["lattice", "vector"] = np.dot(g, rotation_matrix)
