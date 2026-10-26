import numpy as np


# self modification
def translation(self):
    self.model["xyz"] += self["translation_vector"]

def rotation(self):
    xyz = self.model["xyz"] - self["rotation_center"]
    g = self["lattice_vector"]
    xyz = np.dot(g, xyz.T).T
    xyz = np.dot(self["rotation_matrix"], xyz.T).T
    xyz = np.dot(np.linalg.inv(g), xyz.T).T
    xyz += self["rotation_center"]
    self.model["xyz"] = xyz


# model modification
def group_modification(self):
    """
    Apply collective translations and rotations to defined atom groups in P1 model.

    This method interprets 'group_coord[0:3]' as the target fractional coordinates 
    of the Center of Mass (COM) and 'group_coord[3:6]' as rotation angles around 
    Cartesian X, Y, and Z axes. It performs a single unified transformation per group.
    """
    group_coord = self["group_coord"]
    group_axes = self["group_axes"]
    lat_vec = self["lattice_vector"]
    inv_lat = np.linalg.inv(lat_vec)
    
    for group_id, coord in enumerate(group_coord):
        # group 0 is reserved for non-rigid atoms
        if group_id == 0:
            continue

        mask = self.model["group"] == group_id
        xyz_frac = self.model["xyz"][mask]
        if len(xyz_frac) == 0: 
            continue

        # 1. Current positions and current COM in Cartesian space 
        xyz_cart = np.dot(xyz_frac, lat_vec)
        current_com_cart = np.mean(xyz_cart, axis=0)

        # 2. Target COM in Cartesian space (from group_coord[0:3])
        target_com_cart = np.dot(coord[:3], lat_vec)

        # 3. Compute combined rotation matrix (Order: X -> Y -> Z) 
        # Equivalent to sequential rotations around Cartesian axes
        rot_mat = np.eye(3)
        for i in range(3):
            angle = coord[3+i]
            axis = group_axes[group_id][i]
            if angle != 0 and np.linalg.norm(axis) > 1e-6:
                # Compute rotation matrix around arbitrary axis and accumulate product
                R_i = _get_rotation_matrix(angle, axis)
                rot_mat = np.dot(R_i, rot_mat)

        # 4. Apply transformation in one go:
        # X_new = Rotation * (X_old - COM_old) + COM_target
        # This handles both centering, rotation, and translation to target.
        xyz_rel = xyz_cart - current_com_cart
        xyz_transformed_cart = np.dot(xyz_rel, rot_mat.T) + target_com_cart

        # 5. Convert back to fractional coordinates and apply PBC
        self.model["xyz"][mask] = np.dot(xyz_transformed_cart, inv_lat) % 1.0


# structure creation using make_translated_model
def make_translated_model(self, base_sites, vectors):
    """
    Creates a translated structural model by replicating base sites along provided vectors.
    Useful for generating supercells or periodically repeated clusters.

    Args:
        base_sites (dict): Dictionary of atomic sites with keys like 'xyz', 'symbol', etc.
        vectors (NDArray): Array of (N, 3) translation vectors in fractional coordinates.

    Returns:
        dict: A new sites dictionary containing all translated atoms.
    """
    new_sites = {} 

    for key in base_sites.keys():
        if key == "xyz":
            new_sites["xyz"] = np.array(base_sites["xyz"]).reshape(-1, 3)[:,np.newaxis,:] + np.array(vectors)
            new_sites["xyz"] = new_sites["xyz"].reshape(-1, 3)
        elif key in self._expandables:
            new_sites[key] = np.tile(base_sites[key], (vectors.shape[0], 1)).T
            new_sites[key] = new_sites[key].reshape(-1)
        else:
            continue

    return new_sites


def make_supercell(self, matrix=np.array([1, 2, 1])):
    """
    Expands the current unit cell into a supercell based on the replication matrix.
    The resulting space group will be set to 'P1' to allow independent movement of sites.
    Lattice parameters are automatically scaled to reflect the new cell size.

    Args:
        matrix (NDArray): Number of repetitions along [a, b, c]. Default is [1, 2, 1].

    Returns:
        self: The Phase object updated with the supercell structure.
    """
    if self._state != "DONE": self.symmetry_operation()

    # 1. Generate translational grid
    xx, yy, zz = np.meshgrid(
        np.arange(matrix[0]), 
        np.arange(matrix[1]), 
        np.arange(matrix[2]), 
        indexing='ij'
    )
    translational_vectors = np.stack(
        [xx.flatten(), yy.flatten(), zz.flatten()], axis=-1
        )

    # 2. Expand sites using the translation helper
    expand_sites = make_translated_model(self,
                                         base_sites=self.model.copy(),
                                         vectors=translational_vectors
                                                )

    # 3. Normalize coordinates to the new supercell frame
    # Divisor should be the repetition count (matrix)
    expand_sites["xyz"] = expand_sites["xyz"] / matrix
    expand_sites["xyz"] = expand_sites["xyz"] % 1.0

    # 4. Commit changes and update global lattice [4]
    self.update(expand_sites)
    self["lattice"] = self["lattice"] * np.append(matrix, [1,1,1])
    self["space_group"] = "P1" # Reset symmetry to allow arbitrary distortions
    self["space_group","number"] = 1
    self["space_group", "setting"] = 1
    self._state = "INIT"
    return self

def make_rotated_model(
    self,
    base_sites,
    dictionary={
        "center": np.array([0.5, 0.5, 0.5]),
        "angle": 0.0,
        "rotation_vector": [1.0, 0.0, 0.0],
    },
):  # angle in degrees
    sites = {
        "xyz": np.empty((0, 3)),
        "occ": np.empty(0),
        "adp": np.empty(0),
        "label": np.empty(0),
        "symbol": np.empty(0),
        "site_index": np.empty(0),
        "site_index_p1": np.empty(0),
        "cell_index": np.empty(0),
    }

    rotation_matrix = _get_rotation_matrix(dictionary["angle"], dictionary["rotation_vector"])

    for key in sites.keys():
        if key == "xyz":
            xyz = base_sites[key] - dictionary["center"]
            xyz = np.dot(rotation_matrix, xyz.T).T
            xyz += dictionary["center"]
            sites[key] = np.append(sites[key], xyz, axis=0).reshape(-1, 3)

        else:
            sites[key] = np.append(sites[key], base_sites[key])

    for key in base_sites.keys():
        if key not in sites.keys():
            sites[key] = base_sites[key]
    return sites

def _get_rotation_matrix(angle, vector):  # angle in degrees
    angle = np.radians(angle)
    cosa = np.cos(angle)
    sina = np.sin(angle)

    rotation_vector = np.array(vector) / np.linalg.norm(vector)
    rvx, rvy, rvz = rotation_vector

    m00 = cosa + (rvx**2.0) * (1.0 - cosa)
    m01 = rvx * rvy * (1.0 - cosa) + rvz * sina
    m02 = rvx * rvz * (1.0 - cosa) + rvy * sina
    m10 = rvx * rvy * (1.0 - cosa) - rvz * sina
    m11 = cosa + (rvy**2.0) * (1 - cosa)
    m12 = rvy * rvz * (1.0 - cosa) + rvx * sina
    m20 = rvx * rvz * (1.0 - cosa) - rvy * sina
    m21 = rvy * rvz * (1.0 - cosa) - rvx * sina
    m22 = cosa + (rvz**2.0) * (1.0 - cosa)

    rotation_matrix = np.array(
        [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]
    )

    return rotation_matrix


# add_remove atom
def add_atom(self, args):

    xyz = args.get("xyz").reshape(-1, 3)
    number = xyz.shape[0]
    symbol = args.get("symbol")
    occ = args.get("occ")
    adp = args.get("adp")
    label = args.get("label")

    symmetry = args.get("symmetry").reshape(-1, 3)

    new = np.append(self["xyz"], np.array(xyz).reshape(-1, 3), axis=0)
    self["xyz"] = new

    new = np.append(self["occ"], np.array(occ))
    self["occ"] = new

    new = np.append(self["adp"], np.array(adp))
    self["adp"] = new

    new = np.append(self["symbol"], np.array(symbol))
    self["symbol"] = new

    new = np.append(self["label"], np.array(label))
    self["label"] = new

    new = np.append(self.site_symmetry_operation, symmetry, axis=0)
    self.site_symmetry_operation = new

    self.set_variable_coordinates()

def remove_atom(self, index=[-1]):

    for id in index:
        self["xyz"] = np.delete(self["xyz"], id, axis=0)
        self["occ"] = np.delete(self["occ"], id, axis=0)
        self["adp"] = np.delete(self["adp"], id, axis=0)
        self["label"] = np.delete(self["label"], id, axis=0)
        self.site_symmetry_operation = np.delete(
            self.site_symmetry_operation, id, axis=0
        )
        self["xyz","free"] = np.delete(
            self["xyz","free"], id, axis=0
        )
        self["symbol"] = np.delete(self["symbol"], id, axis=0)

