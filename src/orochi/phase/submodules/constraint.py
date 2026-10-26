import re
import numpy as np
import sympy
import copy
from ..submodules.symmetry import _identify_wyckoff_at, Database

x, y, z = sympy.symbols("x y z")

def set_symmetry_constraints(self):
    """
    Establish symmetry-based constraints for both individual atoms and rigid bodies.
    This method integrates Wyckoff position identification with parameter locking.
    """

    curr_consts = copy.deepcopy(self.par["constraints"]["value"])
    for name, val in curr_consts.items():
        if name.startswith(f"@/xyz/value/"):
            self.par["constraints"]["value"].pop(name, None)

    self()  # Ensure model is updated
    site_nums = np.arange(self["xyz"].shape[0])

    group_ids = np.unique(self["group"])
    full_symbol = self["space_group", "full"] 
    wyckoff_letters = np.asarray(self["wyckoff_letters"])
    if len(wyckoff_letters) != self["xyz"].shape[0]:
        try:
            self.set_wyckoff_letters()
            wyckoff_letters = np.asarray(self["wyckoff_letters"])
        except Exception:
            pass
    if len(wyckoff_letters) != self["xyz"].shape[0]:
        wyckoff_letters = np.full(self["xyz"].shape[0], "a", dtype="<U1")
    db = Database()    

    for group in group_ids:
        mask = self["group"] == group
        sites = site_nums[mask]
        
        if group == 0:
            # --- Case A: Individual atoms (Existing logic) ---

            wls = wyckoff_letters[mask]            
            for site_num, letter in zip(sites, wls):
                coordinates = db.get_site_constraints(hm_full=full_symbol, 
                                                      wyckoff_letter=letter)
                if coordinates is None: 
                    for coord_num in range(3):
                        name = f"@/xyz/value/{site_num}/{coord_num}"
                        self.par["constraints"]["value"].pop(name, None)
                    continue                
                
                site_const = sympy.sympify(coordinates, 
                                           locals={'x': x, 'y': y, 'z': z})[0]
                
                for coord_num, coord_const in enumerate(site_const):
                    const = _get_site_constraint(site_num, coord_num, coord_const)
                    name = f"@/xyz/value/{site_num}/{coord_num}"

                    if const is None:
                        self.par["constraints"]["value"].pop(name, None)
                    else:
                        self.par["constraints"]["value"][name] = const
            
        # Note: Rigid body symmetry constraints (group != 0) are reserved for future extension.


def _get_group_translation_constraint(group_id, coord_num, const):
    """
    Helper to convert COM template (e.g. '0') into group translation constraint.
    """
    const_str = str(const)
    # Fix to constant value if constant (e.g., 0, 0.5)
    if const_str.replace('.','',1).isdigit() or const_str == '0':
        return float(const_str)
    
    # Check variable component (x, y, z): no constraint (None) if it is the free variable itself
    vars = ['x', 'y', 'z']
    if const_str == vars[coord_num]:
        return None
        
    # Express coupled coordinates as a symbolic formula
    target_name = f"@/group_coord/value/{group_id}/"
    const_str = const_str.replace('x', target_name + '0')
    const_str = const_str.replace('y', target_name + '1')
    const_str = const_str.replace('z', target_name + '2')
    return const_str

def _get_site_constraint(site_num, coord_num, const):
    const = str(const)
    const = re.sub(r'\s*([+\-*])\s*', r' \1 ', const)
    const_parts = const.split(" ")

    if coord_num == 0 and "x" in const_parts:
        return None
    if coord_num == 1 and "y" in const_parts:
        return None
    if coord_num == 2 and "z" in const_parts:
        return None

    new_x = f"@/xyz/value/{site_num}/0"
    new_y = f"@/xyz/value/{site_num}/1"
    new_z = f"@/xyz/value/{site_num}/2"
    const_parts = [
        new_x if item == "x" else \
        new_y if item == "y" else \
        new_z if item == "z" else \
        item 
        for item in const_parts
    ]
    const = " ".join(const_parts)
    return const

def _get_site_rotation_matrices(templates):
    """
    Retrieve the 3x3 rotation matrices representing the site symmetry 
    at a given fractional coordinate.
    """
    x, y, z = sympy.symbols("x y z")
    matrices = []
    
    for template in templates:
        # Extract the linear part (R) by differentiating each component (x', y', z') with respect to x, y, z
        R = np.zeros((3, 3))
        for i, expr in enumerate(template):
            for j, var in enumerate([x, y, z]):
                coeff = sympy.diff(expr, var)
                R[i, j] = float(coeff)
        matrices.append(R)
        
    return matrices

def _find_invariant_axis(self, matrices):
    """
    Find a common invariant vector (rotation axis) in Cartesian space 
    that is consistent with all site symmetry matrices.
    """
    if not matrices:
        return np.array([0.0, 0.0, 1.0]) # Default axis
        
    # 1. Construct the system matrix A for (R - I)v = 0 across all symmetry operations R
    identity = np.eye(3)
    equations = [R - identity for R in matrices]
    A = np.vstack(equations)
    
    # 2. Perform Singular Value Decomposition (SVD); last row of Vh corresponds to smallest singular value
    _, S, Vh = np.linalg.svd(A)
    
    # 3. Check smallest singular value magnitude to determine rotational degrees of freedom
    tol = 1e-3
    if S[2] > tol:
        # No common invariant axis exists across all operations (high symmetry / inversion center)
        return None

    # Invariant fractional vector v_frac
    v_frac = Vh[2]
    
    # 4. Transform into Cartesian coordinates for rigid-body rotation axis
    lat_vec = self["lattice_vector"]
    v_cart = np.dot(v_frac, lat_vec)
    
    # Normalize and return
    norm = np.linalg.norm(v_cart)
    if norm > 1e-6:
        return v_cart / norm
    else:
        return np.array([0.0, 0.0, 1.0])

def set_lattice_constraints(self):
    crystal_system = self["crystal_system"]

    if crystal_system == "triclinic":
        pass

    elif crystal_system == "monoclinic":
        name = f"@/lattice/value/3"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/5"
        self["constraints", "value", name] = 90.

    elif crystal_system == "orthorhombic":
        name = f"@/lattice/value/3"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/4"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/5"
        self["constraints", "value", name] = 90.

    elif crystal_system == "tetragonal":
        name = f"@/lattice/value/1"
        self["constraints", "value", name] = f"@/lattice/value/0"
        name = f"@/lattice/value/3"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/4"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/5"
        self["constraints", "value", name] = 90.

    elif crystal_system == "trigonal":
        if self["space_group"][0] == "P":
            name = f"@/lattice/value/1"
            self["constraints", "value", name] = f"@/lattice/value/0"
            name = f"@/lattice/value/3"
            self["constraints", "value", name] = 90.
            name = f"@/lattice/value/4"
            self["constraints", "value", name] = 90.
            name = f"@/lattice/value/5"
            self["constraints", "value", name] = 120.
        else:
            name = f"@/lattice/value/1"
            self["constraints", "value", name] = f"@/lattice/value/0"
            name = f"@/lattice/value/2"
            self["constraints", "value", name] = f"@/lattice/value/0"
            name = f"@/lattice/value/4"
            self["constraints", "value", name] = f"@/lattice/value/3"
            name = f"@/lattice/value/5"
            self["constraints", "value", name] = f"@/lattice/value/3"

    elif crystal_system == "hexagonal":
        name = f"@/lattice/value/1"
        self["constraints", "value", name] = f"@/lattice/value/0"
        name = f"@/lattice/value/3"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/4"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/5"
        self["constraints", "value", name] = 120.

    elif crystal_system == "cubic":
        name = f"@/lattice/value/1"
        self["constraints", "value", name] = f"@/lattice/value/0"
        name = f"@/lattice/value/2"
        self["constraints", "value", name] = f"@/lattice/value/0"
        name = f"@/lattice/value/3"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/4"
        self["constraints", "value", name] = 90.
        name = f"@/lattice/value/5"
        self["constraints", "value", name] = 90.
  
def set_spherical_constraints(self):
    """

    Parameters
    ----------

    Returns
    -------
    Phase
        A Phase object representing the spherical/ellipsoidal model.
    """
    self["constraints","value", "@/sphere/value/1"] = "@/sphere/value/0"
    self["constraints","value", "@/sphere/value/2"] = "@/sphere/value/0"

def set_elemental_adp_constraints(self):
    """

    Parameters
    ----------
    """

    symbol = self["symbol"]
    unique_symbols = np.unique(symbol)

    for element in unique_symbols:
        indices = np.where(symbol == element)[0]
        if len(indices) == 1:
            continue

        for id in indices[1:]:
            self["constraints","value", f"@/adp/value/{id}"] = f"@/adp/value/{indices[0]}"
  