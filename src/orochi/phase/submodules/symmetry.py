import numpy as np
from scipy.spatial import distance
import re, copy
import itertools
from ..database.symmetries import Database


def symmetry_operation(self):
    """
    symmetry operation including space group and point group.
    """

    if (self["hall_number"] not in [None, 1]) and (np.size(self["xyz"]) != 0):
        hall_number = self["hall_number"]
        if hall_number is None:
            raise ValueError(f"Hall number not found for SG: {self['space_group']}")
        self.model = apply_symmetry_operations(self, hall_number=hall_number, mode="p1")
    
    elif (self["point_group"] not in [None, "1"]) and (np.size(self["xyz"]) != 0):
        self.model = apply_symmetry_operations(self, point_group=self["point_group"], mode="p1")
    
    else:
        non_symmetry_structure(self)

    self.model["lattice"] = self["lattice"]
    self.model["lattice_vector"] = self["lattice_vector"]
    self.model["pbc"] = self["pbc"]

    if "site_index" not in self.model:
        self.model["site_index"] = np.arange(np.shape(self.model["xyz"])[0])


def apply_symmetry_operations(self, 
                        base_sites=None, 
                        lattice_vector=None, 
                        hall_number=None, 
                        point_group=None, 
                        mode="au", 
                        symprec=0.01):
    if lattice_vector is None:
        lattice_vector = self["lattice_vector"]
    if base_sites is None:
        base_sites = {}
        for key in self._expandables:
            if np.shape(self[key])[0] == np.shape(self["xyz"])[0]:
                base_sites[key] = self[key]

    db = Database()
    if hall_number is not None:
        syms = db.get_symmetry_operations(hall_number=hall_number)
        rotations = syms["rotation"]
        translations = syms["translation"]
    else:
        rotations = Database().get_point_group_rotations(hm_symbol=point_group)
        rotations = np.array(rotations, dtype=float)
        translations = None

    res = _find_overlap(self, 
                           xyz=base_sites["xyz"], 
                           symbol=base_sites["symbol"], 
                           lattice_vector=lattice_vector, 
                           rotation_matrix=rotations, 
                           translation_matrix=translations,
                           symprec=symprec)  
    
    p1_indices = res["p1_indices"]
    p1_xyz = res["p1_xyz"]    
    n_pts = base_sites["xyz"].shape[0]

    if mode == "au":
        au_indices = res["au_indices"]
        _, is_au_site = np.unique(au_indices, return_index=True)
        target_mask = is_au_site
    else:
        target_mask = np.ones(len(p1_indices), dtype=bool)
    
    sites = {"xyz": p1_xyz[target_mask]}   
    m_pts = syms['rotation'].shape[0]
    
    for key, value in base_sites.items():
        if key == "xyz": continue

        if np.shape(value)[0] == n_pts:
            all_attr = np.repeat(value, m_pts)
            sites[key] = all_attr[p1_indices[target_mask]]
    sites["site_index"] = res["au_indices"][target_mask]

    return sites

def _find_overlap(self, 
                  xyz, 
                  symbol, 
                  lattice_vector, 
                  rotation_matrix=None, 
                  translation_matrix=None,
                  symprec=None): # Cartesian overlap distance threshold (Å), typically 0.01-0.05

    if symprec is None:
        symprec = self["symprec"]

    # 1. Apply symmetry operations (N, M, 3)
    if rotation_matrix is not None:
        # Compute Rx (column vector): n=atom, m=operation, i=coordinate component
        xyz_all = np.einsum('mij, nj -> nmi', rotation_matrix, xyz)
    else:
        xyz_all = xyz.copy().reshape(-1, 1, 3)

    if translation_matrix is not None:
        xyz_all = xyz_all + translation_matrix

    n_pts, m_pts, _ = xyz_all.shape
    xyz_all = np.mod(xyz_all, 1.0) # Wrap fractional coordinates into [0, 1)
    
    # 2. Convert to Cartesian coordinates for distance evaluation
    xyz_cart = np.dot(xyz_all, lattice_vector.T)
    
    # 3. Periodic boundary condition shift vectors (Cartesian coordinates)
    shifts = [[-1, 0, 1] if pbc else [0] for pbc in self["pbc"]]
    trans_coords = np.array([list(x) for x in itertools.product(*shifts)])
    trans_vecs = np.dot(trans_coords, lattice_vector.T)
    
    # 4. Build adjacency list of overlapping symmetry-generated sites
    adj = {i: set() for i in range(n_pts * m_pts)}
    symbol_all = np.repeat(symbol, m_pts)
    xyz_cart_flat = xyz_cart.reshape(-1, 3)

    for i in range(n_pts):
        for j in range(m_pts):
            src = i * m_pts + j
            for sv in trans_vecs:
                dists = distance.cdist([xyz_cart[i][j] + sv], xyz_cart_flat).flatten()
                # Identify sites within symprec that share the same chemical symbol
                overlap_indices = np.where((dists < symprec) & (symbol[i] == symbol_all))[0]

                for dst in overlap_indices:
                    if src != dst:
                        adj[src].add(int(dst))
                        adj[dst].add(int(src))

    # 5. Group equivalent sites via breadth-first search (BFS)
    groups = []
    total_pts = n_pts * m_pts
    used = np.zeros(total_pts, dtype=bool)
    
    for i in range(total_pts):
        if used[i]: continue
        component = []
        stack = [i]
        used[i] = True
        while stack:
            u = stack.pop()
            component.append(int(u))
            for v in adj[u]:
                if not used[v]:
                    used[v] = True
                    stack.append(int(v))
        groups.append(sorted(component))

    # 6. Stable circular averaging in fractional coordinates (complex phase mean for PBC safety)
    xyz_frac_flat = xyz_all.reshape(-1, 3)
    p1_xyz = []
    
    for group in groups:
        coords_frac = xyz_frac_flat[group]
        angles = coords_frac * 2.0 * np.pi
        mean_complex = np.mean(np.exp(1j * angles), axis=0)
        mean_angles = np.angle(mean_complex)
        mean_frac = (mean_angles / (2.0 * np.pi)) % 1.0
        p1_xyz.append(mean_frac)
        
    p1_xyz = np.array(p1_xyz)

    # 7. Restore representative indices
    p1_indices = np.array([min(group) for group in groups])
    au_indices = np.array([min(np.array(group) // m_pts) for group in groups])

    return {"p1_xyz": p1_xyz, "p1_indices": p1_indices, "au_indices": au_indices}

def non_symmetry_structure(self):
    self.model = {}
    for key in self._expandables:
        if key == "xyz":
            self.model[key] = self[key]
        else:
            if np.shape(self[key])[0] == np.shape(self["xyz"])[0]:       
                self.model[key] = self[key]



# Basic transformation

def transform_into_standard_cell(self):
    import spglib

    lattice = self["lattice_vector"]
    positions = np.copy(self.model["xyz"])
    site_index = np.copy(self.model["site_index"])

    cell = (lattice, positions, site_index)
    dataset = spglib.get_symmetry_dataset(
        cell, symprec=self["symprec"], hall_number=0
    )

    if dataset == None:
        self["multiplicity"] = np.ones_like(self["occ"])
        self.model["multiplicity"] = np.ones_like(
            self.model["occ"]
        )

    else:
        self["hall_number"] = dataset.hall_number

        # self.set_lattice_system()
        self["lattice_vector"] = dataset.std_lattice

        site_index = dataset.std_types
        equiv_atoms, multiplicity = np.unique(
            dataset.equiv_atoms, return_counts=True
        )

        use_atoms = []
        xyz = np.array(dataset.std_positions).reshape(-1, 3)
        id = np.arange(xyz.shape[0])
        for qa in equiv_atoms:
            use = np.where(qa == dataset.equiv_atoms)
            length = np.linalg.norm(xyz[use], axis=-1)
            use_atoms += [id[use][np.argmin(length)]]
        equiv_atoms = np.array(use_atoms)

        for key in self._expandables:
            if key == "multiplicity":
                self[key] = multiplicity

            elif key == "wyckoff_letters":
                self[key] = np.array(dataset.wyckoffs)[
                    equiv_atoms
                ]

            elif key == "site_symmetry_symbols":
                self[key] = np.array(
                    dataset.site_symmetry_symbols
                )[equiv_atoms]

            elif key == "xyz":            
                self[key] = xyz[equiv_atoms]

            else:
                value = np.array(self[key]).reshape(-1)
                if np.size(value) != np.shape(xyz)[0]:
                    continue
                self[key] = value[site_index][equiv_atoms]

def make_p1_structure(self):
    symmetry_operation(self)
    for key in self.model:
        if key in self.par:
            self[key] = self.model[key]

    self["hall_number"] = 1
    self["multiplicity"] = np.ones_like(self["occ"])
    self["wyckoff_letters"] = np.full(np.size(self["occ"]), "a")


# space group and hall number
def set_space_group(self, 
                    value=None,
                    hm_short:str=None, 
                    hm_full:str=None,
                    table_no:int=None, 
                    setting=1, 
                    hall_number=None):
    db = Database()

    if value is not None:
        if isinstance(value, str):
            hall_number = db.get_hall_number(hm_short=value)
            hm_short = value

            if hall_number is None:
                hall_number = db.get_hall_number(hm_full=value)
        else:
            hall_number = value
    elif hall_number is not None:
        pass

    else:
        hall_number = db.get_hall_number(hm_short=hm_short, 
                                         hm_full=hm_full, 
                                         table_no=table_no, 
                                         setting=setting)
    if hm_short is None:
        hm_short = db.get_hm_info(hall_number=hall_number)
        if hm_short is None:
            hm_short = "P1"
        else:
            hm_short = hm_short.iloc[0]["hm_short"]

    self.par["hall_number"]["value"] = hall_number
    self.par["space_group"]["value"] = hm_short

def get_space_group(self, key="hm_short"):
    db = Database()

    info = db.get_hm_info(hall_number=self["hall_number"])
    if info is None:
        return None
    else:
        val = info.to_dict(orient="records")[0][key]
        return val


    
# Site symmetry
def get_site_symmetries(self, xyz_array=None):
    """
    Retrieve Wyckoff information and site symmetry operations for given coordinates.

    This method performs pattern matching against the Wyckoff database using 
    the current space group. It identifies Wyckoff letters, site symmetry symbols, 
    and generates the corresponding mathematical symmetry operations.

    Parameters
    ----------
    xyz_array : numpy.ndarray, optional
        Array of fractional coordinates to analyze. 
        Defaults to self["xyz"] (the current asymmetric unit).

    Returns
    -------
    tuple of lists
        (site_symmetry_operations, wyckoff_letters, site_symmetry_symbols, multiplicities)
        Each list contains metadata corresponding to the input coordinates.
        Entries are None if no match is found within tolerance.
    """
    if xyz_array is None:
        xyz_array = self["xyz"]

    output = {"site_symmetry_operations": [], 
              "wyckoff_letters": [], 
              "site_symmetry_symbols": [], 
              "multiplicity": []}

    # Ensure the full Hermann-Mauguin symbol is set for database lookup
    db = Database()
    hm_info = db.get_hm_info(hall_number=self["hall_number"])
    if hm_info is None or hm_info.empty:
        return output

    hm_full = hm_info.iloc[0]["hm_full"]
    setting_no = hm_info.iloc[0]["setting_no"]
    
    for xyz in xyz_array:
        # Pattern match coordinate against operations in the dictionary [1]
        res = _identify_wyckoff_at(xyz, hm_full, setting=setting_no)
        
        if res is not None:
            output["site_symmetry_operations"].append(res["operation"])
            output["wyckoff_letters"].append(res["letter"])
            output["site_symmetry_symbols"].append(res["site_symmetry"])
            output["multiplicity"].append(res["multiplicity"])

        else:
            output["site_symmetry_operations"].append(None)
            output["wyckoff_letters"].append(None)
            output["site_symmetry_symbols"].append(None)
            output["multiplicity"].append(None)
    return output
 
def set_site_symmetries(self):
    """
    Update the Phase object with identified site symmetry metadata.

    Executes get_site_symmetries and assigns the results to the internal
    storage keys. This metadata is essential for defining structural 
    constraints during refinement.
    """
    res = get_site_symmetries(self)
    for key, value in res.items():
        self[key] = value

def set_wyckoff_letters(self):
    """
    Update the Phase object with identified Wyckoff letters.

    """
    res = get_site_symmetries(self)
    self["wyckoff_letters"] = res["wyckoff_letters"]
    self["multiplicity"] = res["multiplicity"]

def set_wyckoff_candidates(self):
    """
    Update the Phase objact with identified Wyckoff candidates.

    """
    db = Database()
    wyckoff_letters = copy.deepcopy(np.array(self["wyckoff_letters"]).tolist())
    hall_number = self["hall_number"]

    candidates_dict = {}
    for wl in set(wyckoff_letters):
        if wl is None:
            candidates_dict[str(wl)] = []  # Default empty list when Wyckoff letter is None
            continue
        
        candidates = db.get_equiv_multiplicity_wyckoff_letters(
            hall_number=hall_number, 
            wyckoff_letter=wl
        )
        if candidates is None:
            candidates_dict[str(wl)] = []  # Default empty list when no candidates are found
            continue
        candidates_dict[wl] = candidates.values.tolist()
    self.par["wyckoff_letters"]["candidates"] = [candidates_dict[str(wl)] for wl in wyckoff_letters]


def _identify_wyckoff_at(xyz_frac, hm_full, setting=1, symprec=1e-3):
    """
    Identify Wyckoff position and site symmetry symbol from database operations.
    
    Parameters
    ----------
    xyz_frac : array_like (3,)
        Fractional coordinates to identify.
    hm_full : str
        Full space group symbol (e.g., "P12/m1", "Fm-3m").
    symprec : float
        Tolerance for coordinate matching.
        
    Returns
    -------
    dict
        Dictionary containing 'multiplicity', 'letter', and 'site_symmetry'.
    """
    import sympy
    x, y, z = sympy.symbols("x y z")

    # 1. Normalize coordinates into [0.0, 1.0) under periodic boundary conditions
    xyz_frac = np.array(xyz_frac, dtype=float) % 1.0
    # Round values very close to 1.0 (e.g., 0.999999) to 0.0
    xyz_frac = np.where(np.isclose(xyz_frac, 1.0, atol=symprec), 0.0, xyz_frac)

    sg = hm_full.replace(" ", "")
    db = Database()
    wyckoff_list = db.get_wyckoff_positions(sg, setting)
    if wyckoff_list is None:
        return None
        
    # 2. Scan from special positions to general position
    for site_entry in reversed(wyckoff_list):
        operations = site_entry["coordinates"]

        for operation in eval(operations):
            if _match_operation(xyz_frac, operation, symprec):
                return {
                    "multiplicity": site_entry["multiplicity"],
                    "letter": site_entry["wyckoff_letter"],
                    "site_symmetry": site_entry["point_symmetry"],
                    "operation": operation
                }
                
    # 3. Fallback to general position if no special position matched
    general_site = wyckoff_list[0]
    return {"multiplicity": general_site["multiplicity"], 
            "letter": general_site["wyckoff_letter"], 
            "site_symmetry": general_site["point_symmetry"],
            "operation": general_site["coordinates"]}

def _match_operation(xyz, operation, prec):
    """
    Check if the coordinate matches the operation (e.g., (x, x, 0.5)).
    """
    import sympy
    x_sym, y_sym, z_sym = sympy.symbols("x y z")
    vars_found = {}
    
    for i in range(3):
        expr = operation[i]
        val = xyz[i]
        
        # 1. Constant expression (float/int)
        if isinstance(expr, (float, int, sympy.Number)):
            if abs(float(expr) - val) > prec:
                return False
        
        # 2. Expression containing free variables (x, x+1/2, etc.)
        else:
            potential_vars = expr.atoms(sympy.Symbol)
            if not potential_vars:
                if abs(float(expr.evalf()) - val) > prec: return False
                continue
                
            var = next(iter(potential_vars)) 
            solved_val = sympy.solve(sympy.Eq(expr, val), var)[0]
            
            if solved_val is not None:
                v_val = float(solved_val) % 1.0
                
                if var in vars_found:
                    if abs(vars_found[var] - v_val) > prec:
                        return False
                else:
                    vars_found[var] = v_val
            else:
                return False
            
                
    return True


def check_alternative_space_group(self, symprec=0.0001):
    import spglib
    phase = copy.deepcopy(self)

    # information of current system for using spglib
    lattice = phase["lattice_vector"]
    positions = np.copy(phase.model["xyz"])
    site_indexs = np.copy(phase.model["site_index"])

    # find alternative using spglib
    cell = (lattice, positions, site_indexs)
    sg = spglib.get_spacegroup(cell, symprec=symprec)

    if sg == None:
        return False

    # comparison
    current = phase["space_group", "number"]
    new = int(sg[sg.find("(") + 1 : sg.find(")")])

    if current == new:
        return False
    else:
        return True

def make_higher_symmetry_structure(self, symprec: float = 0.001):
    """
    Detects higher symmetry and reduces the model to its asymmetric unit.
    """
    import spglib
    if self._state != "DONE": symmetry_operation(self)

    # 1. Prepare input data
    lattice = self["lattice_vector"]
    positions = np.copy(self.model["xyz"])
    site_indexs = np.array(self.get_atomic_numbers(self.model["symbol"]),dtype=int)
    cell = (lattice, positions, site_indexs)
    current_num = self["space_group", "number"]

    # 2b. Aggressive symmetry detection with potential idealization
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None: 
        self._logger.info("Symmetry conversion failed. No higher symmetry applied.")
        return False

    new_name = dataset.international
    new_num = int(dataset.number)
    new_hall = dataset.hall_number

    if current_num >= new_num:
        self._logger.info(f"Found hall number {new_hall}. No higher symmetry applied.")
        return False

    self._logger.info(f"Higher symmetry found. Conversion from {self['space_group']} to {new_name} .")

    matrix = dataset.transformation_matrix
    shift = dataset.origin_shift
    equiv_atoms = dataset.equivalent_atoms
    unique_groups, group_indices = np.unique(equiv_atoms, return_index=True)

    new_sites = {}

    for key in self._expandables:
        new_sites[key] = []

    average_keys = ["occ", "adp", "charge"]
    for g in unique_groups:
        members = np.where(equiv_atoms == g)
        for key in average_keys:
            new_sites[key].append(np.mean(self.model[key][members], axis=0))

    xyz = self.model["xyz"][group_indices]
    new_sites["xyz"] = (np.dot(matrix, xyz.T).T + shift) % 1.0

    for key in self._expandables:
        if key in ["xyz"] + average_keys or key not in self.model: continue

        new_sites[key] = self.model[key][group_indices]
    
    lattice_vector = np.dot(matrix.T, self["lattice_vector"])

    assym_unit = apply_symmetry_operations(self, 
                                            hall_number=new_hall, 
                                            base_sites=new_sites, 
                                            lattice_vector=lattice_vector, 
                                            mode="au",
                                            symprec=symprec)
    
    # 6. Apply updates (triggers internal symmetry/constraint sync) 
    self.update(assym_unit)

    self["hall_number"] = new_hall
    self._state = "INIT"

    set_wyckoff_letters(self)

    msg = (f"Symmetry elevated to SG {self['space_group']} (Hall {new_hall}). "
               f"Model reduced to {np.shape(self['xyz'])[0]} sites while idealizing geometry.")
    self._logger.info(msg)
    return msg


##Subgroups
def estimate_cell_expansion(trans_str: str) -> int:
    """
    Estimates the unit cell volume expansion factor from a transformation matrix string (e.g., "2a,2b,c", "a,b,2c;0,0,1/2").
    """
    matrix_part = trans_str.split(";")[0].strip()
    a_mult = 2 if "2a" in matrix_part else 1
    b_mult = 2 if "2b" in matrix_part else 1
    c_mult = 2 if "2c" in matrix_part else 1
    return a_mult * b_mult * c_mult

def suggest_subgroup_hall_number(self, max_gen=3, trials_per_candidate=2, max_atoms=20):
    parent_hall = self["hall_number"]

    # 1. Precompute reachability map and candidate counts up to max_gen
    gen_maps = build_multi_gen_reachability_map(self, parent_hall, max_gen=max_gen, max_atoms=max_atoms)
    n_candidates_per_gen = {g: len(gen_maps[g]) for g in range(2, max_gen + 1)}

    thresholds = {}
    cumulative_trials = 0
    for g in range(2, max_gen + 1):
        cumulative_trials += n_candidates_per_gen[g] * trials_per_candidate
        thresholds[g] = cumulative_trials

    print(f"[Init] Subgroup candidates per generation: {n_candidates_per_gen}")
    print(f"[Init] Generation unlock thresholds (trials): {thresholds}")

    # 2. Per-trial execution function for Optuna
    def run(trial):
        trial_no = trial.number
        all_depths = list(range(2, max_gen + 1))

        if trial_no < thresholds[2]:
            target_depth = 2
        else:
            target_depth = trial.suggest_categorical("target_depth", all_depths)

        target_map = gen_maps[target_depth]
        if not target_map:
            return parent_hall

        gen_choices = list(target_map.keys())
        selected_target_hall = trial.suggest_categorical(f"target_hall_gen{target_depth}", gen_choices)

        possible_paths = target_map[selected_target_hall]
        if len(possible_paths) > 1:
            path_labels = [p["label"] for p in possible_paths]
            selected_path_label = trial.suggest_categorical(f"path_for_{selected_target_hall}", path_labels)
            path_idx = path_labels.index(selected_path_label)
        else:
            path_idx = 0

        chosen_path = possible_paths[path_idx]
        apply_path_transformations(self, chosen_path)

        return selected_target_hall

    return run

def clean_trans_str(trans_str):
    """Formats a transformation matrix string into a safe Optuna parameter/label identifier."""
    return str(trans_str).replace(",", "").replace(";", "_").replace("/", "").replace(" ", "")

def build_multi_gen_reachability_map(self, parent_hall, max_gen=3, max_atoms=None):
    """
    Precomputes all reachable subgroup Hall numbers and transformation paths up to `max_gen` generations from `parent_hall`.
    """
    base_num_atoms = np.shape(self.model["xyz"])[0]
    reachability_map = {g: {} for g in range(2, max_gen + 1)}

    # Queue: (curr_hall, curr_atoms, curr_gen, visited_halls_set, path_history)
    queue = [(int(parent_hall), base_num_atoms, 1, {str(parent_hall)}, [])]

    print(f"[Map Build] Starting subgroup search: Parent Hall={parent_hall}, Max Gen={max_gen}")

    while queue:
        curr_hall, curr_atoms, curr_gen, visited, history = queue.pop(0)

        if curr_gen >= max_gen:
            continue

        next_gen = curr_gen + 1

        df = get_subgroup_candidates(self, curr_hall)
        if df is None or df.empty:
            continue

        for _, row in df.iterrows():
            sub_hall = str(row["hall_number"])
            trans = str(row["transformation"])

            if sub_hall in visited:
                continue

            if ";" in trans:
                matrix_part = trans.split(";")[0].strip()
            else:
                matrix_part = trans.strip()

            try:
                expansion_factor = abs(estimate_cell_expansion(matrix_part))
                if expansion_factor == 0:
                    expansion_factor = 1.0
            except Exception:
                expansion_factor = 1.0

            predicted_atoms = curr_atoms * expansion_factor

            if max_atoms is not None and predicted_atoms > max_atoms:
                continue

            new_history = history + [(sub_hall, trans)]
            new_visited = visited | {sub_hall}

            label_parts = [f"{h}({clean_trans_str(t)})" for h, t in new_history]
            path_label = "->".join(label_parts)

            path_info = {
                "target_hall": sub_hall,
                "label": path_label,
                "path": new_history,
                "predicted_atoms": predicted_atoms,
                "depth": next_gen
            }

            if sub_hall not in reachability_map[next_gen]:
                reachability_map[next_gen][sub_hall] = []

            reachability_map[next_gen][sub_hall].append(path_info)
            queue.append((int(sub_hall), predicted_atoms, next_gen, new_visited, new_history))

    for g in range(2, max_gen + 1):
        total_paths = sum(len(paths) for paths in reachability_map[g].values())
        print(f"  - Gen {g}: {len(reachability_map[g])} space groups ({total_paths} paths)")

    return reachability_map


def apply_path_transformations(self, chosen_path):
    """
    Sequentially applies subgroup transformations along `chosen_path` starting from the parent phase.
    """
    current_hall = self["hall_number"]

    for step_hall, step_trans in chosen_path["path"]:
        make_subgroup_structure(self, current_hall, step_hall, step_trans)
        self()
        current_hall = self["hall_number"]



def get_subgroup_candidates(self, parent_hall_number=None):
    """
    Retrieves the list of international numbers for possible maximal subgroups.
    """
    db = Database()
    if parent_hall_number is None:
        parent_hall_number = self["hall_number"]
    
    # Database returns a list of dictionaries including transformation matrices
    cands = db.get_subgroup_candidates(str(parent_hall_number))

    if cands is None:
        return None

    cands = cands.dropna(subset=["sub_hall_number"])
    cands = cands[cands["sub_hall_number"] != parent_hall_number]
    cands["sub_hall_number"] = cands["sub_hall_number"].fillna(0).astype(int)
    cands.rename(columns={"sub_hall_number": "hall_number", 
                          "transformation_str": "transformation"}, inplace=True)
    return cands

def make_subgroup_structure(self, 
                            parent_hall_number=None, 
                            sub_hall_number=None,
                            transformation_str:str=None,
                            symprec=0.01):
    """
    Transforms the current structure into a specified subgroup symmetry,
    handling lattice expansion and inheriting atomic properties (occ, adp).
    """
    if self._state != "DONE": symmetry_operation(self)

    db = Database()
    if parent_hall_number is None:
        parent_hall_number = db.get_hall_number(hm_short=self["space_group"], 
                                                setting=self["space_group","setting"])
        if parent_hall_number is None:
            log = f"No hall number found for space group {self['space_group']}"
            self._logger.info(log)
            return log
        
    res = db.get_subgroup_conversions(parent_hall_number=parent_hall_number, 
                                      sub_hall_number=sub_hall_number, 
                                      transformation_str=transformation_str)

    if res is None or res.empty:
        log = f"No transformation found to subgroup"
        self._logger.info(log)
        return log

    self()
    res = res.iloc[0].to_dict()
    matrix = np.array(res["matrix_p2s"])
    origin_shift = np.array(res["shift_p2s"])

    # 3. Generate translational vectors T to cover the expanded lattice
    # Determine the search range based on matrix P
    inv_p = np.linalg.inv(matrix)
    t_range = np.arange(-2, 3) # Sufficient for most common index 2-4 transformations
    tt, uu, vv = np.meshgrid(t_range, t_range, t_range)
    all_t = np.vstack([tt.flatten(), uu.flatten(), vv.flatten()]).T
    
    # Filter T such that P^-1 * T is within the new unit cell [0, 1)
    valid_t = []
    for t in all_t:
        shifted_t = np.dot(inv_p, t)
        if np.all((shifted_t >= -0.001) & (shifted_t < 0.999)):
            valid_t.append(t)

    # 4. Map all atoms into the new subgroup lattice

    sub_sites = {}    
    # Transformation: x' = P^-1 * (x + T - s)
    for t in valid_t:
        for key in self._expandables:
            if key not in self.model:
                continue

            if key not in sub_sites:
                sub_sites[key] = []

            if key == "xyz":
                # Apply transformation to all atoms in the P1 model
                xyz = np.dot(inv_p, (self.model["xyz"] + t - origin_shift).T).T
                xyz = xyz % 1.0
                sub_sites[key].extend(xyz)
            else:
                sub_sites[key].extend(self.model[key])

    for key, val in sub_sites.items():
        if key == "xyz":
            sub_sites[key] = np.array(val).reshape(-1, 3)
        else:
            sub_sites[key] = np.array(val).flatten()
    sub_sites["site_index"] = np.arange(np.shape(sub_sites["xyz"])[0])

    # numbers_expanded = self.get_atomic_numbers(sub_sites["symbol"])

    # 5. Transform lattice vectors: G' = G * P^T
    g_old = self["lattice_vector"]
    g_new = np.dot(matrix.T, g_old)

    au_sites = apply_symmetry_operations(self, 
                                         hall_number=sub_hall_number, 
                                         base_sites=sub_sites, 
                                         lattice_vector=g_new, 
                                         mode="au",
                                         symprec=symprec)
    # 8. Update Phase object
    self["lattice_vector"] = g_new
    # self.set_lattice_from_vector(lattice_vector=g_new)
    self.update(au_sites)
    self["hall_number"] = sub_hall_number

    self._state = "INIT"
    set_wyckoff_letters(self)

    log = f"Structure transformation from hall {parent_hall_number} to hall {sub_hall_number} under {transformation_str}"
    self._logger.info(log)
    return log


## Supergroups

def get_supergroup_candidates(self, sub_hall_number=None):
    """
    Retrieves the list of hall numbers for possible minimal supergroups
    using the newly prepared reverse-lookup table.
    """
    db = Database()
    if sub_hall_number is None:
        sub_hall_number = self["hall_number"]
    
    # Retrieve candidates from supergroup table
    cands = db.get_supergroup_candidates(sub_hall_number)
    if cands is None:
        return None
    
    cands = cands.dropna(subset=["parent_hall_number"])
    cands = cands[cands["parent_hall_number"] != sub_hall_number]
    cands["parent_hall_number"] = cands["parent_hall_number"].fillna(0).astype(int)
    cands.rename(columns={"parent_hall_number": "hall_number",
                          "parent_hm_full": "hm_full",
                          "transformation_str": "transformation"}, inplace=True)
    return cands

def make_supergroup_structure(self,                             
                            parent_hall_number=None, 
                            sub_hall_number=None,
                            transformation_str:str=None,    
                            symprec=0.01):
    """
    Transforms the current structure into a specified supergroup symmetry.
    Handles lattice contraction, merges overlapping sites, and inherits properties.
    """
    if self._state != "DONE": symmetry_operation(self)

    db = Database()
    if sub_hall_number is None:
        sub_hall_number = self["hall_number"]

    res = db.get_supergroup_conversions(sub_hall_number=sub_hall_number, 
                                        parent_hall_number=parent_hall_number, 
                                      transformation_str=transformation_str)
    if res is None or res.empty:
        self._logger.info(f"No transformation found to supergroup")
        return False

    res = res.iloc[0].to_dict()
    matrix_inv = np.array(res["matrix_s2p"])
    origin_shift_inv = np.array(res["shift_s2p"])

    xyz_sub = self.model["xyz"]    
    xyz_super_raw = (np.dot(matrix_inv, xyz_sub.T).T - origin_shift_inv) % 1.0
    
    # 4. Transform lattice vectors: G_super = G_sub * (P_inv)^T
    g_sub = self["lattice_vector"]
    g_super = np.dot(matrix_inv.T, g_sub)

    sup_sites = np.copy(self.model)
    sup_sites = {"xyz": xyz_super_raw}

    au_sites = apply_symmetry_operations(self, 
                                         base_sites=sup_sites, 
                                         lattice_vector=g_super, 
                                         hall_number=parent_hall_number, 
                                         mode="au",
                                         symprec=symprec)


    # 8. Update the Phase object
    self.set_lattice_from_vector(lattice_vector=g_super)
    self.update(au_sites)

    self["hall_number"] = parent_hall_number
    self._state = "INIT"
    set_wyckoff_letters(self)
    self._logger.info(f"Structure elevated to supergroup, hall number: {parent_hall_number}")
    
def get_site_symmetry_symbols(self):
    db = Database()
    wyckoff_letters = self["wyckoff_letters"]
    site_symmetry_symbols = []
    for wl in wyckoff_letters:
        if wl is None: site_symmetry_symbols.append(None)
        else: 
            res = db.get_site_symmetry_symbol(hall_number=self["hall_number"],
                                              wyckoff_letter=wl)
            site_symmetry_symbols.append(res)
        
    return site_symmetry_symbols



    

    
