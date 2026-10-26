import numpy as np
import copy
from itertools import compress
from orochi.base.core import OrochiBase
from .submodules import lattice as sblat
from .submodules import symmetry as sbsym
from .submodules import modify as sbmod
from .submodules import composition as sbcom
from .submodules import atom_info as sbatom
from .submodules import properties as sbpro
from .submodules import constraint as sbcon
from .submodules import evaluation as sbeval
from .submodules import explain as sbexp
from .database.atoms import AtomInformation as ai
from .params import default


class Phase(OrochiBase):
    """
    Manages material structural information.

    This class handles crystallographic and atomic information for a material
    phase, including lattice parameters, space group symmetry, site modifications,
    chemical composition, and derived properties. It inherits from `OrochiBase`
    for parameter management and integrates various submodules for specialized
    calculations.

    Attributes
    ----------
    model : dict
        Stores the crystallographic model data after processing, often including
        expanded atomic sites, their properties (xyz, occ, adp, symbol, etc.).
        Populated and used by various methods, especially after symmetry operations.
    update_space_group_constraints : bool
        Flag indicating whether to update constraints based on space group symmetry.
        If True, certain operations like `transform_into_standard_cell` might be
        triggered. Default is False.
    keep_space_group : bool
        Flag indicating whether to preserve the current space group setting during
        updates or transformations. Default is False.

    Parameters / Keys
    -----------------
    Key parameters accessible via dictionary indexing (e.g., `phase["lattice"]`):
    - "file" : Input file name for phase data.
    - "composition" : Chemical composition string or dictionary.
    - "merge_distance" : Distance tolerance for merging atomic sites.
    - "symbol" : Atomic symbols for each site (e.g., `np.array(["Fe", "O"])`).
    - "label" : User-defined labels for each site (e.g., `np.array(["Fe1", "O1"])`).
    - "xyz" : Fractional atomic coordinates (e.g., `np.array([[0,0,0], [0.5,0.5,0.5]])`).
    - "occ" : Site occupancies (e.g., `np.array([1.0, 1.0])`).
    - "adp" : Atomic Displacement Parameters (e.g., isotropic $U_{iso}$ or anisotropic $U_{ij}$).
    - "multiplicity" : Site multiplicities derived from symmetry.
    - "charge" : Formal charges of atoms at each site.
    - "wyckoff_letters" : Wyckoff letters for each site.
    - "site_symmetry_symbols" : Site symmetry symbols (e.g., "m", "-1").
    - "lattice" : Lattice parameters [a, b, c, alpha, beta, gamma]
      (e.g., `np.array([5.0, 5.0, 5.0, 90, 90, 90])`).
    - "space_group" : Space group information (symbol or number).
    - "pbc" : Periodic boundary conditions (e.g., `[True, True, True]`).
    - "map" : Grid-type data associated with the phase.
    - "material_type" : Classification of material (e.g., "oxide", "metal").
    - "shape_type" : Type of particle shape ("analytical", "model", "polygon", "sphere").
    - "sphere" : Parameters for spherical or ellipsoidal shape.
    - "polygon" : Parameters for polygonal shape.
    - "cylinder" : Parameters for cylindrical shape.
    - "surface" : Parameters for surface termination.

    Notes
    -----
    The attributes `lattice_sites`, `mainphase`, `subphase`, `site_symmetry_operation`,
    `unique_site`, `status`, and `substitute_keys` from the original docstring
    are not explicitly initialized as instance attributes in the provided `__init__`
    method. Some of these functionalities might be handled by dictionary indexing (e.g., `self[key]`),
    `self.model`, or methods from submodules. For example, `site_symmetry_operation`
    is likely a result obtained from a symmetry submodule function.

    Many methods related to lattice, symmetry, modification, composition,
    analysis, and constraints are dynamically assigned from submodules
    (sblat, sbsym, etc.) at the class level. Their detailed docstrings
    should be consulted in their respective submodule definitions.

    See Also
    --------
    orochi.base.core.OrochiBase : Base class for parameter handling.
    orochi.phase.submodules.lattice : Functions for lattice manipulations.
    orochi.phase.submodules.symmetry : Functions for crystallographic symmetry.
    orochi.phase.submodules.modify : Functions for modifying atomic structures.
    orochi.phase.submodules.composition : Functions for handling chemical composition.
    orochi.phase.database.atoms.AtomInformation : Database for atomic properties.
    orochi.phase.database.bond_valence.BondValence : Database for bond valence parameters.
    orochi.phase.params.default : Default parameters for the Phase class.

    """

    def __init__(self):
        """
        Initialize the Phase class.

        Sets default parameters from `orochi.phase.params.default`,
        initializes the `model` dictionary for storing structural data,
        and sets flags related to space group constraints.
        """
        self._par = default.par
        self._par_key_info = default.par_key_info

        super().__init__()

        self.model = {}
        self.update_space_group_constraints = False
        self.keep_space_group = False
        self._state = "INIT"

        self._name = "orochi.phase"
        self._id = "0"

        self._expandables = [
                "xyz",
                "occ",
                "adp",
                "label",
                "symbol",
                "group",
                "charge",
                "radius",
                "wyckoff_letters"
            ]

        self._method_dependencies = {
            "set_symmetry_constraints": {"self": ["space_group", "wyckoff_letters"]},
            "symmetry_operation": {"self": ["xyz", "lattice", "lattice_vector", "pbc", "space_group", "occ", "adp", "label", "symbol", "charge", "radius", "shape_type", "sphere", "polygon", "cylinder", "surface"]},
            # "set_lattice_system":{"self": ["space_group"]},
            # "set_full_space_group":{"self": ["space_group"]},
            "set_space_group_number":{"self": ["space_group"]},
            # "set_lattice_vector":{"self": ["lattice"]},
            # "set_lattice_from_vector":{"self": ["lattice_vector"]},
        }

        self.subphase = {}

    @property
    def expandables(self):
        """Backward-compatible property for internal _expandables."""
        return self._expandables

    @expandables.setter
    def expandables(self, value):
        self._expandables = value

    def split_into_subphase(self, site_indices=None, element_types=None, subphase_key="subphase"):
        """
        Extract specified site indices from the current phase into a new Phase instance (subphase).
        """
        if element_types is not None:
            if isinstance(element_types, str):
                element_types = [element_types]
            symbols = self["symbol"]
            add_site_indices = np.where(np.isin(symbols, element_types))[0]
            
            if site_indices is None:
                site_indices = add_site_indices.tolist()
            else:
                site_indices = sorted(list(set(site_indices) | set(add_site_indices)))
                
        num_sites = self["xyz"].shape[0]

        # 1. Create boolean mask from site_indices
        mask = np.zeros(num_sites, dtype=bool)
        mask[site_indices] = True

        # 2. Create a new Phase instance and copy parameters
        new_phase = Phase()
        new_phase.par = copy.deepcopy(self.par)

        # 3. Extract and remove from each expandable attribute
        for key in self._expandables:
            if key != "xyz":
                if np.size(self[key]) == num_sites:
                    pass
                else:
                    continue

            for attr, vals in self.par[key].items():
                all_vals = copy.deepcopy(vals)
                
                # --- (1) NumPy array ---
                if isinstance(all_vals, np.ndarray) and len(all_vals) == num_sites:
                    new_phase.par[key][attr] = all_vals[mask]
                    self.par[key][attr] = all_vals[~mask]
                
                # --- (2) List / tuple (including variable-length lists such as candidates, excluding str) ---
                elif isinstance(all_vals, (list, tuple)) and len(all_vals) == num_sites:
                    new_phase.par[key][attr] = [v for v, m in zip(all_vals, mask) if m]
                    self.par[key][attr] = [v for v, m in zip(all_vals, mask) if not m]
                
                # --- (3) Single string, scalar, or global Phase parameter ---
                else:
                    new_phase.par[key][attr] = copy.deepcopy(all_vals)

        # 4. Register in self.subphases
        if hasattr(self, "subphases") and isinstance(self.subphases, dict):
            self.subphases[subphase_key] = new_phase

        self._reset_constraints()
        new_phase._reset_constraints()
        self()
        new_phase()

        return new_phase

    def merge_subphase(self, subphase_name: str = None):
        """
        Re-integrate specified or all stored subphases back into the main phase
        using the existing `__add__` (__iadd__) operator logic.

        Args:
            subphase_name (str, optional): Key identifier of the subphase to merge.
                If None, all subphases in `self.subphases` will be merged sequentially.
                Defaults to None.

        Returns:
            Phase: self (updated in-place with merged model data).
        """
        if not hasattr(self, "subphases") or not self.subphases:
            print("[Info] No subphases available to merge.")
            return self

        # Identify target subphase keys to merge
        if subphase_name is not None:
            if subphase_name not in self.subphases:
                raise KeyError(f"Subphase '{subphase_name}' not found in self.subphases.")
            keys_to_merge = [subphase_name]
        else:
            keys_to_merge = list(self.subphases.keys())

        current = self
        remaining_subphases = copy.copy(self.subphases)

        # Merge sequentially using the `+` operator (__add__)
        for key_name in keys_to_merge:
            subphase = remaining_subphases.pop(key_name)
            current = current + subphase  # Automatically transforms lattice coordinates and merges models
            print(f"[Merge] Re-integrated '{key_name}' into main Phase via __add__.")

        # Write merged result back to self in place
        self.model = current.model
        self.update(current.par)
        self.subphases = remaining_subphases
        self._state = "INIT"
        self()
        return self
          

    # --- Lattice operations (submodules.lattice) ---
    def set_lattice_from_vector(self, *args, **kwargs):
        """
        Set lattice parameters from lattice vectors.

        Parameters
        ----------
        *args : tuple
            Arguments passed to sblat.set_lattice_from_vector.
        **kwargs : dict
            Keyword arguments passed to sblat.set_lattice_from_vector.
        """
        return sblat.set_lattice_from_vector(self, *args, **kwargs)

    def get_lattice_vector(self, *args, **kwargs):
        """
        Calculate the lattice matrix (Cartesian vectors) from lattice parameters.

        Returns
        -------
        numpy.ndarray
            A 3x3 matrix representing the lattice vectors.
        """
        return sblat.get_lattice_vector(self, *args, **kwargs)

    # def set_lattice_system(self, *args, **kwargs):
    #     """
    #     Determine and set the crystal system based on the space group.

    #     Parameters
    #     ----------
    #     *args, **kwargs
    #         Passed to sblat.set_lattice_system.
    #     """
    #     return sblat.set_lattice_system(self, *args, **kwargs)

    # --- Symmetry operations (submodules.symmetry) ---
    def symmetry_operation(self, *args, **kwargs):
        """
        Apply space group symmetry operations to the asymmetric unit.
        """
        return sbsym.symmetry_operation(self, *args, **kwargs)

    # def get_site_symmetry_operation(self, *args, **kwargs):
    #     """
    #     Retrieve symmetry operations for specific Wyckoff positions.
    #     """
    #     return sbsym.get_site_symmetry_operation(self, *args, **kwargs)
    
    def get_site_symmetries(self, *args, **kwargs):
        """
        Retrieve symmetry operations, Wyckoff positions etc.
        """
        return sbsym.get_site_symmetries(self, *args, **kwargs)

    # def set_site_symmetries(self, *args, **kwargs):
    #     """
    #     Set symmetry operations, Wyckoff positions etc.
    #     """
    #     return sbsym.set_site_symmetries(self, *args, **kwargs)

    def set_wyckoff_letters(self, *args, **kwargs):
        """
        Set Wyckoff letters for each site.
        """
        return sbsym.set_wyckoff_letters(self, *args, **kwargs)

    def set_wyckoff_candidates(self, *args, **kwargs):
        """
        Set candidate Wyckoff letters for each site.
        """
        return sbsym.set_wyckoff_candidates(self, *args, **kwargs)

    def transform_into_standard_cell(self, *args, **kwargs):
        """
        Transform the current unit cell into a standard crystallographic setting.
        """
        return sbsym.transform_into_standard_cell(self, *args, **kwargs)

    def make_p1_structure(self, *args, **kwargs):
        """
        Expand the asymmetric unit into a full P1 unit cell.
        """
        return sbsym.make_p1_structure(self, *args, **kwargs)

    # def apply_site_symmetry_operation(self, *args, **kwargs):
    #     """
    #     Apply site-specific symmetry operations to a set of coordinates.
    #     """
    #     return sbsym.apply_site_symmetry_operation(self, *args, **kwargs)

    def check_alternative_space_group(self, *args, **kwargs):
        """
        Check for higher or alternative symmetry descriptions of the structure.
        """
        return sbsym.check_alternative_space_group(self, *args, **kwargs)

    # --- Modification operations (submodules.modify) ---
    def make_translated_model(self, *args, **kwargs):
        """
        Translate a specific subset of atomic sites.
        """
        return sbmod.make_translated_model(self, *args, **kwargs)

    def make_rotated_model(self, *args, **kwargs):
        """
        Rotate a specific subset of atomic sites around a center.
        """
        return sbmod.make_rotated_model(self, *args, **kwargs)

    def group_modification(self, *args, **kwargs):
        """
        Peform structure modification for groups under rigid body constraints.
        """
        return sbmod.group_modification(self, *args, **kwargs)

    def translation(self, *args, **kwargs):
        """
        Perform a rigid body translation of the entire phase.

        Uses 'translation_vector' from self.par to displace all atoms.
        """
        return sbmod.translation(self, *args, **kwargs)

    def rotation(self, *args, **kwargs):
        """
        Perform a rigid body rotation of the entire phase.

        Uses 'rotation_matrix' and 'rotation_center' from self.par.
        The operation is performed in Cartesian space to prevent distortion.
        """
        return sbmod.rotation(self, *args, **kwargs)

    def get_atom_info_dicionary(self, *args, **kwargs):
        """
        Retrieve physical data (radius, scattering factors, etc.) for atoms.
        """
        return sbatom.get_atom_info_dicionary(self, *args, **kwargs)

    # --- Composition operations (submodules.composition) ---
    def set_symbol_occ_adp_from_string(self, *args, **kwargs):
        """
        Parse and set chemistry information from a descriptive string.
        """
        return sbcom.set_symbol_occ_adp_from_string(self, *args, **kwargs)

    def set_composition_from_txt(self, *args, **kwargs):
        """
        Set chemical composition from a text file or multi-line string.
        """
        return sbcom.set_composition_from_txt(self, *args, **kwargs)

    # --- Analysis operations (submodules.analysis) ---
    def get_formal_charge(self, *args, **kwargs):
        """
        Estimate formal charges based on the chemical environment.
        """
        return sbpro.get_formal_charge(self, *args, **kwargs)

    def get_material_type(self, *args, **kwargs):
        """
        Identify the class of material (e.g., metal, oxide, alloy).
        """
        return sbpro.get_material_type(self, *args, **kwargs)

    def get_atom_radius(self, *args, **kwargs):
        """
        Retrieve atomic or ionic radii for the current elements.
        """
        return sbpro.get_atom_radius(self, *args, **kwargs)

    def get_coordination_number(self, *args, **kwargs):
        """
        Calculate the coordination number for each atomic site.
        """
        return sbpro.get_coordination_number(self, *args, **kwargs)

    def get_site_relationship(self, *args, **kwargs):
        """
        Analyze distances and charge interactions between sites.
        """
        return sbpro.get_site_relationship(self, *args, **kwargs)

    def get_bond_relationship(self, *args, **kwargs):
        """
        Analyze chemical bonding based on distance and valence criteria.
        """
        return sbpro.get_bond_relationship(self, *args, **kwargs)

    # --- Evalution operatoins (submodules.evaluation) ---
    def eval_site_scores(self, *args, **kwargs):
        """
        Computes multiple site-wise physical and chemical scores in a single pass.
        """
        return sbeval.site_scores(self, *args, **kwargs)

    # --- Constraint operations (submodules.constraint) ---
    def set_spherical_constraints(self, *args, **kwargs):
        """Apply spherical boundary constraints to atomic positions.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to ``submodules.constraint.set_spherical_constraints``.
        """
        return sbcon.set_spherical_constraints(self, *args, **kwargs)

    def set_lattice_constraints(self, *args, **kwargs):
        """Apply crystallographic constraints to lattice parameters based on crystal system.

        Ensures equality and angular invariants (e.g. a=b=c, alpha=beta=gamma=90 for cubic)
        during multi-parameter refinement.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to ``submodules.constraint.set_lattice_constraints``.
        """
        return sbcon.set_lattice_constraints(self, *args, **kwargs)

    def set_symmetry_constraints(self, *args, **kwargs):
        """Apply Wyckoff site symmetry constraints to atomic coordinates.

        Restricts atomic coordinate shifts to invariant special positions defined by
        the space group Wyckoff site.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to ``submodules.constraint.set_symmetry_constraints``.
        """
        return sbcon.set_symmetry_constraints(self, *args, **kwargs)

    def set_elemental_adp_constraints(self, *args, **kwargs):
        """Apply elemental equality constraints to isotropic atomic displacement parameters.

        Constrains all atoms of the same chemical element to share identical Uiso values
        during structural refinement.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to ``submodules.constraint.set_elemental_adp_constraints``.
        """
        return sbcon.set_elemental_adp_constraints(self, *args, **kwargs)

    # LLM IO functions
    def explain(self, *args, **kwargs):
        """Generate structured diagnostic and feasibility reports for the structural model.

        Parameters
        ----------
        *args, **kwargs
            Optional arguments passed to explanation formatting modules.

        Returns
        -------
        dict
            Dictionary containing 'structure_parameters' and 'structure_feasibility' reports.
        """
        reports = {}
        reports["structure_parameters"] = {
            "report":sbexp.structure(self, *args, **kwargs),
            "type":"conditions",
            "dtype": "text"
            }

        scores = sbeval.site_scores(self, *args, **kwargs)
        reports["structure_feasibility"] = {
            "report":sbexp.feasibility(self, scores, *args, **kwargs),
            "type":"results",
            "dtype": "text"
            }
        return reports

    def make_subgroup_structure(self, hall_number: int=None, transformation:str=None):
        """Lower the symmetry of the current structure to a specified subgroup.

        Enables atomic coordinate displacements or lattice distortions along
        group-subgroup transformation pathways.

        Parameters
        ----------
        hall_number : int, optional
            Target Hall space group number for the 1-step transition.
        transformation : str, optional
            Transformation matrix string relating parent and subgroup coordinate axes.

        Returns
        -------
        str
            Log message detailing the symmetry descent and Wyckoff splitting.
        """
        logs = ""

        df = sbsym.get_subgroup_candidates(self)
        if df is None or df.empty:
            logs += "No subgroup candidates found."
            self._logger.info(logs)
            return logs

        markdown = df.to_markdown(index=False, tablefmt="github")
    
        if hall_number is None:
            max_idx = df["hall_number"].idxmax()
            hall_number = df.loc[max_idx]["hall_number"]
            if transformation is None:
                transformation = df.iloc[max_idx]["transformation"]
        elif transformation is None:
            transformation = df[df["hall_number"] == hall_number]["transformation"].values
            if len(transformation) == 0:
                logs += "No transformation found for the given hall number."
                self._logger.info(logs)
                return logs
            transformation = transformation[0]

        info = f"Subgroup candidates: \n{markdown}\n"
        # self._logger.info(info)
        log = sbsym.make_subgroup_structure(self, 
                                            parent_hall_number=self["hall_number"], 
                                            sub_hall_number=hall_number, 
                                            transformation_str=transformation)
        logs = info + log
        return logs

    def get_subgroup_candidates(self, format="full"):
        """
        Extracts valid subgroup space group numbers from the internal database 
        based on the current space group number.
        
        If format is "full", traces all unique 1-step subgroup transitions 
        step-by-step down to P1 (Hall 1). [Updated to Flat-Edge Map]
        """
        start_hall = int(self["hall_number"])

        if format in ["full", "tree", "paths"]:
            queue = [start_hall]
            visited_nodes = set()
            parent_order = []  # Record topological exploration order of parent nodes
            
            transitions = []
            seen_transitions = set()
            hm_map = {start_hall: self["space_group", "full"]}

            while queue:
                current_hall = queue.pop(0)
                if current_hall in visited_nodes:
                    continue
                visited_nodes.add(current_hall)
                
                # Record parent in BFS exploration order
                parent_order.append(current_hall)

                if current_hall == 1:
                    continue

                df = sbsym.get_subgroup_candidates(self, parent_hall_number=current_hall)
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    sub_hall = int(row["hall_number"])
                    hm = row["hm_full"]
                    trans = row["transformation"]

                    if sub_hall == current_hall and (trans == "a,b,c" or trans == "identity"):
                        continue

                    if sub_hall not in hm_map:
                        hm_map[sub_hall] = hm

                    transition_key = (current_hall, sub_hall, trans)
                    if transition_key not in seen_transitions:
                        seen_transitions.add(transition_key)
                        transitions.append({
                            "parent": current_hall,
                            "sub": sub_hall,
                            "trans": trans
                        })

                        if sub_hall not in visited_nodes and sub_hall != 1:
                            queue.append(sub_hall)

            # Pass topological parent_order to formatter
            return self._format_transition_edges(transitions, hm_map, parent_order)

        # ======================================================================
        # Direct (1-step) maximal subgroups
        # ======================================================================
        else:
            df = sbsym.get_subgroup_candidates(self, parent_hall_number=start_hall)

            if format == "text":
                if df is None:
                    return "No subgroup candidates found."
                return df.to_string(index=False)
            elif format == "markdown":
                if df is None:
                    return "No subgroup candidates found."
                return df.to_markdown(index=False, tablefmt="github")
            else:
                if df is None:
                    return None
                return df

    def suggest_subgroup_hall_number(self, max_gen=3, max_atoms=None):
        """
        Searching the best hall number (space group).     
        
        """
        hall_number = sbsym.suggest_subgroup_hall_number(self, max_gen=max_gen, max_atoms=max_atoms)
        sbsym.set_wyckoff_candidates(self)
        return hall_number

    def _format_transition_edges(self, transitions, hm_map, parent_order):
        """
        Format subgroup transition edges ordered by parent_order (BFS exploration order).
        """
        if not transitions:
            return "No subgroup transition steps to P1 (Hall 1) found."

        from collections import defaultdict
        grouped = defaultdict(list)
        for t in transitions:
            grouped[t["parent"]].append(t)

        lines = [
            "### [CRYSTALLOGRAPHIC SYMMETRY DESCENT MAP TO P1]",
            "Below are the valid 1-step subgroup transitions step-by-step down to P1 (Hall 1).",
            "Use these specific transformations to build your next model:\n"
        ]

        sorted_parents = [p for p in parent_order if p in grouped]

        for parent in sorted_parents:
            parent_hm = hm_map.get(parent, f"Unknown ({parent})")
            lines.append(f"**From {parent_hm} ({parent}):**")
            
            for edge in grouped[parent]:
                sub_hall = edge["sub"]
                sub_hm = hm_map.get(sub_hall, f"Unknown ({sub_hall})")
                trans = edge["trans"]
                lines.append(f"  - {sub_hm} ({sub_hall}), `{trans}`")
            lines.append("")

        return "\n".join(lines)
 

    def make_supergroup_structure(self, hall_number: int=None, transformation:str=None):
        """
        Transition the current structure to a specified higher-symmetry supergroup.

        This action is recommended when the current model appears to have redundant
        degrees of freedom, or when experimental data (obs) can be effectively
        described by a simpler, more symmetric average structure. It helps in
        identifying the idealized underlying lattice.

        Parameters
        ----------
        hall_number : int, optional
            The target hall number to transition to. If not specified, the minimum
            hall number in the candidates list will be used.
        transformation : str, optional
            The description of transformation for the inverse transformation (from
            supergroup to subgroup). If not specified, the first candidate for the
            selected hall number will be used.

        Returns
        -------
        str
            A log report detailing the supergroup candidates and transition result.
        """
        logs = ""
        df = sbsym.get_supergroup_candidates(self)
        markdown = df.to_markdown(index=False, tablefmt="github")

        if hall_number is None:
            min_idx = df["hall_number"].idxmin()
            hall_number = df.iloc[min_idx]["hall_number"]
            if transformation is None:
                transformation = df.iloc[min_idx]["transformation"]
        elif transformation is None:
            transformation = df[df["hall_number"] == hall_number]["transformation"].values[0]

        logs += f"Supergroup candidates: \n{markdown}\n"
        log = sbsym.make_supergroup_structure(self, 
                                        parent_hall_number=hall_number, 
                                        sub_hall_number=self["hall_number"], 
                                        transformation_str=transformation)
        logs += log
        self._logger.info(logs)
        return logs
    
    def get_supergroup_candidates(self, format="markdown"):
        """
        Retrieve a list of valid supergroup space group numbers.

        Parameters
        ----------
        format : str, optional
            Output format: "markdown" (default), "text", or other (returns DataFrame).

        Returns
        -------
        str or pandas.DataFrame or None
            Candidates formatted according to `format`.
        """
        df = sbsym.get_supergroup_candidates(self)
        if format == "text":
            if df is None:
                return "No supergroup candidates found."
            return df.to_string(index=False)
        elif format == "markdown":
            if df is None:
                return "No supergroup candidates found."
            return df.to_markdown(index=False, tablefmt="github")
        else:
            if df is None:
                return None
            return df

    def make_supercell(self, 
                       a_axis: int = 1, 
                       b_axis: int = 1, 
                       c_axis: int = 1, 
                       ):
        """
        Expands the current unit cell into a supercell as a P1 structure.

        Parameters
        ----------
        a_axis : int, optional
            Multiplication factor along the a-axis. Default is 1.
        b_axis : int, optional
            Multiplication factor along the b-axis. Default is 1.
        c_axis : int, optional
            Multiplication factor along the c-axis. Default is 1.

        Returns
        -------
        str
            A message confirming the final structure's dimensions and its space group.
        """
        matrix = np.array([a_axis, b_axis, c_axis])        
        sbmod.make_supercell(self, matrix)
        
        msg = f"Supercell created: {a_axis}x{b_axis}x{c_axis}."
        msg += " Space group is set to P1 to allow independent atomic displacements."
        self._logger.info(msg)
        return msg

    def make_higher_symmetry_structure(self, symprec: float = 0.001):
        """
        Analyze current atomic arrangement to detect hidden higher symmetry.

        Reduces the model to its asymmetric unit under the identified space group.

        Parameters
        ----------
        symprec : float, optional
            The tolerance for symmetry detection in Angstroms. Default is 0.001.

        Returns
        -------
        str
            A detailed message describing the result of the symmetry elevation.
        """
        msg = sbsym.make_higher_symmetry_structure(self, symprec)
        if not msg:
            return "No higher symmetry was detected with the given precision. The model remains unchanged."
        
        return msg

    def update_lattice_parameters(self, 
                                  a: float = None, 
                                  b: float = None, 
                                  c: float = None, 
                                  alpha: float = None, 
                                  beta: float = None, 
                                  gamma: float = None):
        """
        Update the lattice parameters (unit cell constants) of the crystal structure.

        Lengths are in Angstroms (Å) and angles are in degrees (°).

        Parameters
        ----------
        a : float, optional
            Lattice constant a (Å).
        b : float, optional
            Lattice constant b (Å).
        c : float, optional
            Lattice constant c (Å).
        alpha : float, optional
            Lattice angle alpha (degree).
        beta : float, optional
            Lattice angle beta (degree).
        gamma : float, optional
            Lattice angle gamma (degree).
        """
        if a is not None: self["lattice", 0] = a
        if b is not None: self["lattice", 1] = b
        if c is not None: self["lattice", 2] = c
        if alpha is not None: self["lattice", 3] = alpha
        if beta is not None: self["lattice", 4] = beta
        if gamma is not None: self["lattice", 5] = gamma
        self._changed_parameters.add("lattice")


    def update_atomic_coordinates(self, site_id: int, x: float, y: float, z: float):
        """
        Update atomic coordinates for a specific site.

        Parameters
        ----------
        site_id : int
            Target site ID from the report.
        x : float
            New x-coordinate (fractional).
        y : float
            New y-coordinate (fractional).
        z : float
            New z-coordinate (fractional).
        """
        self["xyz", "value", site_id] = np.array([x, y, z])
        self._changed_parameters.add("xyz")

    def displace_site(self, site_id: int, direction: list, magnitude: float):
        """
        Displace an atomic site by a specified distance (Å) in a given direction.

        Parameters
        ----------
        site_id : int
            Target site ID from the report.
        direction : list of float
            Fractional direction vector [dx, dy, dz].
        magnitude : float
            Distance in Å.
        """

        try:
            g = self["lattice_vector"]
            v_frac = np.array(direction, dtype=float)
            
            # 1. Convert fractional direction vector to Cartesian coordinates
            v_cart = np.dot(g, v_frac)
            
            # 2. Normalize to the specified magnitude (Å) in Cartesian space
            norm = np.linalg.norm(v_cart)
            if norm > 0:
                v_displace_cart = (v_cart / norm) * magnitude
            else:
                v_displace_cart = np.zeros(3)
                
            # 3. Convert Cartesian displacement back to fractional coordinate increments
            g_inv = np.linalg.inv(g)
            d_frac = np.dot(g_inv, v_displace_cart)
            
            # 4. Update the coordinates of the specified site
            current_xyz = self["xyz", int(site_id)]
            new_xyz = current_xyz + d_frac
            self["xyz", int(site_id)] = new_xyz
            self._changed_parameters.add("xyz")
        except Exception as e:
            self._logger.info(f"Error in displace_site: {e}")


    def update_site_occupancy(self, site_id: int, occupancy: float):
        """
        Update the occupancy of a specific atomic site.

        Attempts to maintain the total elemental composition defined in `self["composition"]`.

        Parameters
        ----------
        site_id : int
            The ID of the target site from the report structure table.
        occupancy : float
            The new occupancy value for this site (typically 0.0 to 1.0).
        """
        target_symbol = self["symbol"][site_id]
        target_total = self["composition","nominal"].get(target_symbol)

        # 1. Update occupancy of the specified site
        self["occ", site_id] = occupancy
        self._changed_parameters.add("occ")

        # 2. Maintain overall composition (adjust only if other sites of the same element exist)
        if target_total is not None:
            other_site_indices = [i for i, s in enumerate(self["symbol"]) 
                                  if s == target_symbol and i != site_id]
            
            if other_site_indices:
                # Check current model composition (calculated on ASU basis rather than expanded model)
                # Calculate remaining required amount accounting for site multiplicity
                multiplicities = self["multiplicity"] # From symmetry submodule
                current_at_site = occupancy * multiplicities[site_id]
                remaining_needed = target_total - current_at_site
                
                current_other_sum = sum(self["occ"][i] * multiplicities[i] for i in other_site_indices)
                
                if current_other_sum > 0:
                    scale_factor = remaining_needed / current_other_sum
                    for i in other_site_indices:
                        self["occ", i] *= scale_factor
            else:
                # If no other sites are available for adjustment, allow the value as-is
                pass
            

    def get_composition_summary(self) -> str:
        """
        Return a summary comparing target composition and current model composition.

        Returns
        -------
        str
            Markdown formatted table of composition comparison.
        """
        model_comp = self.get_model_composition()
        target_comp = self["composition","nominal"]
        
        lines = ["| Element | Target | Current (Model) | Associated Site IDs |", 
                 "|---|---|---|---|"]

        all_symbols = np.append(list(target_comp.keys()), list(model_comp.keys()))
        
        unique_elements = np.unique(all_symbols)
        for sym in unique_elements:
            target = target_comp.get(sym, "Not set")
            current = model_comp.get(sym, 0.0)
            ids = [str(i) for i, s in enumerate(self["symbol"]) if s == sym]
            lines.append(f"| {sym} | {target} | {current:.4f} | {', '.join(ids)} |")
            
        return "\n".join(lines)

    def update_element_adps(self, element: str, adp_value: float):
        """
        Update the isotropic ADP (U_iso, Å²) for a specific element.

        Parameters
        ----------
        element : str
            The target element chemical symbol.
        adp_value : float
            New U_iso value.
        """
        adps = self["adp"]
        symbols = self["symbol"]
        self["adp"] = np.where(symbols == element, adp_value, adps)
        self._changed_parameters.add("adp")


    def update_particle_size(self, diameter: float):
        """
        Update the particle size (diameter in Å).

        Parameters
        ----------
        diameter : float
            New diameter value in Å.
        """
        self["sphere"] = [diameter]*3        
        self._changed_parameters.add("sphere")


    def update_site_adp(self, site_index: int, adp_value: float):
        """
        Update the isotropic ADP (U_iso, Å²) for a specific site.

        Parameters
        ----------
        site_index : int
            The target site ID.
        adp_value : float
            New U_iso value (Å²).
        """
        self["adp", site_index] = adp_value
        self._changed_parameters.add("adp")



    def __getitem__(self, key):
        """
        Retrieve a phase parameter or derived structural information.

        Allows item access using `[]`. It provides shortcuts for individual
        lattice parameters (e.g., "a", "b", "alpha") by mapping them to
        their corresponding string addresses in the `par` dictionary.
        For other keys, it calls the parent `OrochiBase.__getitem__`.

        Parameters
        ----------
        key : str or tuple
            If a string representing a lattice parameter shortcut ("a", "b", "c",
            "alpha", "beta", "gamma"), it's converted to an internal Orochi
            address. Otherwise, the key is passed to the superclass method.
            If a tuple, it's typically `(parameter_key, attribute_key)` for
            accessing nested parameter attributes.

        Returns
        -------
        any
            The value of the requested parameter or attribute.
        """
        if isinstance(key, str):
            if key in ["a", "b", "c", "alpha", "beta", "gamma"]:
                prefix = self["prefix"]
                delimiter = self["address_delimiter"]
                if key == "a":
                    id = "0"
                elif key == "b":
                    id = "1"
                elif key == "c":
                    id = "2"
                elif key == "alpha":
                    id = "3"
                elif key == "beta":
                    id = "4"
                elif key == "gamma":
                    id = "5"
                key = delimiter.join([prefix, "lattice", "value", id])

        if key == "lattice_vector":
            value = self.get_lattice_vector(self["lattice"])
            return value
        elif key in ["crystal_system", "lattice_system"]:
                from .database.symmetries import Database
                db = Database()
                crystal_system = db.get_crystal_system(self["hall_number"])
                return crystal_system
        elif "space_group" in key:
            if key == "space_group":
                return sbsym.get_space_group(self, key="hm_short")
            elif len(key) == 2:
                if key[1] in ["value", "hm_short"]:
                    return sbsym.get_space_group(self, key="hm_short")
                elif key[1] == "number":
                    return sbsym.get_space_group(self, key="table_no")
                elif key[1] in ["hm_full", "full"]:
                    return sbsym.get_space_group(self, key="hm_full")
                else:
                    return sbsym.get_space_group(self, key="hm_short")
            else:
                return sbsym.get_space_group(self, key="hm_short")
        elif "site_symmetry_symbols" in key:
            return sbsym.get_site_symmetry_symbols(self)
        elif "diameter" in key:
            return self["sphere"][0]            
        else:
            value = super().__getitem__(key)
            return value

    def __setitem__(self, key, value):
        """
        Set a phase parameter or structural information.

        Allows item assignment using `[]`. It provides shortcuts for individual
        lattice parameters. When setting parameters, it relies on the
        parent `OrochiBase.__setitem__` which handles parameter updates,
        logging changes, and potentially triggering dependent calculations.

        Parameters
        ----------
        key : str or tuple
            If a string representing a lattice parameter shortcut ("a", "b", "c",
            "alpha", "beta", "gamma"), it's converted to an internal Orochi
            address. Otherwise, the key is used directly.
            If a tuple, it's typically `(parameter_key, attribute_key, [indices...])`
            for setting nested parameter attributes or array elements.
        value : any
            The value to set for the specified parameter or attribute.

        Returns
        -------
        tuple
            A tuple `(key, attribute, index1, index2)` as returned by
            `OrochiBase.__setitem__`, representing the processed key,
            attribute, and indices used for the assignment.
        """

        if isinstance(key, str):
            if key in ["a", "b", "c", "alpha", "beta", "gamma"]:
                prefix = self["prefix"]
                delimiter = self["address_delimiter"]
                if key == "a":
                    id = "0"
                elif key == "b":
                    id = "1"
                elif key == "c":
                    id = "2"
                elif key == "alpha":
                    id = "3"
                elif key == "beta":
                    id = "4"
                elif key == "gamma":
                    id = "5"
                key = delimiter.join([prefix, "lattice", "value", id])

        if key == "lattice_vector":
            self.set_lattice_from_vector(value)
            return key, "value", None, None
        elif "space_group" in key:
            if key == "space_group":
                sbsym.set_space_group(self, value)
            elif len(key) == 2:
                if key[1] in ["value", "hm_short"]:
                    sbsym.set_space_group(self, hm_short=value)
                elif key[1] in ["number"]:
                    sbsym.set_space_group(self, table_no=value)
                elif key[1] in ["hm_full", "full"]:
                    sbsym.set_space_group(self, hm_full=value)
                else:
                    sbsym.set_space_group(self, value)
            else:
                sbsym.set_space_group(self, value)

            # self.set_site_symmetries()
            self.set_wyckoff_letters()
            return key, "value", None, None

        elif key == "composition":
            key, attribute, index1, index2 = super().__setitem__(key, value)   
            if attribute == "value":
                self.par["composition"]["nominal"] = sbcom.text_to_dictionary(value)
            return key, attribute, index1, index2

        elif key == "wyckoff_letters":
            key, attribute, index1, index2 = super().__setitem__(key, value)   
            if attribute == "value":
                self.set_symmetry_constraints()
        elif key == "diameter":
            key, attribute, index1, index2 = super().__setitem__("sphere", [value, value, value])        
            return key, attribute, index1, index2
        else:
            key, attribute, index1, index2 = super().__setitem__(key, value)        
            return key, attribute, index1, index2


    def __call__(self, force: bool = False):
        """
        Process the phase information and update the structural model.

        This method triggers several update steps:
        1. Calls the parent `OrochiBase.__call__` method.
        2. If `shape_type` is not "analytical", it calculates atomic radii
           and coordination numbers.
        3. Expands the crystallographic structure based on symmetry operations
           using `symmetry_operation`.
        4. If `shape_type` is not "analytical", it attempts to apply a shape
           model (e.g., sphere, polygon) to the phase using functions from
           `orochi.phase.submodules.shape.explicit`.

        Parameters
        ----------
        force : bool, optional
            If True, resets internal state to 'INIT' to force full symmetry expansion
            and shape truncation without skipping cached steps, by default False.
        """
        if force:
            self._state = "INIT"
        try:
            super().__call__(force=force)
            num_assym_atoms = np.shape(self["xyz"])[0]

            if np.size(self["group"]) != num_assym_atoms:
                self["group"] = np.zeros(num_assym_atoms, dtype=int)
            if np.size(self["charge"]) != num_assym_atoms:
                self["charge"] = np.zeros(num_assym_atoms, dtype=float)

            if self["shape_type"] != "analytical":
                if np.all(np.asarray(self["charge"], dtype=float) == 0.0):
                    self.get_formal_charge()
                self.get_atom_radius()
                self.get_coordination_number()

            self.symmetry_operation()
            self.group_modification()

            if self["shape_type"] != "analytical":
                if "cn" in self.par and "value" in self.par["cn"] and len(self.par["cn"]["value"]) == num_assym_atoms:
                    cn_strs = np.array([str(c) for c in self.par["cn"]["value"]], dtype=object)
                    self.model["coordination_number"] = cn_strs[self.model["site_index"]]
                from .submodules.shape import explicit
                try:
                    getattr(explicit, self["shape_type"])(self)
                except AttributeError:
                    pass

            if not np.allclose(self["rotation_matrix"], np.eye(3)):
                self.rotation()

            if not np.allclose(self["translation_vector"], np.zeros(3)):
                self.translation()

            self._state = "DONE"
        except:
            if self._state == "DONE":
                self._state = "INIT"
                self.__call__(force=True)
            else:
                self._state = "ERROR"

    def run(self, force: bool = False):
        """Explicit execution entry point for user scripts and interactive environments.

        Parameters
        ----------
        force : bool, optional
            If True, forces full recalculation by bypassing cached state, by default False.
        """
        return self.__call__(force=force)

    def __add__(self, other):
        """
        Merge this phase with another Phase instance including site-dependent parameters in self.par.
        """
        # 1. Update both instances to the latest state
        if self._state != "DONE": self()
        if other._state != "DONE": other()

        # 2. Retrieve lattice vectors and transform coordinates
        g_self = self["lattice_vector"]
        g_other = other["lattice_vector"]

        xyz_other_cart = np.dot(g_other, other.model["xyz"].T).T
        xyz_other_in_self_frac = np.dot(np.linalg.inv(g_self), xyz_other_cart.T).T

        # 3. Concatenate model data
        merged_model = {}
        for key in self.model:
            if key == "xyz":
                merged_model[key] = np.concatenate((self.model[key], xyz_other_in_self_frac))
            elif key in other.model:
                merged_model[key] = np.concatenate((self.model[key], other.model[key]))
            else:
                merged_model[key] = np.copy(self.model[key])

        # 4. Create new Phase and copy base parameters
        merged_phase = Phase()
        merged_phase.update(self.par)  # Inherit base parameters (lattice, space group, etc.)
        merged_phase.model = merged_model

        # 5. Merge site-dependent attributes (expandables) in self.par
        for key in self._expandables:
            if key in self.par and key in other.par:
                for attr, val1 in self.par[key].items():
                    if attr in other.par[key]:
                        val2 = other.par[key][attr]

                        # For NumPy arrays
                        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                            merged_phase.par[key][attr] = np.concatenate([val1, val2], axis=0)

                        # For lists (including variable-length lists such as candidates)
                        elif isinstance(val1, list) and isinstance(val2, list):
                            merged_phase.par[key][attr] = val1 + val2

        merged_phase._state = "DONE"
        return merged_phase

    def read_file(self, file=None):
        """
        Read phase information from a file.

        Updates the current phase instance with data from the specified file
        using the `update_phase` function (typically from `orochi.phase.io.files`).
        After reading, it initializes site indices and calls several `set_*`
        methods to ensure consistency of lattice and space group information.

        Parameters
        ----------
        file : str, optional
            Path to the file containing phase data. If provided, `self["file"]`
            is updated with this path. If `None`, `self["file"]` is used.
        """
        if file is not None:
            self["file"] = file

        from .io.files import read
        read(self, file)
        try:
            self.set_wyckoff_letters()
        except Exception:
            pass

  
    def save_file(self, filename=None):
        """
        Save phase information to a file.

        Parameters
        ----------
        filename : str, optional
            Path to the file containing phase data. If provided, `self["file"]`
            is updated with this path. If `None`, `self["file"]` is used.
        """
        from .io.files import save
        import os

        if filename is None:
            filename = os.path.join(self.work_dir, self.study_name + ".cif")
        else:
            ext = os.path.splitext(filename)[1]
            if len(ext) == 0 or ext is None:
                filename = os.path.splitext(filename)[0] + ".cif"

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        save(self, filename)

    def get_cell_volume(self):
        """
        Calculate the volume of the unit cell.

        The volume is computed from the determinant of the lattice vectors
        matrix (`self["lattice_vector"]`).

        Returns
        -------
        float
            The volume of the unit cell in cubic Angstroms ($Å^3$).
        """
        volume = np.linalg.det(self["lattice_vector"])
        return volume

    def get_occupied_volume(self):
        """
        Calculate the total volume occupied by atoms in the unit cell.

        This is estimated by summing the volumes of spherical atoms, using radii
        obtained from `self.get_radii()`.

        Returns
        -------
        float
            The total estimated volume occupied by atoms in cubic Angstroms ($Å^3$).
        """
        radii = self.get_radii()
        volume = np.sum((4.0 * np.pi * radii**3.0) / 3.0)
        return volume

    def get_atomic_volume(self, atom):
        """
        Retrieve the estimated spherical volume for a single atom type.

        Uses the atomic radius from the `AtomInformation` database.

        Parameters
        ----------
        atom : str
            The atomic symbol (e.g., "Fe", "O").

        Returns
        -------
        float
            The estimated volume of a single atom of the specified type
            in cubic Angstroms ($Å^3$).
        """
        info = ai()
        r = info.get_radius(symbol=atom)
        volume = (4.0 * np.pi * r**3.0) / 3.0
        return volume

    def get_atomic_numbers(self, symbols):
        """
        Retrieve atomic numbers for a list of atomic symbols.

        Parameters
        ----------
        symbols : list of str
            A list of atomic symbols (e.g., `["Fe", "O", "Si"]`).

        Returns
        -------
        list of int
            A list of atomic numbers corresponding to the input symbols.
        """
        numbers = []
        info = ai()
        for atom_symbol in symbols:
            symbol, charge = info.parse_symbol(atom_symbol)
            atomic_number = info.get_atomic_number(symbol)
            numbers.append(atomic_number)
        return numbers

    def get_atomic_symbols(self, numbers):
        """
        Retrieve atomic symbols for a list of atomic numbers.

        Parameters
        ----------
        numbers : list of int
            A list of atomic numbers (e.g., `[26, 8, 14]`).

        Returns
        -------
        list of str
            A list of atomic symbols corresponding to the input numbers.
        """
        symbols = []
        info = ai()
        for atom_number in numbers:
            symbols.append(info.get_symbol(atomic_number=atom_number))
        return symbols


    def get_density(self):
        """
        Calculate the crystallographic density of the phase.

        Density is computed using the cell volume, the sum of atomic masses
        (weighted by occupancy) in the unit cell, and Avogadro's number.
        Assumes `self.model` or `self.par` (if model not populated)
        contains the necessary "symbol" and "occ" data for all sites in the cell.

        Returns
        -------
        float
            The density of the phase in g/cm³.
        """
        volume = self.get_cell_volume()
        avogadro = 6.0221408e23

        sites_data_container = self.model if self.model and "symbol" in self.model else self.par

        site_symbols = np.array(sites_data_container["symbol"])
        site_occ = np.array(sites_data_container["occ"])

        mass_numbers = np.ones(site_symbols.size, dtype=float)
        unique_symbols = np.unique(site_symbols)

        for atom_symbol in unique_symbols:
            info = ai()
            info.read(atom_symbol)
            mass = info.get_mass()
            mass_numbers[site_symbols == atom_symbol] = mass

        total_mass_in_cell = np.sum(mass_numbers * site_occ)
        mass_grams = total_mass_in_cell / avogadro
        density = mass_grams / (volume * (1e-8)**3)
        return density

    def get_number_density(self):
        """
        Calculate the number density of atoms in the phase.

        Number density is the total number of atoms (considering occupancies)
        per unit cell volume. Assumes `self.model` or `self.par`
        contains the occupancy data for all sites in the cell.

        Returns
        -------
        float
            The number density in atoms per cubic Angstrom (atoms/$Å^3$).
        """
        volume = self.get_cell_volume()
        sites_data_container = self.model if self.model and "occ" in self.model else self.par
        total_occupied_sites = np.sum(sites_data_container["occ"])
        number_density = total_occupied_sites / volume
        return number_density

    def get_center(self, key="lattice"):
        """
        Calculate the geometric center of the lattice or all atomic sites.

        Parameters
        ----------
        key : str, optional
            Specifies the target for center calculation:
            - "lattice": Calculates the center considering periodic boundary
              conditions if `self["lattice", "centroid"]` is True using sites
              from `self.par["xyz"]`. If False, returns the pre-defined
              `self["lattice", "center"]`.
            - "all": Calculates the average position of all sites listed in
              `self.model["xyz"]`.
            Defaults to "lattice".

        Returns
        -------
        numpy.ndarray
            A 3-element array representing the Cartesian coordinates of the center.
        """
        if key == "lattice":
            if self["lattice", "centroid"]:
                sites_xyz = np.array(self.par["xyz"])
                center = np.average(sites_xyz, axis=0)
                pbc_zero = np.ones(3) - self["pbc"]
                pbc_one = np.abs(pbc_zero - 1.0)
                center *= pbc_zero
                center += pbc_one * 0.5
                return center
            else:
                return self["lattice", "center"]

        elif key == "all":
            if self.model and "xyz" in self.model:
                center = np.average(self.model["xyz"], axis=0)
                return center
            else:
                self._logger.warning("'model' data for 'xyz' not found for get_center(key='all').")
                return np.array([0.0, 0.0, 0.0])


    def get_model_composition(self):
        """
        Retrieve the chemical composition from the current structural model.

        Calculates the total occupancy for each unique atomic symbol present
        in `self.model`.

        Returns
        -------
        dict
            A dictionary where keys are atomic symbols (str) and values are
            their total summed occupancies (float) in the model. Returns an
            empty dictionary if model data is unavailable.
        """
        if not (self.model and "symbol" in self.model and "occ" in self.model):
            self._logger.warning("Model data (symbol, occ) not available for get_model_composition.")
            return {}

        symbols = np.array(self.model["symbol"])
        occ = np.array(self.model["occ"])

        model_composition = {}
        unique_symbols = np.unique(symbols)
        for atom_symbol in unique_symbols:
            model_composition[atom_symbol] = np.sum(occ[symbols == atom_symbol])

        return model_composition

    def get_radii(self):
        """
        Retrieve atomic radii for unique sites in the asymmetric unit.

        Radii are obtained from the `AtomInformation` database, considering
        the symbol and charge specified in `self.par` for the input sites
        (typically the asymmetric unit).

        Returns
        -------
        numpy.ndarray
            An array of atomic radii corresponding to the sites defined in
            `self.par["symbol"]`. The order matches these input sites.
        """
        symbols_in_par = np.array(self["symbol"])
        charges_in_par = np.array(self["charge"])

        radii = np.ones(symbols_in_par.size, dtype=float)
        info = ai()

        for i, atom_symbol in enumerate(symbols_in_par):
            symbol, charge = info.parse_symbol(atom_symbol)
            radii[i] = info.get_radius(symbol=symbol, charge=charge)
        return radii

    def get_shape_matrix_from_diameter(self):
        """
        Calculate the shape matrix for a sphere or ellipsoid from diameter(s).

        The shape matrix $M$ is such that $r^T M r = 1$ defines the surface.
        For an ellipsoid with semi-axes a, b, c, $M = diag((1/a)^2, (1/b)^2, (1/c)^2)$.
        Diameters are $2a, 2b, 2c$. So, $a = D_a/2$, etc.
        $M = diag((2/D_a)^2, (2/D_b)^2, (2/D_c)^2)$.

        Assumes `self["sphere"]` contains the diameter(s). If scalar, it's a sphere.
        If an array, it's diameters along principal axes for an ellipsoid.

        Returns
        -------
        numpy.ndarray
            A 3x3 diagonal matrix representing the shape matrix.
        """
        diameters = self["sphere"]
        if np.size(diameters) == 1:
            diag_elements = np.full(3, 2.0 / diameters)
        else:
            diag_elements = 2.0 / np.array(diameters)

        matrix = np.diag(np.square(diag_elements))
        return matrix

    def get_shape_volume(self):
        r"""
        Calculate the volume of an ellipsoidal shape.

        The volume is calculated using the semi-axes derived from the eigenvalues
        of the shape matrix `self["sphere","matrix"]`.
        Volume $V = (4/3) \pi a b c$. Semi-axes $a, b, c$ are $1/\sqrt{\lambda_i}$
        where $\lambda_i$ are eigenvalues of the shape matrix.

        Returns
        -------
        list of float
            A list containing a single float value: the volume of the ellipsoid
            in the same units as the input diameters squared (typically $Å^3$).
            Returns `[0.0]` if calculation is not possible (e.g., non-positive eigenvalues).
        """
        shape_matrix = self["sphere","matrix"]
        eigenvalues = np.linalg.eigvals(shape_matrix)
        positive_eigenvalues = eigenvalues[eigenvalues > 1e-9]
        if len(positive_eigenvalues) < 3:
            self._logger.warning("Shape matrix yields fewer than 3 positive eigenvalues for volume calculation.")
            return [0.0]

        semi_axes = 1.0 / np.sqrt(positive_eigenvalues[:3]) # Ensure only 3 are used
        a, b, c = semi_axes[0], semi_axes[1], semi_axes[2]
        volume = (4.0 / 3.0) * np.pi * a * b * c
        return [volume]