try:
    import optuna
except ImportError:
    optuna = None
import numpy as np
from ..database.atoms import AtomInformation as ai
from typing import Any, Dict, List, Union
from numpy.typing import NDArray


def get_formal_charge(self):
    """
    Determines and sets the formal charge of each atom in the crystal structure.

    This function uses the Optuna library to optimize the assignment of formal
    charges to atoms in a crystal structure, aiming to achieve charge neutrality.
    It considers the electronegativity and valence of the elements, and the
    material type.

    Parameters
    ----------
    self : Phase
        A Phase object containing the structural information, including atomic
        symbols and material type.

    Returns
    -------
    Phase
        The modified Phase object with the 'charge' attribute updated.
        The charge for each atom in the model is stored in  `self.model["charge"]`

    """
    mat_type = self["material_type"]
    if not mat_type or mat_type == "" or (isinstance(mat_type, (list, np.ndarray)) and (len(mat_type) == 0 or mat_type[0] == "")):
        get_material_type(self)

    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(_find_formal_charge(self, "optimize"), n_trials=100)
        self["charge"] = _find_formal_charge(self, "update")(study.best_params)
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        import itertools
        atom_information = get_atom_information(self)
        symbols = self["symbol"]
        if isinstance(symbols, str):
            symbols = np.array([symbols])
        candidates_per_site = []
        for idx, atom in enumerate(symbols):
            val = atom_information[atom]["valence"]
            vals = [float(v) for v in np.atleast_1d(val)] if np.size(val) > 0 else [0.0]
            if atom_information[atom]["number"] == 8 and self["material_type"] == "oxide":
                vals = [-2.0]
            candidates_per_site.append(vals)
        best_params = {str(idx): candidates_per_site[idx][0] for idx in range(len(symbols))}
        best_score = float("inf")
        update_fn = _find_formal_charge(self, "update")
        mult = np.asarray(self["multiplicity"], dtype=float) if np.size(self["multiplicity"]) == len(symbols) else np.ones(len(symbols))
        occ = np.asarray(self["occ"], dtype=float) if np.size(self["occ"]) == len(symbols) else np.ones(len(symbols))
        for combo in itertools.product(*candidates_per_site):
            trial_dict = {str(idx): combo[idx] for idx in range(len(symbols))}
            chg = update_fn(trial_dict)
            score = abs(float(np.sum(chg * mult * occ)))
            if score < best_score:
                best_score = score
                best_params = trial_dict
                if score == 0.0:
                    break
        self["charge"] = update_fn(best_params)

    if self.model and "occ" in self.model:
        if np.size(self["charge"]) == 1:
            self.model["charge"] = (
                np.ones_like(self.model["occ"]) * self["charge"]
            )
        elif "site_index" in self.model:
            self.model["charge"] = np.zeros(
                np.size(self.model["occ"])
            )
            for index, symbol in enumerate(self["symbol"]):
                r = np.where(self.model["site_index"] == index)
                self.model["charge"][r] = self["charge"][index]

    return self

def _find_formal_charge(self, work: str = "optimize") -> Union[callable, NDArray[np.float64]]:
    """
    Creates an objective function for optimizing formal charges or updates charges.

    This function defines how formal charges are determined for atoms in the structure.
    It can be used in two modes: 'optimize' and 'update'.

    In 'optimize' mode, it returns a function that Optuna will use to suggest
    charge values and evaluate the total charge of the structure.  Optuna will
    try to minimize the absolute value of the total charge.

    In 'update' mode, it returns a function that takes a trial (dictionary
    of optimized parameters from Optuna) and returns the charge array.

    Parameters
    ----------
    self : Phase
        A Phase object containing the structural information, including atomic
        symbols and material type.
    work : str, optional
        The operation mode: "optimize" or "update".
        - "optimize":  The function returns an objective function for Optuna.
        - "update": The function returns a function to update the charges
          based on optimization results.
        Default is "optimize".

    Returns
    -------
    callable or numpy.ndarray
        If work is "optimize", returns a function that takes an Optuna trial
        and returns the score (absolute total charge).
        If work is "update", returns a function that takes an Optuna trial
        (dictionary of optimized parameters) and returns the numpy array of atomic charges.
    """

    def get_value(trial: Any) -> Union[float, NDArray[np.float64]]:
        """
        Helper function to suggest or retrieve charge values.

        This function is used internally by `_find_formal_charge` to either
        suggest charge values to Optuna during optimization or to retrieve the
        optimized charge values from an Optuna trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The Optuna trial object.

        Returns
        -------
        float or numpy.ndarray
            If `work` is "optimize", returns the score (absolute total charge).
            If `work` is "update", returns the numpy array of atomic charges.
        """
        symbols = self["symbol"]
        if isinstance(symbols, str):
            symbols = np.array([symbols])

        charge = np.zeros(np.size(symbols))
        atom_information = get_atom_information(self)

        for index, atom in enumerate(symbols):
            element_type = atom_information[atom]["element_type"]

            if atom_information[atom]["charge"] != 0:
                charge[index] = atom_information[atom]["charge"]

            else:
                valence = atom_information[atom]["valence"]

                if self["material_type"] in ["metal", "metalloid"]:
                    charge[index] = 0
                    continue

                elif element_type == "rare_gas":
                    charge[index] = 0
                    continue

                elif element_type in ["alkali", "alkali_earth"]:
                    if self["material_type"] in [
                        "oxide",
                        "nitride",
                        "phosphide",
                        "halide",
                        "carbide",
                        "chalcogenide",
                    ]:
                        charge[index] = valence[0]

                    elif self["material_type"] == "metal":
                        charge[index] = 0.0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif element_type == "halogen":
                    if self["material_type"] == "halide":
                        charge[index] = valence[0]
                    elif self["material_type"] == "gas":
                        charge[index] = 0.0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif atom_information[atom]["number"] == 8:  # Oxygen
                    if self["material_type"] == "oxide":
                        charge[index] = -2
                    elif self["material_type"] == "organic":
                        charge[index] = 0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif atom_information[atom]["number"] == 7:  # Nitrogen
                    if self["material_type"] == "nitride":
                        charge[index] = -3
                    elif self["material_type"] == "organic":
                        charge[index] = 0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif atom_information[atom]["number"] == 6:  # Carbon
                    if self["material_type"] == "carbide":
                        charge[index] = -4
                    elif self["material_type"] == "organic":
                        charge[index] = 0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif atom_information[atom]["number"] == 5:  # Boron
                    if self["material_type"] == "organic":
                        charge[index] = 0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue
                elif atom_information[atom]["number"] == 15:  # Phosphorus
                    if self["material_type"] == "organic":
                        charge[index] = 0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif element_type == "chalcogen":
                    if self["material_type"] == "chalcogenide":
                        charge[index] = -2
                    elif self["material_type"] == "organic":
                        charge[index] = 0
                    else:
                        if work == "optimize":
                            charge[index] = trial.suggest_categorical(
                                str(index), valence
                            )
                        else:
                            charge[index] = trial[str(index)]
                    continue

                elif (
                    element_type
                    in ["transition_metal", "lanthanoid", "metalloid"]
                    and valence.size > 1
                ):
                    if work == "optimize":
                        charge[index] = trial.suggest_categorical(
                            str(index), valence
                        )
                    else:
                        charge[index] = trial[str(index)]
                    continue

                else:
                    if work == "optimize":
                        charge[index] = trial.suggest_categorical(
                            str(index), valence
                        )
                    else:
                        charge[index] = trial[str(index)]
                    continue

        if work == "optimize":
            if self["space_group"] != "P1":
                remaining_charge = np.sum(
                    charge * self["multiplicity"] * self["occ"]
                )
            else:
                remaining_charge = np.sum(charge * self["occ"])

            score = np.abs(remaining_charge)
            if score == 0.0:
                trial.study.stop()
            return score
        else:
            return charge

    if work == "optimize":
        return get_value
    else:
        return get_value


def get_atom_information(self) -> Dict:
    """
    Retrieves atomic information for the elements in the Phase.

    This function retrieves information about the elements present in the Phase
    object, such as element type, atomic number, and valence.

    Parameters
    ----------
    self : Phase
        A Phase object containing the atomic symbols.

    Returns
    -------
    dict
        A dictionary where keys are atomic symbols and values are dictionaries
        containing the atomic information.
    """
    atom_information = {}
    symbols = np.unique(self["symbol"])
    info = ai()

    for atom in symbols:        
        symbol, charge = info.parse_symbol(atom)
        atom_information[atom] = {}
        atom_information[atom]["charge"] = float(charge)
        atom_information[atom]["number"] = info.get_atomic_number(symbol=symbol)
        atom_information[atom]["element_type"] = info.get_element_type(symbol=symbol)
        valence_list = info.get_typical_valences(symbol=symbol)
        atom_information[atom]["valence"] = np.array(valence_list, dtype=float) if len(valence_list) != 0 else np.array([0.0])

    return atom_information

def get_atom_radius(self):
    """
    Assigns atomic radii to each atom in the Phase object.

    This function retrieves the atomic radii for each atom in the Phase object
    based on their symbols and charges.

    Parameters
    ----------
    self : Phase
        A Phase object containing the atomic symbols and charges.

    Returns
    -------
    Phase
        The modified Phase object with the 'radius' attribute set.
    """
    symbols = self["symbol"]
    charge = self["charge"]
    radius = np.zeros(symbols.size).astype(float)

    uni_sym, symbol_index = np.unique(symbols, return_index=True)
    uni_chr = charge[symbol_index]

    info = ai()
    for n, atom in enumerate(uni_sym):
        symbol, parsed_charge = info.parse_symbol(atom)
        chg = float(uni_chr[n]) if uni_chr[n] != 0 else float(parsed_charge)
        r = info.get_radius(symbol=symbol, charge=chg)
        radius[np.where(symbols == atom)] = r if r is not None else 1.0

    self["radius"] = radius
    return self

def get_coordination_number(self):
    """
    Retrieves the expected coordination number for each atom type.

    This function retrieves the typical coordination numbers for each atomic symbol
    present in the Phase object.

    Parameters
    ----------
    self : Phase
        A Phase object containing the atomic symbols.

    Returns
    -------
    Phase
        The modified Phase object with the 'cn' parameter and 'coordination_number' model attribute set.
    """
    symbols = np.asarray(self["symbol"])
    charges = np.asarray(self["charge"], dtype=float)
    info = ai()

    cn_lists = []
    expected_cns = np.empty(symbols.size, dtype=object)
    for idx, atom in enumerate(symbols):
        sym, parsed_chg = info.parse_symbol(atom)
        chg = charges[idx] if idx < charges.size and charges[idx] != 0 else parsed_chg
        cn_list = info.get_typical_coordination_number(symbol=sym, charge=chg)
        cn_lists.append(cn_list)
        expected_cns[idx] = str(cn_list)

    self.par["cn"]["value"] = cn_lists
    self.model["coordination_number"] = expected_cns
    return self

def get_material_type(self):
    """
    Determines the material type of the self.

    This function analyzes the atomic composition of the self to determine its
    material type (e.g., "oxide", "metal", "organic").

    Parameters
    ----------
    self : Phase
        A Phase object containing the atomic symbols.

    Returns
    -------
    Phase
        The modified Phase object with the 'material_type' attribute set.
    """
    element_types = []
    atom_numbers = []
    material_type = ""

    atom_information = get_atom_information(self)
    for atom, dic in atom_information.items():
        element_types += [dic["element_type"]]
        atom_numbers += [dic["number"]]

    if (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and (8 in atom_numbers):
        material_type = "oxide"

    elif (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and (7 in atom_numbers):
        material_type = "nitride"

    elif (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and (6 in atom_numbers):
        material_type = "carbide"

    elif (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and (5 in atom_numbers):
        material_type = "boride"

    elif (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and (15 in atom_numbers):
        material_type = "phosphide"

    elif (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and ("chalcogen" in element_types):
        material_type = "chalcogenide"

    elif (
        any(
            x in element_types
            for x in [
                "transition_metal",
                "lanthanoid",
                "alkali",
                "alkali_earth",
            ]
        )
    ) and ("halogen" in element_types):
        material_type = "halide"

    elif all(x in [1, 6, 7, 8, 9] for x in atom_numbers):
        material_type = "organic"

    elif all(
        x in ["transition_metal", "lanthanoid", "alkali", "alkali_earth"]
        for x in element_types
    ):
        material_type = "metal"

    elif any(x in [13, 31, 49, 50, 81, 82] for x in atom_numbers) | any(
        x in ["transition_metal", "lanthanoid", "alkali", "alkali_earth"]
        for x in element_types
    ):
        material_type = "metal"

    elif all(x in ["metalloid"] for x in element_types):
        material_type = "metalloid"

    self["material_type"] = material_type

    return self

def get_site_relationship(self, threshold: float = 1.4) -> Dict[str, NDArray[np.float64]]:
    """
    Analyzes the relationships between atomic sites in a crystal structure.

    This function calculates and returns information about the distances, charge
    relations, and coordination between atomic sites within a crystal structure
    defined by a Phase object.  It calculates the actual distance between each pair
    of sites, compares it to the expected distance (based on atomic radii),
    and determines the charge relationship.  It then uses this information to
    identify coordinating sites based on a distance threshold.

    Parameters
    ----------
    self : Phase
        A Phase object containing the structural information, including atomic
        coordinates ('xyz'), atomic radii ('radius'), and atomic charges ('charge').
    threshold : float, optional
        The distance ratio threshold for considering two sites as coordinating.
        The ratio is calculated as actual distance / expected distance.
        Default is 1.4.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "actual_distance": A NumPy array of pairwise distances between atoms.
        - "expected_distance": A NumPy array of expected pairwise distances
          (based on atomic radii).
        - "charge_relation": A NumPy array indicating the product of charges for each
          atom pair (positive for repulsion, negative for attraction, 0 for neutral).
        - "coordinating": A dictionary with keys "attractive" and "neutral".
          Each value is a NumPy array representing the coordination:
          1 indicates coordination, 0 indicates no coordination.

    """
    info = {}
    xyz = self.model["xyz"]
    g = self.model["lattice_vector"]
    xyz_cartesian = np.dot(g, xyz.T).T  # Convert to Cartesian coordinates

    # Calculate pairwise distances
    info["actual_distance"] = np.sqrt(
        np.sum(np.square(xyz_cartesian[:, np.newaxis] - xyz_cartesian[np.newaxis, :]), axis=-1)
    )

    radius = self.model["radius"]
    charge = self.model["charge"]
    info["expected_distance"] = radius[:, np.newaxis] + radius[np.newaxis, :]
    info["charge_relation"] = charge[:, np.newaxis] * charge[np.newaxis, :]

    distance_ratio = info["actual_distance"] / info["expected_distance"]

    # Identify coordinating sites based on distance and charge
    info["coordinating"] = {}
    info["coordinating"]["attractive"] = np.where(
        (info["charge_relation"] < 0)
        & (distance_ratio < threshold)
        & (info["actual_distance"] > 0.0),
        1.0,
        0.0,
    )
    info["coordinating"]["neutral"] = np.where(
        (info["charge_relation"] == 0)
        & (distance_ratio < threshold)
        & (info["actual_distance"] > 0.0),
        1.0,
        0.0,
    )
    return info

def get_bond_relationship(self, threshold: float = 1.4):
    self.get_atom_radius()
    self.get_formal_charge()
    self()
    site_relation_info = self.get_site_relationship(threshold=threshold)
    bond_table = site_relation_info["coordinating"]["attractive"] + site_relation_info["coordinating"]["neutral"]

    bonds_list = []
    for i in range(bond_table.shape[0]):
        row = bond_table[i, :]
        non_zero_indices = np.where(row != 0.0)[0].tolist()
        bonds_list.append(non_zero_indices)
    return bonds_list

