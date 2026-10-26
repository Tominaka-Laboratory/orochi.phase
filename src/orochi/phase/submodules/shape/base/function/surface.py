from typing import List, Union
from numpy.typing import NDArray
import numpy as np


def trim_under_coordinated_ions(self, min_occ: float = 0.01, cutoff: float = 1.4):
    """
    Trims under-coordinated ions from the Phase object.

    This function removes atoms from the model that are considered under-coordinated
    based on a coordination number criterion. It first calls `remove_dilute_atoms`
    to remove atoms with low occupancy.  Then, for each atom, it calculates its
    coordination number and removes cations and anions that are under-coordinated.

    Parameters
    ----------
    self 
        A Phase object containing the atomic coordinates, charges, and occupancies.
    min_occ : float, optional
        Minimum occupancy threshold for an atom to be considered present.
        Atoms with occupancy below this value are removed. Default is 0.01.

    Returns
    -------
    Phase
        The modified Phase object with under-coordinated ions removed.
    """

    remove_dilute_atoms(self, min_occ)
    relation = self.get_site_relationship(cutoff) 
    site_symbol = self.model["symbol"]
    site_charge = self.model["charge"]
    relation_coordinating = relation["coordinating"]["attractive"]
    
    expected_cns = np.zeros(site_symbol.size).astype(float)
    symbol = np.unique(site_symbol[np.where(site_charge > 0)])
    for atom in symbol:
        cn_str = self.model["coordination_number"][np.where(site_symbol == atom)]
        cn_list = eval(cn_str[0])
        cns = np.sum(relation_coordinating[np.where(site_symbol == atom)], axis=-1)
        cns = cns[cns != 0]

        unique_cns, counts = np.unique(cns, return_counts=True)
        order = np.argsort(counts)[::-1]
        unique_cns = unique_cns[order]
        for cn in unique_cns:
            if cn in cn_list:
                expected_cns[np.where(site_symbol == atom)] = cn
                break

    # trim under coordinated cations
    cations = np.where(site_charge > 0)
    site_cn = np.sum(relation_coordinating , axis=-1)
    under_coordinated = np.where(site_cn[cations] < expected_cns[cations], 0., 1.)
    self.model["occ"][cations] *= under_coordinated

    # trim isolated anions
    anions = np.where(site_charge < 0)
    relation_coordinating[cations] *= under_coordinated.reshape(-1, 1)
    site_cn = np.sum(relation_coordinating.T , axis=-1)
    self.model["occ"][anions] *= np.where(site_cn[anions] < 1., 0., 1.)

    remove_dilute_atoms(self, min_occ)
    return self



def remove_dilute_atoms(self, min_occ: float = 0.01):
    """
    Removes atoms with low occupancy from the Phase object.

    This function removes atoms whose occupancy is below a specified threshold.
    It iterates through the site-dependent attributes of the Phase object
    ('xyz', 'occ', 'adp', etc.) and removes the corresponding elements
    for atoms with low occupancy.

    Parameters
    ----------
    self 
        A Phase object containing the atomic occupancies and other atomic properties.
    min_occ : float, optional
        Minimum occupancy threshold. Atoms with occupancy below this value are removed.
        Default is 0.01.

    Returns
    -------
    Phase
        The modified Phase object with dilute atoms removed.
    """
    use = np.where(self.model["occ"] >= min_occ)
    for key in list(self._expandables) + ["coordination_number", "site_index"]:
        if key not in self.model:
            continue
        value = self.model[key]
        self.model[key] = value[use]
    return self



def surface_substitution(self, surface_function: np.ndarray) -> None:
    """
    Substitutes atomic symbols for surface atoms.

    This function substitutes the atomic symbols of atoms identified as surface atoms
    with new symbols, as specified in the Phase object's 'surface' substitution list.

    Parameters
    ----------
    self 
        A Phase object containing the atomic symbols and surface substitution information.
    surface_function : numpy.ndarray
        A NumPy array representing the surface function.  Values of 1 indicate surface
        atoms, while 0 indicates bulk atoms.
    """
    atoms = np.array(self["surface", "substitute"]).reshape(-1, 2) #changed local variable name to atoms
    for old, new in atoms:
        select = np.where(
            (surface_function == 1) & (self.model["symbol"] == old)
        )
        symbols = np.copy(self.model["symbol"])
        symbols[select] = new
        self.model["symbol"] = symbols
