from ..database.atoms import AtomInformation as ai
from ..database.f0_waasmaier_kirfel import f0_function
from ..database.f1f2_brennan_cowan import f1f2_function
from ..database.compton_cromer_mann_ana import compton_function


def get_atom_info_dicionary(
    self, symbols="all", key="radius"
):  # key=[radius, electron, charge, f0, f1f2, compton
    """
    Retrieves a dictionary of atomic information for specified symbols.

    This method fetches atomic properties (e.g., radius, number of electrons)
    for a given set of atomic symbols from the internal database.

    Parameters
    ----------
    symbols : str or list of str, optional
        If "all", information for all unique symbols in the phase is returned.
        Otherwise, a list of specific atomic symbols. Defaults to "all".
    key : str, optional
        The atomic property to retrieve (e.g., "radius", "electron", "charge",
        "f0", "f1f2", "compton"). Defaults to "radius".

    Returns
    -------
    dict
        A dictionary where keys are atomic symbols and values are the
        requested atomic property.
    """
    if isinstance(symbols, str):
        if symbols == "all":
            symbols = set(self.model["symbol"])

    dic = {}
    info = ai()
    for atom in symbols:        
        symbol, charge = info.parse_symbol(atom)

        if key == "radius":
            dic[atom] = info.get_radius(symbol=symbol, charge=charge)
        elif key == "electron":
            dic[atom] = info.get_atomic_number(symbol=symbol) - charge

        elif key == "charge":
            dic[atom] = charge

        elif key == "f0":
            dic[atom] = f0_function(symbol)

        elif key == "f1f2":
            dic[atom] = f1f2_function(symbol)

        elif key == "compton":
            dic[atom] = compton_function(symbol)

    return dic


def atom_info_dicionary(symbols, key="radius"):
    """Standalone helper for retrieving atomic properties by symbol list."""
    return get_atom_info_dicionary(None, symbols=symbols, key=key)


atom_info_dictionary = atom_info_dicionary

