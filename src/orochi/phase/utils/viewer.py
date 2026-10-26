import numpy as np
import re


def view_raw(phase):
    try:
        from ase.visualize import view
        from ase import Atoms
    except ImportError as e:
        raise ImportError("ase is required to use view_raw. Please install it via 'pip install ase'.") from e
    symbols = phase["symbol"]
    symbols = np.array(symbols).reshape(-1)
    symbols = [re.sub("[0-9,\-,+]", "", s) for s in symbols]
    positions = phase["xyz"]
    lattice = phase["lattice"]
    atoms = Atoms(
        symbols=symbols, scaled_positions=positions, cell=lattice, pbc=True
    )
    view(atoms)


def view_lattice(phase):
    try:
        from ase.visualize import view
        from ase import Atoms
    except ImportError as e:
        raise ImportError("ase is required to use view_lattice. Please install it via 'pip install ase'.") from e
    symbols = phase.lattice_sites["symbol"]
    symbols = np.array(symbols).reshape(-1)
    symbols = [re.sub("[0-9,\-,+]", "", s) for s in symbols]
    positions = phase.lattice_sites["xyz"]
    lattice = phase["lattice"]
    atoms = Atoms(
        symbols=symbols, scaled_positions=positions, cell=lattice, pbc=True
    )
    view(atoms)
