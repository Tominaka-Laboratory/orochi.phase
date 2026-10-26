import numpy as np


def coordination(
    change=[0.1, 0.1, 0.1],
    fix_elements=[],
    fix_site_xyz=[[0, 0], [0, 1], [0, 2]],
    fix_coordinate_values=[0.0, 1 / 4, 1 / 3, 1 / 2],
    ):
    
    def restraint(phase):
        _xyz_exclusion(phase, fix_elements, fix_site_xyz, fix_coordinate_values)
        _xyz_change_in_coordinates(phase, change)
    return restraint


def angstrome(
    change=[0.1, 0.1, 0.1],
    fix_elements=[],
    fix_site_xyz=[[0, 0], [0, 1], [0, 2]],
    fix_coordinate_values=[0.0, 1 / 4, 1 / 3, 1 / 2],
    ):

    def restraint(phase):
        _xyz_exclusion(phase, fix_elements, fix_site_xyz, fix_coordinate_values)
        _xyz_change_in_angstrome(phase, change)
    return restraint


def _xyz_exclusion(phase, fix_elements, fix_site_xyz, fix_coordinate_values):
    xyz = phase["xyz"]
    symbol = phase["symbol"]
    opt_bool = np.full_like(xyz, True)

    for element in fix_elements:
        opt_bool *= np.where(symbol == element, False, True)

    for c, r in fix_site_xyz:
        opt_bool[c, r] = False

    for c in fix_coordinate_values:
        xyz_2 = 0.5 - xyz
        xyz_2 = np.abs(xyz_2 - np.floor(xyz_2) - 0.5)
        opt_bool *= np.where(np.abs(xyz_2 - c) < 0.001, False, True)

    phase["xyz", "fit"] = opt_bool
    return phase


def _xyz_change_in_coordinates(phase, change):
    xyz = phase["xyz"]
    change = np.tile(change, (np.shape(xyz)[0], 1))
    phase["xyz", "min"] = xyz - change

    r = np.where(phase["xyz", "min"][:, 2] < 0.0)
    min = phase["xyz", "min"]
    min[r][:, 2] = 0.0
    phase["xyz", "min"] = min

    phase["xyz", "max"] = xyz + change
    r = np.where(phase["xyz", "max"][:, 2] > 1.0)
    max = phase["xyz", "max"]
    max[r][:, 2] = 0.0
    phase["xyz", "max"] = max

    return phase


def _xyz_change_in_angstrome(phase, change):
    xyz = phase["xyz"]
    lattice = phase["lattice"]

    change = change / lattice[:3]
    change = np.tile(change, (np.shape(xyz)[0], 1))

    phase["xyz", "min"] = xyz - change
    min = phase["xyz", "min"]
    for n, pbc in enumerate(phase["pbc"]):
        if pbc:
            continue
        else:
            m = np.where(min[:, n] < 0.0)
            min[m][:, n] = 0.0
    phase["xyz", "min"] = min

    phase["xyz", "max"] = xyz + change
    max = phase["xyz", "max"]
    for n, pbc in enumerate(phase["pbc"]):
        if pbc:
            continue
        else:
            m = np.where(min[:, n] > 0.0)
            max[m][:, n] = 1.0
    phase["xyz", "max"] = max
    return phase