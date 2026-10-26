import numpy as np
from ...database.atoms import AtomInformation as ai



def lattice_density_constraints(density, density_gap_ratio):
    def function(orochi_module, key):
        if key == "lattice":
            return lattice_constraint_within_width(
                orochi_module, key, density, density_gap_ratio
            )
        else:
            return None
    return function


def lattice_constraint_within_width(
    orochi_module, key, density, difference_ratio
):
    class calc_volume:
        volume = get_lattice_volume_at_density(
            orochi_module, density
        )

    c = calc_volume()
    volume = c.volume
    volume_min = volume * (1 - difference_ratio)
    volume_max = volume * (1 + difference_ratio)

    lattice = orochi_module.par["lattice"]
    lattice_min = orochi_module.par.par["lattice"]["min"]
    lattice_max = orochi_module.par.par["lattice"]["max"]
    lattice_system = orochi_module.par.par["lattice"]["system"]

    a = lattice[0]
    b = lattice[1]
    c = lattice[2]
    alpha = np.radians(lattice[3])
    beta = np.radians(lattice[4])
    gamma = np.radians(lattice[5])

    a_min = lattice_min[0]
    b_min = lattice_min[1]
    c_min = lattice_min[2]
    a_max = lattice_max[0]
    b_max = lattice_max[1]
    c_max = lattice_max[2]

    cos_alpha_star = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (
        np.sin(beta) * np.sin(gamma)
    )
    sin_alpha_star = (1 - cos_alpha_star**2.0) ** 0.5

    cos_beta_star = (np.cos(gamma) * np.cos(alpha) - np.cos(beta)) / (
        np.sin(gamma) * np.sin(alpha)
    )
    sin_beta_star = (1 - cos_beta_star**2.0) ** 0.5

    cos_gamma_star = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (
        np.sin(gamma) * np.sin(beta)
    )
    sin_gamma_star = (1 - cos_gamma_star**2.0) ** 0.5

    # a_star = b * c * np.sin(alpha)/volume
    # b_star = c * a * np.sin(beta)/volume
    # c_star = a * b * np.sin(gamma)/volume

    # a = b_star * c_star * sin_alpha_star * volume
    # b = c_star * a_star * sin_beta_star * volume
    # c = a_star * b_star * sin_gamma_star * volume

    # volume = abc(np.sin(alpha)**2. +np.sin(beta)**2 + np.sin(gamma)**2 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) -2)**0.5
    # volume = abc sin_alpha_star sin_beta sin_gamma

    def local1(id):
        def return_min_max(min, max):
            min_in = orochi_module.par[key, "min"][id]
            max_in = orochi_module.par[key, "max"][id]
            min_in, max_in = np.amin([min_in, max_in]), np.amax(
                [min_in, max_in]
            )

            if (max < min_in) or (min > max_in):
                return min, max

            else:
                min = np.amax([min, min_in])
                max = np.amin([max, max_in])
                return min, max

        if lattice_system in ["monoclinic", "triclinic", "orthorhombic"]:
            if id == 0:
                min = volume_min / (
                    b * c * sin_alpha_star * np.sin(beta) * np.sin(gamma)
                )
                max = volume_max / (
                    b * c * sin_alpha_star * np.sin(beta) * np.sin(gamma)
                )
                return return_min_max(min, max)

            elif id == 1:
                min = volume_min / (
                    a_max * c * sin_alpha_star * np.sin(beta) * np.sin(gamma)
                )
                max = volume_max / (
                    a_min * c * sin_alpha_star * np.sin(beta) * np.sin(gamma)
                )
                return return_min_max(min, max)

            elif id == 2:
                min = volume_min / (
                    a_max
                    * b_max
                    * sin_alpha_star
                    * np.sin(beta)
                    * np.sin(gamma)
                )
                max = volume_max / (
                    a_min
                    * b_min
                    * sin_alpha_star
                    * np.sin(beta)
                    * np.sin(gamma)
                )
                return return_min_max(min, max)

            elif id > 2:
                min = orochi_module.par[key, "min"][id]
                max = orochi_module.par[key, "max"][id]
                return min, max

        elif lattice_system in ["cubic"]:
            if id == 0:
                min = volume_min ** (1 / 3)
                max = volume_max ** (1 / 3)
                return return_min_max(min, max)

            elif id == 1:
                min = volume_min ** (1 / 3)
                max = volume_max ** (1 / 3)
                return return_min_max(min, max)

            elif id == 2:
                min = volume_min ** (1 / 3)
                max = volume_max ** (1 / 3)
                return return_min_max(min, max)

            elif id > 2:
                min = 90
                max = 90
                return min, max

        elif lattice_system in ["tetragonal"]:
            if id == 0:
                min = (volume_min / c) ** (1 / 2)
                max = (volume_max / c) ** (1 / 2)
                return return_min_max(min, max)

            elif id == 1:
                min = (volume_min / c) ** (1 / 2)
                max = (volume_max / c) ** (1 / 2)
                return return_min_max(min, max)

            elif id == 2:
                min = volume_min / (a_max * a_max)
                max = volume_max / (a_min * a_min)
                return return_min_max(min, max)

            elif id > 2:
                min = 90
                max = 90
                return min, max

        elif lattice_system in ["hexagonal"]:
            if id == 0:
                min = (volume_min / (c * sin_alpha_star * np.sin(gamma))) ** (
                    1 / 2
                )
                max = (volume_max / (c * sin_alpha_star * np.sin(gamma))) ** (
                    1 / 2
                )
                return return_min_max(min, max)

            elif id == 1:
                min = (volume_min / (c * sin_alpha_star * np.sin(gamma))) ** (
                    1 / 2
                )
                max = (volume_max / (c * sin_alpha_star * np.sin(gamma))) ** (
                    1 / 2
                )
                return return_min_max(min, max)

            elif id == 2:
                min = volume_min / (
                    a_max * a_max * sin_alpha_star * np.sin(gamma)
                )
                max = volume_max / (
                    a_min * a_min * sin_alpha_star * np.sin(gamma)
                )
                return return_min_max(min, max)

            elif id in [3, 4]:
                min = 90
                max = 90
                return min, max

            elif id == 5:
                min = 120
                max = 120
                return min, max

        if lattice_system in ["trigonal"]:
            if id == 0:
                min = (
                    volume_min
                    / (sin_alpha_star * np.sin(beta) * np.sin(gamma))
                ) ** (1 / 3)
                max = (
                    volume_max
                    / (sin_alpha_star * np.sin(beta) * np.sin(gamma))
                ) ** (1 / 3)
                return return_min_max(min, max)

            elif id == 1:
                min = (
                    volume_min
                    / (sin_alpha_star * np.sin(beta) * np.sin(gamma))
                ) ** (1 / 3)
                max = (
                    volume_max
                    / (sin_alpha_star * np.sin(beta) * np.sin(gamma))
                ) ** (1 / 3)
                return return_min_max(min, max)

            elif id == 2:
                min = (
                    volume_min
                    / (sin_alpha_star * np.sin(beta) * np.sin(gamma))
                ) ** (1 / 3)
                max = (
                    volume_max
                    / (sin_alpha_star * np.sin(beta) * np.sin(gamma))
                ) ** (1 / 3)
                return return_min_max(min, max)

            elif id in [3, 4]:
                min = gamma
                max = gamma
                return min, max

            elif id == 5:
                min = orochi_module.par[key, "min"][id]
                max = orochi_module.par[key, "max"][id]
                return min, max

    return local1


def get_lattice_volume_at_density(phase, density):
    symbols = phase.model["symbol"]
    atoms = np.unique(symbol)
    mass_numbers = np.ones(np.size(symbol)).astype(float)
    info = ai()

    for atom in atoms:
        symbol, charge = info.parse_symbol(atom)
        mass = info.get_atomic_mass(symbol=symbol)
        mass_numbers[np.where(symbols == atom)] = mass

    mass = np.sum(mass_numbers * phase.model["occ"])  # g/mol unit cell
    mass /= 6.0221408 * 10**23  # avogadro number --> g/unit cell

    volume = mass / density  # cm3/ unit cell
    volume *= 10**24  # angstrome^3
    return volume
