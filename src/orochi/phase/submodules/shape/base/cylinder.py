from typing import List, Union
from numpy.typing import NDArray
import numpy as np
from .function.surface import trim_under_coordinated_ions, surface_substitution
from .function.expansion import expand_structure

def make(self, min_occ: float = 0.01) :
    """
    Generates an explicit structural model for PDF calculation.

    This function takes a Phase object, calculates translational vectors, expands the structure,
    applies a shape function, and trims under-coordinated ions to create an explicit
    structural model suitable for materials modeling.

    Parameters
    ----------
    self 
        A Phase object containing the initial structural information.
    min_occ : float, optional
        Minimum occupancy threshold for trimming ions. Ions with occupancy below this
        value are removed. Default is 0.01.

    Returns
    -------
    Phase
        A Phase object representing the explicit structural model.
    """
    self.get_formal_charge()
    tvs_car = calc_translation(self)
    expand_structure(self, tvs_car)
    apply_shape_function(self)
    trim_under_coordinated_ions(self, min_occ)

    if len(self["surface", "substitute"]) != 0:
        surface_function = get_surface_function(self)
        surface_substitution(surface_function)

    return self


def calc_translation(self) -> NDArray[np.float64]:
    """
    Calculates the translational vectors for structure expansion (Cylinder).

    This function calculates the translational vectors needed to expand the unit cell
    structure to the desired nanoparticle size, specifically for a cylindrical shape.
    It uses the lattice vectors, cylinder dimensions, and origin/direction from the
    Phase object.

    Parameters
    ----------
    self 
        A Phase object containing the lattice information, cylinder dimensions,
        origin, and direction.

    Returns
    -------
    NDArray[np.float64]
        A NumPy array of translational vectors in Cartesian coordinates.
    """
    length = np.array(self["cylinder"]).flatten()
    origin = self["cylinder", "origin"].reshape(-1, 3)
    origin = np.dot(self["lattice", "vector"], origin.T).T

    direction = np.array(self["cylinder", "direction"]).reshape(-1, 3)
    direction = (
        direction.T / np.linalg.norm(direction, axis=-1)
    ).T  # unit vectors
    direction -= origin
    direction = np.dot(
        self.model["lattice_vector"].T, direction.flatten()
    )
    direction /= np.linalg.norm(direction)
    direction = direction.flatten()

    norm_vec = np.zeros(3)
    max_id = np.argmax(np.abs(direction))
    index = np.ones(3)
    index[max_id] = 0.0
    other_id = np.arange(3)[index.astype(bool)]
    norm_vec[other_id[0]] = 1.0

    if direction[other_id[1]] != 0:
        norm_vec[other_id[1]] = direction[other_id[0]] / direction[other_id[1]]
    else:
        norm_vec[other_id[1]] = 0

    norm_vec /= np.linalg.norm(norm_vec)
    norm_vec *= length[1] * 2.0 / (3**0.5)
    diag_vec = direction * length[0] - norm_vec

    norm_vec2 = np.cross(norm_vec, direction)
    norm_vec2 /= np.linalg.norm(norm_vec2)
    norm_vec2 *= length[1] * 2.0 / (3**0.5)
    diag_vec2 = direction * length[0] - norm_vec2

    norm_vec /= self.model["lattice"][:3]
    x1, y1, z1 = (
        np.abs(np.floor(norm_vec).astype(int) + 1)
        * np.sign(norm_vec)
        * self["pbc"]
    )
    diag_vec /= self.model["lattice"][:3]
    x2, y2, z2 = (
        np.abs(np.floor(diag_vec).astype(int) + 1)
        * np.sign(diag_vec)
        * self["pbc"]
    )

    norm_vec2 /= self.model["lattice"][:3]
    x3, y3, z3 = (
        np.abs(np.floor(norm_vec2).astype(int) + 1)
        * np.sign(norm_vec2)
        * self["pbc"]
    )
    diag_vec2 /= self.model["lattice"][:3]
    x4, y4, z4 = (
        np.abs(np.floor(diag_vec2).astype(int) + 1)
        * np.sign(diag_vec2)
        * self["pbc"]
    )

    x = np.arange(np.amin([x1, x2, x3, x4]), np.amax([x1, x2, x3, x4]) + 1, 1)
    y = np.arange(np.amin([y1, y2, y3, y4]), np.amax([y1, y2, y3, y4]) + 1, 1)
    z = np.arange(np.amin([z1, z2, z3, z4]), np.amax([z1, z2, z3, z4]) + 1, 1)
    x, y, z = np.meshgrid(x, y, z)

    xyz = np.concatenate(
        [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=-1
    ).astype(float)
    translational_vectors_car = np.dot(
        self.model["lattice_vector"].T, xyz.T
    ).T

    return translational_vectors_car


def apply_shape_function(self) -> None:
    """
    Applies a shape function to the atomic occupancies to define the nanoparticle shape (Cylinder).

    This function modifies the atomic occupancies within the Phase object to represent a cylindrical
    shape. Atoms outside the cylinder are assigned zero occupancy.

    Parameters
    ----------
    self 
        A Phase object containing the atomic coordinates, cylinder parameters,
        and lattice information.
    """
    length = np.array(self["cylinder"]).flatten()
    origin = self["cylinder", "origin"].reshape(-1, 3)
    origin = np.dot(self["lattice", "vector"], origin.T).T

    xyz = self.model["xyz"] - origin.flatten()

    direction = np.array(self["cylinder", "direction"]).flatten()
    direction = np.dot(self["lattice", "vector"], direction)
    direction /= np.linalg.norm(direction)  # unit vectors
    direction = direction.flatten()

    length_along_direction = np.dot(xyz, direction)
    along_direction = length_along_direction[:, np.newaxis] * direction
    distance_to_direction_vec = np.linalg.norm(xyz - along_direction, axis=-1)

    std = np.array(self["cylinder", "error"]).flatten()

    if np.sum(std) == 0.0:
        y = 1.0 - np.heaviside(length_along_direction - length[0], 0.0)
        y *= np.heaviside(length_along_direction, 0.0)
        y *= 1.0 - np.heaviside(distance_to_direction_vec - length[1], 0.0)

    else:
        r_std = std * (2.0**0.5)
        y = (
            1.0
            - np.erf((length_along_direction - length[0]) / r_std[0], 0.0)
            / 2.0
        )
        y *= np.erf(length_along_direction / r_std[0], 0.0) / 2.0
        y *= (
            1.0
            - np.heaviside(
                (distance_to_direction_vec - length[1]) / r_std[1], 0.0
            )
            / 2.0
        )

    shape_function = np.round(y, 7)
    self.model["occ"] *= shape_function



def get_surface_function(self) -> np.ndarray:
    """
    Calculates a surface function to identify surface atoms (Cylinder).

    This function calculates a function that identifies atoms located at the surface
    of the cylindrical nanoparticle, based on a defined surface layer thickness.

    Parameters
    ----------
    self 
        A Phase object containing the atomic coordinates, cylinder parameters,
        and lattice information.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the surface function. Values of 1 indicate surface
        atoms, while 0 indicates bulk atoms.
    """
    length = np.array(self["cylinder"]).flatten()
    origin = self["cylinder", "origin"].reshape(-1, 3)
    origin = np.dot(self["lattice", "vector"], origin.T).T

    xyz = self.model["xyz"] - origin.flatten()

    direction = np.array(self["cylinder", "direction"]).flatten()
    direction = np.dot(self["lattice", "vector"], direction)
    direction /= np.linalg.norm(direction)  # unit vectors
    direction = direction.flatten()

    length_along_direction = np.dot(xyz, direction)
    along_direction = length_along_direction[:, np.newaxis] * direction
    distance_to_direction_vec = np.linalg.norm(xyz - along_direction, axis=-1)

    y = 1.0 - np.heaviside(
        length_along_direction - (length[0] - self["surface"][0]), 0.0
    )
    y *= np.heaviside(length_along_direction + self["surface"][0], 0.0)
    y *= 1.0 - np.heaviside(
        distance_to_direction_vec - (length[1] - self["surface"][1]), 0.0
    )
    surface_function = np.abs(1 - y).astype(int)

    return surface_function
