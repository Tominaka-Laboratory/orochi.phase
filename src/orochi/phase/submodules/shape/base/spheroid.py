from typing import List, Union
from numpy.typing import NDArray
import numpy as np
from .function.surface import trim_under_coordinated_ions, surface_substitution
from .function.expansion import expand_structure

def make(self, min_occ: float = 0.01):
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

    if np.all(np.asarray(self["charge"], dtype=float) == 0.0):
        self.get_formal_charge()
    tvs_car = calc_translation(self)
    expand_structure(self, tvs_car)
    apply_shape_function(self)
    trim_under_coordinated_ions(self, min_occ)

    if len(self["surface", "substitute"]) != 0:
        surface_function = get_surface_function(self)
        surface_substitution(self, surface_function)

    return self


def calc_translation(self) -> NDArray[np.float64]:
    """
    Calculates the translational vectors for structure expansion.

    Parameters
    ----------
    self
        A Phase object containing the lattice information and shape parameters.

    Returns
    -------
    NDArray[np.float64]
        A NumPy array of translational vectors in Cartesian coordinates.
    """
    # define vectors in uc
    xyz = np.array([1, 1, 1])
    xyz_car = np.dot(self.model["lattice_vector"].T, xyz.T).T

    radii = np.asarray(self["sphere"], dtype=float) / 2.0

    mag = (
        np.abs(np.floor(radii / xyz_car).astype(int)) + 1
    )  # including spheroid
    x, y, z = mag * self["pbc"]

    # find mesh grid points
    x, y, z = np.meshgrid(
        np.arange(-x, x + 1), np.arange(-y, y + 1), np.arange(-z, z + 1)
    )
    translational_vectors = np.concatenate(
        [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=-1
    ).astype(float)
    translational_vectors_car = np.dot(
        self.model["lattice_vector"].T, translational_vectors.T
    ).T.reshape(-1, 3)

    extension = np.sum(self.model["lattice"][:3])
    spheroid = np.sum(
        np.power(translational_vectors_car / (radii + extension), 2.0),
        axis=-1,
    )
    use = np.where(spheroid <= 1.0)  # including spheroid
    translational_vectors_car = np.round(translational_vectors_car[use], 7)

    return translational_vectors_car



def apply_shape_function(self) -> None:
    """
    Applies a shape function to the atomic occupancies to define the nanoparticle shape.

    Parameters
    ----------
    self
        A Phase object containing the atomic coordinates, shape parameters,
        and lattice information.
    """
    origin = np.asarray(self["sphere", "origin"], dtype=float).reshape(-1, 3)
    origin = np.dot(self["lattice_vector"], origin.T).T

    xyz = self.model["xyz"] - origin
    radii = np.asarray(self["sphere"], dtype=float) / 2.0
    std = np.asarray(self["sphere", "std"], dtype=float) / 2.0

    if np.any(std > 0.0):
        from scipy.special import erf
        r_std = std * (2.0**0.5)
        r_std[r_std < 0.0001] = 0.0001
        shape_function = np.prod(
            (1.0 - erf((np.abs(xyz) - radii) / r_std)) / 2.0, axis=-1
        )

    else:
        x = np.power(xyz / radii, 2.0)  # including spheroid
        x = np.sum(x, axis=-1) - 1.0
        shape_function = 1.0 - np.heaviside(x, 0.0)

    self.model["occ"] *= shape_function



def get_surface_function(self) -> np.ndarray:
    """
    Calculates a surface function to identify surface atoms.

    Parameters
    ----------
    self
        A Phase object containing the atomic coordinates and shape parameters.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the surface function. Values of 1 indicate surface
        atoms, while 0 indicates bulk atoms.
    """
    origin = np.asarray(self["sphere", "origin"], dtype=float).reshape(-1, 3)
    origin = np.dot(self["lattice_vector"], origin.T).T

    xyz = self.model["xyz"] - origin
    radii = np.asarray(self["sphere"], dtype=float) / 2.0

    x = xyz / np.maximum(radii - self["surface"], 1e-6)
    x = np.power(x, 2.0)
    x = np.sum(x, axis=-1) - 1.0
    surface_function = np.heaviside(x, 1.0).astype(int)

    return surface_function
