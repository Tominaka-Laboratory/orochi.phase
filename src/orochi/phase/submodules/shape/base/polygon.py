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
    structural model suitable for PDF simulations.

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


def apply_lattice_symmetry(self):
    """
    Applies lattice symmetry to define the shape of the nanoparticle.

    This function defines the shape of the nanoparticle based on the lattice symmetry
    of the material. It sets the direction and origin of the shape, and also sets the
    value, min, max, and error for the shape parameters ('polygon').

    Parameters
    ----------
    self 
        A Phase object containing the lattice information and shape parameters.

    Returns
    -------
    Phase
        The modified Phase object with updated shape parameters.
    """
    def orthogonal_shape() -> NDArray[np.float64]:
        """
        Defines the direction vectors for an orthogonal shape (e.g., cube).
        """
        direction = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        return direction

    def diagonal_shape() -> NDArray[np.float64]:
        """
        Defines the direction vectors for a diagonal shape (e.g., octahedron).
        """
        direction = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [-1, 1, 1],
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [1, -1, -1],
            ]
        )
        return direction

    def along_c_shape() -> NDArray[np.float64]:
        """
        Defines the direction vectors for a shape elongated along the c-axis.
        """
        direction = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        return direction

    if self["shape_type"] == "cube":
        direction = orthogonal_shape()
        self["polygon", "direction"] = direction
        self["polygon", "origin"] = np.zeros_like(direction)

        length = self["polygon"]
        if isinstance(length, float):
            length = length / 2.0
        else:
            length = length[0] / 2.0

        self["polygon", "value"] = np.full(direction.shape[0], length)
        self["polygon", "min"] = np.full(
            direction.shape[0], length / 3.0
        )
        self["polygon", "max"] = np.full(
            direction.shape[0], length * 3.0
        )

        std = self["polygon", "std"]
        if isinstance(std, float):
            std = std / 2.0
        else:
            std = std[0] / 2.0
        self["polygon", "std"] = np.full(direction.shape[0], std)

    elif self["shape_type"] == "octahedron":
        direction = diagonal_shape()
        self["polygon", "direction"] = direction
        self["polygon", "origin"] = np.zeros_like(direction)

        length = self["polygon"]
        if isinstance(length, float):
            length = length / 2.0
        else:
            length = length[0] / 2.0

        self["polygon", "value"] = np.full(direction.shape[0], length)
        self["polygon", "min"] = np.full(
            direction.shape[0], length / 3.0
        )
        self["polygon", "max"] = np.full(
            direction.shape[0], length * 3.0
        )

        std = self["polygon", "std"]
        if isinstance(std, float):
            std = std / 2.0
        else:
            std = std[0] / 2.0

        self["polygon", "std"] = np.full(direction.shape[0], std)

    elif self["shape_type"] == "cuboid":
        direction = orthogonal_shape()
        self["polygon", "direction"] = direction
        self["polygon", "origin"] = np.zeros_like(direction)

        length = self["polygon"][:3] / 2.0
        length = np.append(length, length)

        self["polygon", "value"] = length
        self["polygon", "min"] = length / 3.0
        self["polygon", "max"] = length * 3.0

        std = np.asarray(self["polygon", "std"], dtype=float)[:3] / 2.0
        std = np.append(std, std)

        self["polygon", "std"] = std

    elif self["shape_type"] == "along_c":
        direction = along_c_shape()
        self["polygon", "direction"] = direction
        self["polygon", "origin"] = np.zeros_like(direction)

        length = self["polygon"][:3] / 2.0
        length = np.append(length, length)

        self["polygon", "value"] = length
        self["polygon", "min"] = length / 3.0
        self["polygon", "max"] = length * 3.0

        std = np.asarray(self["polygon", "std"], dtype=float)[:3] / 2.0
        std = np.append(std, std)

        self["polygon", "std"] = std

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
    extension = np.sum(self.model["lattice"][:3])
    length = np.array(self["polygon"]).flatten()
    origin = np.asarray(self["polygon", "origin"], dtype=float).reshape(-1, 3)
    origin = np.dot(self["lattice_vector"], origin.T).T

    direction = np.array(self["polygon", "direction"]).reshape(-1, 3)
    direction = (
        direction.T / np.linalg.norm(direction, axis=-1)
    ).T  # unit vectors
    direction -= origin

    # vectors to grid points
    vectors = direction * length[:, np.newaxis]
    vectors /= self.model["lattice"][:3]
    vectors = np.floor(vectors).astype(int) + np.sign(vectors)

    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    px = np.sum(x[np.where(x >= 0)])
    py = np.sum(y[np.where(y >= 0)])
    pz = np.sum(z[np.where(z >= 0)])

    nx = np.sum(x[np.where(x <= 0)])
    ny = np.sum(y[np.where(y <= 0)])
    nz = np.sum(z[np.where(z <= 0)])

    positive = np.array([px, py, pz])
    negative = np.array([nx, ny, nz])

    max_len = float(np.atleast_1d(self["polygon", "max_length"])[0])
    vectors = (
        np.full(3, max_len)
        / self.model["lattice"][:3]
    )
    vectors = np.abs(np.floor(vectors).astype(int) + 1)

    positive[positive == 0] = vectors[positive == 0]
    negative[negative == 0] = -vectors[negative == 0]
    positive *= self.model["pbc"]
    negative *= self.model["pbc"]

    # find mesh grid points
    x = np.arange(negative[0], positive[0] + 1, 1)
    y = np.arange(negative[1], positive[1] + 1, 1)
    z = np.arange(negative[2], positive[2] + 1, 1)
    x, y, z = np.meshgrid(x, y, z)

    xyz = np.concatenate(
        [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=-1
    ).astype(float)
    translational_vectors_car = np.dot(
        self.model["lattice_vector"].T, xyz.T
    ).T
    origin = np.dot(self.model["lattice_vector"].T, origin.T).T
    vectors = (
        translational_vectors_car[np.newaxis, :, :] - origin[:, np.newaxis, :]
    )

    x = np.sum(vectors * direction[:, np.newaxis, :], axis=-1)
    x = x - (length + extension)[:, np.newaxis]
    x = 1.0 - np.heaviside(x, 0.0)
    x = np.prod(x, axis=0)
    use = np.where(x > 0.0)

    translational_vectors_car = np.round(translational_vectors_car[use], 7)

    return translational_vectors_car


def apply_shape_function(self) -> None:
    """
    Applies a shape function to the atomic occupancies to define the nanoparticle shape.

    Parameters
    ----------
    self
        A Phase object containing the atomic coordinates, shape parameters (polygon),
        and lattice information.
    """
    length = np.array(self["polygon"]).flatten()
    origin = np.asarray(self["polygon", "origin"], dtype=float).reshape(-1, 3)
    origin = np.dot(self["lattice_vector"], origin.T).T

    xyz = self.model["xyz"][np.newaxis, :, :] - origin[:, np.newaxis, :]

    direction = np.array(self["polygon", "direction"]).reshape(-1, 3)
    direction = (
        direction.T / np.linalg.norm(direction, axis=-1)
    ).T  # unit vectors

    std = np.array(self["polygon", "std"]).flatten()
    y = np.sum(xyz * direction[:, np.newaxis, :], axis=-1)
    y = y - length[:, np.newaxis]

    if np.sum(std) == 0.0:
        y = 1.0 - np.heaviside(y, 0.0)

    else:
        from scipy.special import erf
        r_std = std * (2.0**0.5)
        y = (1.0 - erf(y / r_std[:, np.newaxis])) / 2.0

    shape_function = np.prod(y, axis=0).flatten()
    self.model["occ"] *= shape_function



def get_surface_function(self) -> np.ndarray:
    """
    Calculates a surface function to identify surface atoms.

    Parameters
    ----------
    self
        A Phase object containing the atomic coordinates, shape parameters (polygon),
        and lattice information.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the surface function. Values of 1 indicate surface
        atoms, while 0 indicates bulk atoms.
    """
    length = np.array(self["polygon"]).flatten()
    origin = np.asarray(self["polygon", "origin"], dtype=float).reshape(-1, 3)
    origin = np.dot(self["lattice_vector"], origin.T).T

    xyz = self.model["xyz"][np.newaxis, :, :] - origin[:, np.newaxis, :]

    direction = np.array(self["polygon", "direction"]).reshape(-1, 3)
    direction = (
        direction.T / np.linalg.norm(direction, axis=-1)
    ).T  # unit vectors

    y = np.sum(xyz * direction[:, np.newaxis, :], axis=-1)
    y = (
        y
        - np.array((self["polygon"] - self["surface"])).flatten()[
            :, np.newaxis
        ]
    )
    y = 1 - np.heaviside(y, 1.0).astype(int)
    surface_function = np.prod(y, axis=0).flatten()
    surface_function = np.abs(1 - surface_function)

    return surface_function
