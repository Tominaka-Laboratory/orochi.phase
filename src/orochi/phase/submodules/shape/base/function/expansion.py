from typing import List
from numpy.typing import NDArray
import numpy as np

def expand_structure(self, translational_vectors_car: NDArray[np.float64]):
    """
    Expands the structural model based on translational vectors.

    This function expands the atomic coordinates and other structural properties
    within the Phase object by applying the provided translational vectors.
    This effectively creates a larger structure, such as a nanoparticle, from the
    original unit cell.

    Parameters
    ----------
    self : Phase
        A Phase object containing the initial structural information, including
        atomic coordinates ('xyz') and other atomic properties in the model.
    translational_vectors_car : NDArray[np.float64]
        A NumPy array of translational vectors in Cartesian coordinates,
        representing the displacements needed to generate the expanded structure.
        Shape is (n_translations, 3).

    Returns
    -------
    Phase
        The modified Phase object with expanded atomic coordinates and other
        structural properties.  The lattice and periodic boundary conditions are
        also reset to represent a non-periodic nanoparticle.
    """
    p = translational_vectors_car.shape[0]

    for key in list(self._expandables) + ["coordination_number", "site_index"]:
        if key not in self.model:
            continue
        value = self.model[key]

        if key == "xyz":
            g = self.model["lattice_vector"]
            xyz = np.dot(g, value.T).T
            xyz = xyz[np.newaxis, :] + translational_vectors_car[:, np.newaxis]
            value = xyz.reshape(-1, 3)  # Flatten the coordinates
        else:
            value = np.tile(value, (p, 1))
            value = value.reshape(p, -1).flatten()

        self.model[key] = value

    self.model["lattice"] = np.array([[1.0, 0.0, 0.0], 
                                      [0.0, 1.0, 0.0], 
                                      [0.0, 0.0, 1.0]]) #set to identity matrix
    self.model["lattice_vector"] = np.array([[1.0, 0.0, 0.0], 
                                             [0.0, 1.0, 0.0], 
                                             [0.0, 0.0, 1.0]])
    self.model["pbc"] = np.array([False, False, False])  # Set to non-periodic
    return self
