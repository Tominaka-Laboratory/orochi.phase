from typing import Union, List
import numpy as np

def sphere(self):
    """
    Creates a spherical/ellipsoidal explicit nanoparticle model.

    Parameters
    ----------
    self : Phase
        The Phase instance to expand and carve into a sphere/ellipsoid.

    Returns
    -------
    Phase
        The updated Phase object representing the spherical/ellipsoidal model.
    """
    from .base import spheroid
    return spheroid.make(self)


def cube(self):
    """
    Creates a cubic explicit nanoparticle model.

    Parameters
    ----------
    self : Phase
        The Phase instance to expand and carve into a cube.

    Returns
    -------
    Phase
        The updated Phase object representing the cubic model.
    """
    from .base import polygon
    polygon.apply_lattice_symmetry(self)
    return polygon.make(self)


def octahedron(self):
    """
    Creates an octahedral explicit nanoparticle model.

    Parameters
    ----------
    self : Phase
        The Phase instance to expand and carve into an octahedron.

    Returns
    -------
    Phase
        The updated Phase object representing the octahedral model.
    """
    from .base import polygon
    polygon.apply_lattice_symmetry(self)
    return polygon.make(self)


def cylinder(self):
    """
    Creates a cylindrical explicit nanoparticle model.

    Parameters
    ----------
    self : Phase
        The Phase instance to expand and carve into a cylinder.

    Returns
    -------
    Phase
        The updated Phase object representing the cylindrical model.
    """
    from .base import cylinder
    self["cylinder", "origin"] = [[0.0, 0.0, 0.0]]
    self["cylinder", "direction"] = [[0, 0, 1]]
    self["cylinder", "max_length"] = [100.0]
    return cylinder.make(self)

