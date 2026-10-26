import numpy as np
import sympy
x, y, z = sympy.symbols("x y z")

"""
Point Group Symmetry Operations Database
Keys: Hermann-Mauguin symbols as defined in International Tables for Crystallography.
Format: List of (x, y, z) strings compatible with SymPy parsing.
Note: Translations (e.g., +1/2) are excluded as these are point group operations.
"""



point_group_dictionary = {
    # --- Triclinic ---
    "1": [(x, y, z)],
    "-1": [(x, y, z), (-x, -y, -z)],

    # --- Monoclinic (b-axis unique setting) ---
    "2": [(x, y, z), (-x, y, -z)],
    "m": [(x, y, z), (x, -y, z)],
    "2/m": [(x, y, z), (-x, y, -z), (-x, -y, -z), (x, -y, z)],

    # --- Orthorhombic ---
    "222": [(x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z)],
    "mm2": [(x, y, z), (-x, -y, z), (x, -y, z), (-x, y, z)],
    "mmm": [
        (x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
        (-x, -y, -z), (x, y, -z), (x, -y, z), (-x, y, z)
    ],

    # --- Tetragonal ---
    "4": [(x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z)],
    "-4": [(x, y, z), (-x, -y, z), (y, -x, -z), (-y, x, -z)],
    "4/m": [
        (x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z),
        (-x, -y, -z), (x, y, -z), (y, -x, -z), (-y, x, -z)
    ],
    "422": [
        (x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z),
        (-x, y, -z), (x, -y, -z), (y, x, -z), (-y, -x, -z)
    ],
    "4mm": [
        (x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z),
        (x, -y, z), (-x, y, z), (-y, -x, z), (y, x, z)
    ],
    "-42m": [
        (x, y, z), (-x, -y, z), (y, -x, -z), (-y, x, -z),
        (-x, y, -z), (x, -y, -z), (-y, -x, z), (y, x, z)
    ],
    "4/mmm": [
        (x, y, z), (-x, -y, z), (-y, x, z), (y, -x, z),
        (-x, y, -z), (x, -y, -z), (y, x, -z), (-y, -x, -z),
        (-x, -y, -z), (x, y, -z), (y, -x, -z), (-y, x, -z),
        (x, -y, z), (-x, y, z), (-y, -x, z), (y, x, z)
    ],

    # --- Trigonal (Hexagonal axes) ---
    "3": [(x, y, z), (-y, x - y, z), (-x + y, -x, z)],
    "-3": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-x, -y, -z), (y, -x + y, -z), (x - y, x, -z)
    ],
    "32": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (y, x, -z), (x - y, -y, -z), (-x, -x + y, -z)
    ],
    "3m": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-y, -x, z), (-x + y, y, z), (x, x - y, z)
    ],
    "-3m": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (y, x, -z), (x - y, -y, -z), (-x, -x + y, -z),
        (-x, -y, -z), (y, -x + y, -z), (x - y, x, -z),
        (-y, -x, z), (-x + y, y, z), (x, x - y, z)
    ],

    # --- Hexagonal ---
    "6": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-x, -y, z), (y, -x + y, z), (x - y, x, z)
    ],
    "-6": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (x, y, -z), (-y, x - y, -z), (-x + y, -x, -z)
    ],
    "6/m": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-x, -y, z), (y, -x + y, z), (x - y, x, z),
        (-x, -y, -z), (y, -x + y, -z), (x - y, x, -z),
        (x, y, -z), (-y, x - y, -z), (-x + y, -x, -z)
    ],
    "622": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-x, -y, z), (y, -x + y, z), (x - y, x, z),
        (y, x, -z), (x - y, -y, -z), (-x, -x + y, -z),
        (-y, -x, -z), (-x + y, y, -z), (x, x - y, -z)
    ],
    "6mm": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-x, -y, z), (y, -x + y, z), (x - y, x, z),
        (-y, -x, z), (-x + y, y, z), (x, x - y, z),
        (y, x, z), (x - y, -y, z), (-x, -x + y, z)
    ],
    "-6m2": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (x, y, -z), (-y, x - y, -z), (-x + y, -x, -z),
        (-y, -x, z), (-x + y, y, z), (x, x - y, z),
        (-y, -x, -z), (-x + y, y, -z), (x, x - y, -z)
    ],
    "6/mmm": [
        (x, y, z), (-y, x - y, z), (-x + y, -x, z),
        (-x, -y, z), (y, -x + y, z), (x - y, x, z),
        (y, x, -z), (x - y, -y, -z), (-x, -x + y, -z),
        (-y, -x, -z), (-x + y, y, -z), (x, x - y, -z),
        (-x, -y, -z), (y, -x + y, -z), (x - y, x, -z),
        (x, y, -z), (-y, x - y, -z), (-x + y, -x, -z),
        (-y, -x, z), (-x + y, y, z), (x, x - y, z),
        (y, x, z), (x - y, -y, z), (-x, -x + y, z)
    ],

    # --- Cubic ---
    "23": [
        (x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
        (z, x, y), (z, -x, -y), (-z, -x, y), (-z, x, -y),
        (y, z, x), (-y, z, -x), (y, -z, -x), (-y, -z, x)
    ],
    "m-3": [
        (x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
        (z, x, y), (z, -x, -y), (-z, -x, y), (-z, x, -y),
        (y, z, x), (-y, z, -x), (y, -z, -x), (-y, -z, x),
        (-x, -y, -z), (x, y, -z), (x, -y, z), (-x, y, z),
        (-z, -x, -y), (-z, x, y), (z, x, -y), (z, -x, y),
        (-y, -z, -x), (y, -z, x), (-y, z, x), (y, z, -x)
    ],
    "432": [
        (x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
        (z, x, y), (z, -x, -y), (-z, -x, y), (-z, x, -y),
        (y, z, x), (-y, z, -x), (y, -z, -x), (-y, -z, x),
        (y, x, -z), (-y, -x, -z), (y, -x, z), (-y, x, z),
        (x, z, -y), (-x, z, y), (-x, -z, -y), (x, -z, y),
        (z, y, -x), (z, -y, x), (-z, y, x), (-z, -y, -x)
    ],
    "-43m": [
        (x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
        (z, x, y), (z, -x, -y), (-z, -x, y), (-z, x, -y),
        (y, z, x), (-y, z, -x), (y, -z, -x), (-y, -z, x),
        (y, x, z), (-y, -x, z), (y, -x, -z), (-y, x, -z),
        (x, z, y), (-x, z, -y), (-x, -z, y), (x, -z, -y),
        (z, y, x), (z, -y, -x), (-z, y, -x), (-z, -y, x)
    ],
    "m-3m": [
        (x, y, z), (-x, -y, z), (-x, y, -z), (x, -y, -z),
        (z, x, y), (z, -x, -y), (-z, -x, y), (-z, x, -y),
        (y, z, x), (-y, z, -x), (y, -z, -x), (-y, -z, x),
        (y, x, -z), (-y, -x, -z), (y, -x, z), (-y, x, z),
        (x, z, -y), (-x, z, y), (-x, -z, -y), (x, -z, y),
        (z, y, -x), (z, -y, x), (-z, y, x), (-z, -y, -x),
        (-x, -y, -z), (x, y, -z), (x, -y, z), (-x, y, z),
        (-z, -x, -y), (-z, x, y), (z, x, -y), (z, -x, y),
        (-y, -z, -x), (y, -z, x), (-y, z, x), (y, z, -x),
        (-y, -x, z), (y, x, z), (-y, x, -z), (y, -x, -z),
        (-x, -z, y), (x, -z, -y), (x, z, y), (-x, z, -y),
        (-z, -y, x), (-z, y, -x), (z, -y, -x), (z, y, x)
    ],
}