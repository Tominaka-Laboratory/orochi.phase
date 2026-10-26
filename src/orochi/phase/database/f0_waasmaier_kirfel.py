import re
import numpy as np
from .info.f0 import symbol_data

def f0_function(atom):
    """Atomic scattering factors.

    Source: D. Waasmaier & A. Kirfel (Acta Cryst. (1995). A51, 416-413) (f0 for nonionic atoms)
    list [a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5]
    f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
    return q dependent function

    Parameters
    ----------
    atom : string
        element type

    Returns
    -------
    f(q)
        Function returning 1d array form of scattering factor at eneryg of q
    """    


    list = symbol_data.get(atom)
    if list == None:
        atom = re.sub(r"[-+,\d]", "", atom)
        list = symbol_data[atom]

    data = np.array(list)
    f0a = data[0:5]
    f0b = data[6:11]
    f0c = data[5]

    def get_f0(q):
        s = np.square(q / (4.0 * np.pi))  # =(sin(theta)/rambda)^2
        f0 = (
            np.sum(
                f0a * np.exp(-1.0 * f0b[np.newaxis, :] * s[:, np.newaxis]),
                axis=-1,
            )
            + f0c
        )
        return f0

    return get_f0
