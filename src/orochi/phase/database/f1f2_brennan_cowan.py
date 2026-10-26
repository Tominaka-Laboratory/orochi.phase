import re
import numpy as np
from .info.f1f2 import symbol_data

def f1f2_function(atom):
    """Energy-dependence functions of scattering factors.
    phton energy [ev], f1, f2

    Source: S. Brennan and P.L. Cowan (1992), Rev. Sci. Instrum. 63, 850.

    Parameters
    ----------
    atom : string
        Element type

    Returns
    -------
    list
        list(numpy.ndarray(f1), numpy.ndarray(f2))
    """    


    atom = re.sub(r"[-+,\d]", "", atom)
    list = symbol_data.get(atom)
    if list == None:
        list = [[0, 0, 0], [200000, 0, 0]]

    data = np.array(list)

    def get_f1f2(energy):
        f1 = np.interp(energy, data[:, 0], data[:, 1])
        f2 = np.interp(energy, data[:, 0], data[:, 2])
        return [f1, f2]

    return get_f1f2
