import re
import numpy as np
from .info.compton import symbol_data

def compton_function(atom):
    """Compton scattering

    Source: D.T. Cormer and J. B. Mann: J. Chem. Phys., 47 1892 (1967)
    return q-dependent function


    Returns
    -------
    function(q)
        Function returning 1d array form of Compoton scattering at eneryg of q
    """    

    atom = re.sub(r"[-+,\d]", "", atom)
    p = symbol_data.get(atom)

    def compton(q):
        cf = (
            p[0]
            - (
                p[1] * np.exp(-p[2] * (q / 4.0 / np.pi) ** 2.0)
                + p[3] * np.exp(-p[4] * (q / 4.0 / np.pi) ** 2)
                + p[5] * np.exp(-p[6] * (q / 4.0 / np.pi) ** 2.0)
                + p[7]
            )
            ** 2.0
        )
        return cf

    return compton
