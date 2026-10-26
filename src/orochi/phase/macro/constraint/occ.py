import numpy as np


def identical(key="occ"):
    def constraint(phase):
        value = phase[key, "value"]
        if isinstance(value, (float, int)):
            pass
        else:
            phase[key] =  np.full_like(value, value[0])
    return constraint

def elemental_wise_identical(key="occ"):
    def constraint(phase):
        value = phase[key, "value"]
        symbol = phase[key, "symbol"]
        unique_symbols = np.unique(symbol)

        for sym in unique_symbols:
            indices = np.where(symbol == sym)[0]
            value[indices] = value[indices[0]]

        phase[key, "value"] = value
    return constraint

    




        
