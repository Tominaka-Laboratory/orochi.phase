import numpy as np


def elemental_wise_identical(min=[0.001], max=[0.1]):
    def restraint(phase):
        symbol = phase["symbol"]
        phase["adp", "fit"] = np.full_like(symbol, False)
        unique_symbol, id = np.unique(symbol, return_index=True)

        for n, index in enumerate(id):
           phase["adp", "fit", index] = True
           phase["adp", "min", index] = min[n]
           phase["adp", "max", index] = max[n]
        
    return restraint

def identical(min=0.001, max=0.1):
    def restraint(phase):
        phase["adp", "fit", 0] = True
        phase["adp", "min", 0] = min
        phase["adp", "max", 0] = max

    return restraint
