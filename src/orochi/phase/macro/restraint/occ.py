import numpy as np


def select_elemental(element, min=0.001, max=0.1):
    def restraint(phase):
        symbol = phase["symbol"]
        phase["occ", "fit"] = np.full_like(symbol, False)

        index = np.where(symbol == element)[0][0]
        phase["occ", "fit", index] = True
        phase["occ", "min", index] = min
        phase["occ", "max", index] = max
        
    return restraint

def elemental_wise_identical(min=[0.001], max=[0.1]):
    def restraint(phase):
        symbol = phase["symbol"]
        phase["occ", "fit"] = np.full_like(symbol, False)
        unique_symbol, id = np.unique(symbol, return_index=True)

        for n, index in enumerate(id):
           phase["occ", "fit", index] = True
           phase["occ", "min", index] = min[n]
           phase["occ", "max", index] = max[n]
        
    return restraint

def identical(min=0.001, max=0.1):
    def restraint(phase):
        phase["occ", "fit", 0] = True
        phase["occ", "min", 0] = min
        phase["occ", "max", 0] = max

    return restraint




