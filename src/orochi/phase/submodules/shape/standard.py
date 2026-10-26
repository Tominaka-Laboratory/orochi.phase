from ..phase import Phase

def ceo2():
    phase = Phase()
    phase["symbol"] = ["Ce", "O"]
    phase["label"] = ["Ce1", "O1"]
    phase["occ"] = [1.0, 1.0]
    phase["adp"] = [0.004, 0.02]
    phase["charge"] = [0, 0]
    phase["lattice"] = [5.41165, 5.41165, 5.41165, 90, 90, 90]
    phase["space_group", "number"] = 225
    phase["xyz"] = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    phase["shape_type"] = "analytical"
    phase["sphere"] = [100, 100, 100]
    return phase


def ni():
    phase = Phase()
    phase["symbol"] = ["Ni"]
    phase["label"] = ["Ni1"]
    phase["occ"] = [1.0]
    phase["adp"] = [0.003]
    phase["charge"] = [0]
    phase["lattice"] = [3.52760, 3.52760, 3.52760, 90, 90, 90]
    phase["space_group", "number"] = 225
    phase["xyz"] = [[0.0, 0.0, 0.0]]
    phase["shape_type"] = "analytical"
    phase["sphere"] = [100, 100, 100]
    return phase
