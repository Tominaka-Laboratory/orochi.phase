import numpy as np

def sphere():
    def constraint(phase):
        value = phase["sphere", "value"]
        phase["sphere", "value"] = np.array([value[0],value[0],[value[0]]])
    return constraint
