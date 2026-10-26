import numpy as np
from ..macro.modify.stack import pattern_stacking


def general(
    phase, inplane_vector, stack_sequence
):  # This is for simulating stacking faults
    stacking_axis = 2
    phase = pattern_stacking(
        phase, stack_sequence, inplane_vector, stacking_axis
    )
    return phase


def hexagonal(phase, stack_sequence=np.array(["A", "B", "C", "A", "B", "C"])):

    inplane_vector = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([1.0 / 3.0, 2.0 / 3.0]),
        "C": np.array([2.0 / 3.0, 1.0 / 3.0]),
    }

    stacking_axis = 2
    phase = pattern_stacking(
        phase, stack_sequence, inplane_vector, stacking_axis
    )

    return phase
