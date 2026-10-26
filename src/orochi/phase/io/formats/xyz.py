import numpy as np


def phase_to_xyz(phase, file):
    def ndarray_1d(input):
        return np.array(input).flatten()

    symbols = phase.model["symbol"]
    x = np.round(phase.model["xyz"][:, 0], 6)
    y = np.round(phase.model["xyz"][:, 1], 6)
    z = np.round(phase.model["xyz"][:, 2], 6)

    header = str(int(symbols.size)) + "\n"
    data = np.stack(
        [
            ndarray_1d(symbols),
            ndarray_1d(x),
            ndarray_1d(y),
            ndarray_1d(z),
        ]
    ).T

    with open(file, "w") as f:
        np.savetxt(f, data, fmt="%s", header=header, comments="")

