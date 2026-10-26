import numpy as np
import re
bohr = 0.5291772  # angstrom

def cube_to_phase(cube_file, phase):
    with open(cube_file, "r") as f:
        data = f.readlines()
        number_atoms = int(data[2].split()[0])
        number_xaxis = int(data[3].split()[0])
        number_yaxis = int(data[4].split()[0])
        number_zaxis = int(data[5].split()[0])

        origin = np.array(data[2].split()[1:]).astype(float)  # in bohr
        step_xaxis = np.array(data[3].split()[1:]).astype(float)  # in bohr
        step_yaxis = np.array(data[4].split()[1:]).astype(float)  # in bohr
        step_zaxis = np.array(data[5].split()[1:]).astype(float)  # in bohr

        lattice_vector = np.array(
            [step_xaxis, step_yaxis, step_zaxis]
        ) * np.array([number_xaxis, number_yaxis, number_zaxis])
        lattice_vector *= bohr
        lattice = np.round(phase.get_lattice_from_vector(lattice_vector), 6)

        xaxis = np.linspace(0.0, 1.0, number_xaxis + 1)
        yaxis = np.linspace(0.0, 1.0, number_yaxis + 1)
        zaxis = np.linspace(0.0, 1.0, number_zaxis + 1)
        xaxis = (xaxis[1:] + xaxis[:-1]) / 2.0
        yaxis = (yaxis[1:] + yaxis[:-1]) / 2.0
        zaxis = (zaxis[1:] + zaxis[:-1]) / 2.0

        xx, yy, zz = np.meshgrid(xaxis, yaxis, zaxis, indexing="ij")
        xyz = np.concatenate(
            [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=-1
        ).reshape((number_xaxis, number_yaxis, number_zaxis, 3))

        number_header = 6 + number_atoms
        data = " ".join(data[number_header:])
        data = re.sub("[\n]", " ", data)
        data = data.split()
        data = (
            np.array(data)
            .astype(float)
            .reshape((number_xaxis, number_yaxis, number_zaxis))
        )  # e/Bohr^3

    phase["lattice"] = lattice
    phase["map","xyz"] = xyz
    volume = np.linalg.det(phase.get_lattice_vector(lattice))
    phase["map","value"] = (
        data
        * (bohr**3.0)
        * (volume / number_xaxis / number_yaxis / number_zaxis)
    )  # number of electrons


def cube_to_laplacian(cube_file):
    with open(cube_file, "r") as f:
        data = f.readlines()
        number_atoms = int(data[2].split()[0])
        number_xaxis = int(data[3].split()[0])
        number_yaxis = int(data[4].split()[0])
        number_zaxis = int(data[5].split()[0])

        origin = np.array(data[2].split()[1:]).astype(float)  # in bohr
        step_xaxis = np.array(data[3].split()[1:]).astype(float)  # in bohr
        step_yaxis = np.array(data[4].split()[1:]).astype(float)  # in bohr
        step_zaxis = np.array(data[5].split()[1:]).astype(float)  # in bohr

        lattice_vector = np.array(
            [step_xaxis, step_yaxis, step_zaxis]
        ) * np.array([number_xaxis, number_yaxis, number_zaxis])
        lattice_vector *= bohr
        lattice = np.round(phase.get_lattice_from_vector(lattice_vector), 6)

        xaxis = np.linspace(0.0, 1.0, number_xaxis + 1)
        yaxis = np.linspace(0.0, 1.0, number_yaxis + 1)
        zaxis = np.linspace(0.0, 1.0, number_zaxis + 1)
        xaxis = (xaxis[1:] + xaxis[:-1]) / 2.0
        yaxis = (yaxis[1:] + yaxis[:-1]) / 2.0
        zaxis = (zaxis[1:] + zaxis[:-1]) / 2.0

        xx, yy, zz = np.meshgrid(xaxis, yaxis, zaxis, indexing="ij")
        xyz = np.concatenate(
            [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=-1
        )

        number_header = 6 + number_atoms
        data = " ".join(data[number_header:])
        data = re.sub("[\n]", " ", data)
        data = data.split()
        data = (
            np.array(data)
            .astype(float)
            .reshape((number_xaxis, number_yaxis, number_zaxis))
        )

        dx = lattice[0] / number_xaxis
        dy = lattice[1] / number_yaxis
        dz = lattice[2] / number_zaxis

        laplacian = (
            np.gradient(np.gradient(data, axis=0), axis=0) / (dx**2.0)
            + np.gradient(np.gradient(data, axis=1), axis=1) / (dy**2.0)
            + np.gradient(np.gradient(data, axis=2), axis=2) / (dz**2.0)
        )

    phase["lattice"] = lattice
    phase["xyz"] = xyz
    phase["occ"] = data.flatten()
    # phase["symbol"] = np.full(xyz.shape[0], "e")
    # phase["label"] = np.full(xyz.shape[0], "e")
    # phase["adp"] = np.full(xyz.shape[0], 0.01)
    phase["occ","laplacian"] = laplacian.flatten()


def phase_to_cube(phase, file, number_xaxis, number_yaxis, number_zaxis):

    lattice_vector = phase.get_lattice_vector(phase["lattice"])
    lattice_vector_bohr = lattice_vector / bohr
    lattice_step_bohr = lattice_vector_bohr / np.array(
        [number_xaxis, number_yaxis, number_zaxis]
    )
    lattice_step_bohr_x = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[0],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )
    lattice_step_bohr_y = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[1],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )
    lattice_step_bohr_z = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[2],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )

    xyz = phase["map","xyz"]
    data = phase["map","value"] / (bohr**3.0)
    data = data.reshape((number_xaxis, number_yaxis, number_zaxis))
    pad = 6 - (number_zaxis % 6)
    pad = np.full((number_xaxis, number_yaxis, pad), np.nan)
    data = np.concatenate([data, pad], axis=-1)
    data = data.reshape((number_xaxis * number_yaxis, -1, 6))

    header = [
        " Cubefile created from phase module",
        "Contains the selected quantity on a FFT grid",
        "0    0.000000    0.000000    0.000000",
        str(number_xaxis) + " " + lattice_step_bohr_x,
        str(number_yaxis) + " " + lattice_step_bohr_y,
        str(number_zaxis) + " " + lattice_step_bohr_z,
    ]
    header = "\n".join(header)

    with open(file, "w") as f:
        f.writelines(header)
        f.writelines("\n")

        for line in data:
            np.savetxt(f, line, fmt="%.5E")

    with open(file, "r") as f:
        lines = f.read()
        lines = re.sub("NAN", "", lines)

    with open(file, "w") as f:
        f.write(lines)


def phase_to_laplacian_cube(
    phase, file, number_xaxis, number_yaxis, number_zaxis
):
    lattice_vector = phase.get_lattice_vector(phase["lattice"])
    lattice_vector_bohr = lattice_vector / bohr
    lattice_step_bohr = lattice_vector_bohr / np.array(
        [number_xaxis, number_yaxis, number_zaxis]
    )
    lattice_step_bohr_x = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[0],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )
    lattice_step_bohr_y = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[1],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )
    lattice_step_bohr_z = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[2],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )

    xyz = phase["map","xyz"]
    data = phase["map","value"] / (bohr**3.0)
    data = data.reshape((number_xaxis, number_yaxis, number_zaxis))

    lattice = phase["lattice"]
    dx = lattice[0] / number_xaxis
    dy = lattice[1] / number_yaxis
    dz = lattice[2] / number_zaxis

    data = (
        np.gradient(np.gradient(data, axis=0), axis=0) / (dx**2.0)
        + np.gradient(np.gradient(data, axis=1), axis=1) / (dy**2.0)
        + np.gradient(np.gradient(data, axis=2), axis=2) / (dz**2.0)
    )

    pad = 6 - (number_zaxis % 6)
    pad = np.full((number_xaxis, number_yaxis, pad), np.nan)
    data = np.concatenate([data, pad], axis=-1)
    data = data.reshape((number_xaxis * number_yaxis, -1, 6))

    header = [
        " Cubefile created from phase module",
        "Contains the selected quantity on a FFT grid",
        "0    0.000000    0.000000    0.000000",
        str(number_xaxis) + " " + lattice_step_bohr_x,
        str(number_yaxis) + " " + lattice_step_bohr_y,
        str(number_zaxis) + " " + lattice_step_bohr_z,
    ]
    header = "\n".join(header)

    with open(file, "w") as f:
        f.writelines(header)
        f.writelines("\n")

        for line in data:
            np.savetxt(f, line, fmt="%.5E")

    with open(file, "r") as f:
        lines = f.read()
        lines = re.sub("NAN", "", lines)

    with open(file, "w") as f:
        f.write(lines)


def phase_to_rdf_cube(
    phase, file, number_xaxis, number_yaxis, number_zaxis, origin, maxr=0.0
):

    lattice_vector = phase.get_lattice_vector(phase["lattice"])
    lattice_vector_bohr = lattice_vector / bohr
    lattice_step_bohr = lattice_vector_bohr / np.array(
        [number_xaxis, number_yaxis, number_zaxis]
    )
    lattice_step_bohr_x = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[0],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )
    lattice_step_bohr_y = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[1],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )
    lattice_step_bohr_z = re.sub(
        "[],[]+",
        "",
        np.array2string(
            lattice_step_bohr[2],
            formatter={"float_kind": lambda x: "%.6f" % x},
        ),
    )

    xyz = phase["map","xyz"]
    data = phase["map","value"] / (bohr**3.0)

    r = np.linalg.norm(np.dot(lattice_vector, (xyz - origin).T).T, axis=-1)

    if maxr != 0.0:
        r[r > maxr] = 0.0
    data = (occ * (r**2.0)) * 4.0 * np.pi
    data = data.reshape((number_xaxis, number_yaxis, number_zaxis))

    pad = 6 - (number_zaxis % 6)
    pad = np.full((number_xaxis, number_yaxis, pad), np.nan)
    data = np.concatenate([data, pad], axis=-1)
    data = data.reshape((number_xaxis * number_yaxis, -1, 6))

    header = [
        " Cubefile created from phase module",
        "Contains the selected quantity on a FFT grid",
        "0    0.000000    0.000000    0.000000",
        str(number_xaxis) + " " + lattice_step_bohr_x,
        str(number_yaxis) + " " + lattice_step_bohr_y,
        str(number_zaxis) + " " + lattice_step_bohr_z,
    ]
    header = "\n".join(header)

    with open(file, "w") as f:
        f.writelines(header)
        f.writelines("\n")

        for line in data:
            np.savetxt(f, line, fmt="%.5E")

    with open(file, "r") as f:
        lines = f.read()
        lines = re.sub("NAN", "", lines)

    with open(file, "w") as f:
        f.write(lines)
