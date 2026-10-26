import sys, os
from .formats.cif import cif_to_phase, phase_to_cif
from .formats.cube import cube_to_phase, phase_to_cube
from .formats.xyz import phase_to_xyz


def save(phase, file_name):
    save_object(file_name)(phase)


def read(phase, file_name):
    update_phase(file_name)(phase)


def update_phase(file_name):
    ext = os.path.splitext(file_name)[-1].strip(".")
    def run(phase):
        if ext in ["cif", "CIF"]:
            cif_to_phase(file_name, phase)

        elif ext in ["cube"]:
            cube_to_phase(file_name, phase)
    return run


def save_object(file_name):
    ext = os.path.splitext(file_name)[-1].strip(".")
    def run(phase):
        if ext in ["cif", "CIF"]:
            phase_to_cif(phase, file_name)

        elif ext in ["cube"]:
            phase_to_cube(phase, file_name)

        elif ext in ["xyz"]:
            phase_to_xyz(phase, file_name)
    return run




