from orochi.phase.phase import par as ppar
from orochi.phase.phase import calculator as pcal
from orochi.phase.macro import typical_phase


def formal_charge(cif_file):
    phase = typical_phase.bulk(cif_file)

    phase.par["shape_type"] = "analytical"
    phase.par.par["composition"]["nominal"] = {"Ce4+": 1, "O2-": 2}

    phase = pcal(phase_par)
    phase.get_all_sites()
    phase.get_atom_information()
    phase.analyze_material_type()
    phase.get_formal_charge()

    print(phase.lattice_sites)
    return phase
