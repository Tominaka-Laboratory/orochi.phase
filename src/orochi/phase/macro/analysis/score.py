import numpy as np
from orochi.phase.submodules import evaluation as pot
from orochi.phase.macro.analysis import analyze_structure


def pair_potential(averaging=True):
    def scoring(phase):
        pair_potentials = pot.get_pair_potential(phase) 
        if averaging:
            return np.average(pair_potentials)
        else:
            return pair_potentials
    return scoring


def LJC_potential(averaging=True):
    def scoring(phase):
        pair_potentials = pot.get_LJC_potential(phase) 
        if averaging:
            return np.average(pair_potentials)
        else:
            return pair_potentials
    return scoring

def effective_charge_potential(averaging=True):
    def scoring(phase):
        pair_potentials = pot.get_effective_charge_potential(phase) 
        if averaging:
            return np.average(pair_potentials)
        else:
            return pair_potentials
    return scoring

def effective_charge_potential_charge_neutrality(averaging=True):
    def scoring(phase):
        pair_potentials = pot.get_effective_charge_potential_charge_neutrality_vesta(
            phase, longest_threshold=1.3
        )
        if averaging:
            return np.average(pair_potentials)
        else:
            return pair_potentials
    return scoring

def distortion_indices(averaging=True):
    def scoring(phase):
        pair_potentials = pot.get_distortion_index(
            phase, longest_threshold=1.3
        )
        if averaging:
            return np.average(pair_potentials)
        else:
            return pair_potentials
    return scoring

def precision_alternative_space_group():
    def scoring(phase):
        symmetry_score = 0.0
        points = np.logspace(-0.1, -6, num=20)
        for p in points:
            is_new = analyze_structure.check_alternative_space_group(
                phase, symprec=p
            )
            if is_new:
                symmetry_score = np.log10(p)
            else:
                break
        return symmetry_score
    return scoring



