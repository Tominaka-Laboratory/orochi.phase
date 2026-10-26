import numpy as np


def cubic_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        phase["lattice", "fit", 0] = True
        phase["lattice", "min", 0] = value[0] * min
        phase["lattice", "max", 0] = value[0] * max

    return restraint

def tetragonal_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = value[index] * min
            phase["lattice", "max", index] = value[index] * max
    return restraint


def orthorhombic_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1,2]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = value[index] * min
            phase["lattice", "max", index] = value[index] * max
    return restraint

def hexagonal_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,2]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = value[index] * min
            phase["lattice", "max", index] = value[index] * max
    return restraint

def rhombohedral_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,3]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = value[index] * min
            phase["lattice", "max", index] = value[index] * max
    return restraint

def monoclinic_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1,2,4]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = value[index] * min
            phase["lattice", "max", index] = value[index] * max
    return restraint

def triclinic_relative(min=0.98, max=1.02):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1,2,3,4,5]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = value[index] * min
            phase["lattice", "max", index] = value[index] * max
    return restraint

 

 
def cubic_absolute(min, max):
    def restraint(phase):
        value = phase["lattice", "value"]
        phase["lattice", "fit", 0] = True
        phase["lattice", "min", 0] = min
        phase["lattice", "max", 0] = max

    return restraint

def tetragonal_absolute(min=[], max=[]):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = min[index]
            phase["lattice", "max", index] = max[index]
    return restraint


def orthorhombic_absolute(min=[], max=[]):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1,2]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = min[index]
            phase["lattice", "max", index] = max[index]
    return restraint

def hexagonal_absolute(min=[], max=[]):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,2]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = min[index]
            phase["lattice", "max", index] = max[index]
    return restraint

def rhombohedral_absolute(min=[], max=[]):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,3]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = min[index]
            phase["lattice", "max", index] = max[index]
    return restraint

def monoclinic_absolute(min=[], max=[]):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1,2,4]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = min[index]
            phase["lattice", "max", index] = max[index]
    return restraint

def triclinic_absolute(min=[], max=[]):
    def restraint(phase):
        value = phase["lattice", "value"]
        for index in [0,1,2,3,4,5]:
            phase["lattice", "fit", index] = True
            phase["lattice", "min", index] = min[index]
            phase["lattice", "max", index] = max[index]
    return restraint

 

  