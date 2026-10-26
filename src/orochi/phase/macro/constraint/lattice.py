import numpy as np

def cubic():
    def constraint(phase):
        value = phase["lattice", "value"]
        phase["lattice","value"] =  np.array([value[0],value[0],value[0],90.,90.,90.])
    return constraint

def tetragonal():
    def constraint(phase):
        value = phase["lattice", "value"]
        phase["lattice","value"] =  np.array([value[0],value[0],value[2],90.,90.,90.])
    return constraint

def orthorhombic():
    def constraint(phase):
        value = phase["lattice", "value"]
        phase["lattice","value"] =  np.array([value[0],value[1],value[2],90.,90.,90.])
    return constraint

def hexagonal():
    def constraint(phase):
        value = phase["lattice", "value"]
        phase["lattice","value"] =  np.array([value[0],value[0],value[2],90.,90.,120.])
    return constraint

def rhombohedral():
    def constraint(phase):
        value = phase["lattice", "value"]
        phase["lattice","value"] =  np.array([value[0],value[0],value[0],value[3],value[3],value[3]])
    return constraint

def monoclinic():
    def constraint(phase):
        value = phase["lattice", "value"]
        phase["lattice","value"] =  np.array([value[0],value[0],value[0],90.,value[3],90.])
    return constraint





        
