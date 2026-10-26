import numpy as np
from ..modify import xyz as modify_xyz

def max_move_constraints(max_move_radius=0.01):
    def function(orochi_module, key):
        if key == "xyz":
            return xyz_constraint(orochi_module, key, max_move_radius)
        else:
            return None
    return function


def max_move_without_collision_to_all_others_constraints(max_move_radius=0.01):
    def function(orochi_module, key):
        if key == "xyz":
            return xyz_constraint_without_collision_to_all_others(
                orochi_module, key, max_move_radius
            )
        else:
            return None
    return function


def max_move_without_collision_to_youngers_constraints(max_move_radius=0.01):
    def function(orochi_module, key):
        if key == "xyz":
            return xyz_constraint_without_collision_to_youngers(
                orochi_module, key, max_move_radius
            )
        else:
            return None
    return function



def xyz_constraint(orochi_module, key, max_move_radius):
    def local1(id1):
        class coordinates:
            value = modify_xyz.spherical_coordinates_to_move(
                phase=orochi_module,
                atom_index=id1,
                max_move_radius=max_move_radius,
            )

        c = coordinates()
        def local2(c):
            def objective(id2):
                if id2 == 0:
                    return np.amin(c.value[:, 0]), np.amax(c.value[:, 0])

                elif id2 == 1:
                    d = np.abs(c.value[:, 0] - orochi_module.par[key][id1][0])
                    c.value = c.value[np.where(d == np.amin(d))]
                    return np.amin(c.value[:, 1]), np.amax(c.value[:, 1])

                elif id2 == 2:
                    d = np.abs(c.value[:, 0] - orochi_module.par[key][id1][1])
                    c.value = c.value[np.where(d == np.amin(d))]
                    return np.amin(c.value[:, 2]), np.amax(c.value[:, 2])

            return objective
        return local2(c)
    return local1


def xyz_constraint_without_collision_to_all_others(
    orochi_module, key, max_move_radius
):

    def local1(id1):
        class coordinates:
            value = modify_xyz.clear_coordinates_to_move_without_collision_to_all_others(
                phase=orochi_module,
                atom_index=id1,
                max_move_radius=max_move_radius,
                longest_threshold=1.2,
                shortest_threshold=0.8,
            )

        c = coordinates()
        def local2(c):
            def objective(id2):
                if id2 == 0:
                    return np.amin(c.value[:, 0]), np.amax(c.value[:, 0])

                elif id2 == 1:
                    d = np.abs(c.value[:, 0] - orochi_module.par[key][id1][0])
                    c.value = c.value[np.where(d == np.amin(d))]
                    return np.amin(c.value[:, 1]), np.amax(c.value[:, 1])

                elif id2 == 2:
                    d = np.abs(c.value[:, 0] - orochi_module.par[key][id1][1])
                    c.value = c.value[np.where(d == np.amin(d))]
                    return np.amin(c.value[:, 2]), np.amax(c.value[:, 2])

            return objective
        return local2(c)
    return local1


def xyz_constraint_without_collision_to_youngers(
    orochi_module, key, max_move_radius
):
    def local1(id1):
        class coordinates:
            value = modify_xyz.clear_coordinates_to_move_without_collision_to_youngers(
                phase=orochi_module,
                atom_index=id1,
                max_move_radius=max_move_radius,
                longest_threshold=1.2,
                shortest_threshold=0.8,
            )

        c = coordinates()
        def local2(c):
            def objective(id2):
                if id2 == 0:
                    return np.amin(c.value[:, 0]), np.amax(c.value[:, 0])

                elif id2 == 1:
                    d = np.abs(c.value[:, 0] - orochi_module.par[key][id1][0])
                    c.value = c.value[np.where(d == np.amin(d))]
                    return np.amin(c.value[:, 1]), np.amax(c.value[:, 1])

                elif id2 == 2:
                    d = np.abs(c.value[:, 0] - orochi_module.par[key][id1][1])
                    c.value = c.value[np.where(d == np.amin(d))]
                    return np.amin(c.value[:, 2]), np.amax(c.value[:, 2])

            return objective

        return local2(c)
    return local1