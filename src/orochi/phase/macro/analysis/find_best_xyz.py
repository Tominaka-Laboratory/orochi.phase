from ..modify import xyz

def potential_minimum(
    max_move_radius=0.1, longest_threshold=1.2
    ):
    def in_selected_phase(phase):
        def in_selected_index(atom_index):
            res = xyz.find_pair_potential_minimum_for_atom(
                phase, atom_index, max_move_radius, longest_threshold
            )
            return res
        return in_selected_index
    return in_selected_phase


def charge_neutrality_minimum(
    max_move_radius=0.1, longest_threshold=2.0
    ):
    def in_selected_phase(phase):
        def in_selected_index(atom_index):
            res = xyz.find_charge_neutrality_minimum_for_atom_without_collision(
                phase, atom_index, max_move_radius, longest_threshold
            )
            return res
        return in_selected_index
    return in_selected_phase


def charge_neutrality_minimum_vesta(
    max_move_radius=0.1, longest_threshold=2.0
    ):
    def in_selected_phase(phase):
        def in_selected_index(atom_index):
            res = xyz.find_charge_neutrality_minimum_for_atom_without_collision_vesta(
                phase, atom_index, max_move_radius, longest_threshold
            )
            return res
        return in_selected_index
    return in_selected_phase


def distortion_index_minimum_vesta(
    max_move_radius=0.1, longest_threshold=1.3
    ):
    def in_selected_phase(phase):
        def in_selected_index(atom_index):
            res = xyz.find_distortion_index_minimum_for_atom_without_collision_vesta(
                phase, atom_index, max_move_radius, longest_threshold
            )
            return res
        return in_selected_index
    return in_selected_phase


def symmetrized_positions(symprec=10**-4):
    def in_selected_phase(phase):
        is_symmerized, xyz = (
            xyz.find_symmtric_xyz_under_fixed_lattice_vector(
                phase, symprec
            )
        )
        return is_symmerized, xyz
    return in_selected_phase