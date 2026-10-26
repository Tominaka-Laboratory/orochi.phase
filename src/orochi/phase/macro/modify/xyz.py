import numpy as np
from scipy.spatial import distance
import spglib
from sklearn.cluster import AgglomerativeClustering as agc
import re

from ...database.atoms import AtomInformation as ai
from ...database.bond_valence import BondValence as bv



def spherical_coordinates_to_move(phase, atom_index, max_move_radius=0.1):
    index = np.where(phase.model["site_index"] == atom_index)

    g = phase.model["lattice_vector"].T
    xyz_index = phase["xyz"][index]

    # make spherical mesh around the atom. origin  0,0,0
    mesh_cartesian = make_sphere_points(radius=max_move_radius)
    mesh_lattice = np.dot(np.linalg.inv(g), mesh_cartesian.T).T + xyz_index
    pbc = phase.model["pbc"]
    use = np.ones(mesh_cartesian.shape[0])

    for id, pbc_each in enumerate(pbc):
        if pbc_each:
            pass
        else:
            use *= np.where(
                (mesh_lattice[:, id] <= 1.0) & (mesh_lattice[:, id] >= 0.0),
                1,
                0,
            )

    mesh_lattice = mesh_lattice[np.where(use == 1)]

    return mesh_lattice


def clear_coordinates_to_move_without_collision_to_all_others(
    phase,
    atom_index,
    max_move_radius=0.1,
    longest_threshold=1.2,
    shortest_threshold=0.8,
):
    atom = phase["symbol"][atom_index]
    closest_distance = get_closest_distance_for_atom_type(phase, atom)
    closest_distance[atom_index] = 0.0

    g = phase.model["lattice_vector"].T
    xyz = phase.model["xyz"]
    xyz_index = phase["xyz"][atom_index]
    translational_vectors = get_translational_vectors_under_pbc(
        phase, np.amax(closest_distance) * longest_threshold
    )

    # make spherical mesh around the atom. origin  0,0,0
    mesh_cartesian = make_sphere_points(radius=max_move_radius)
    mesh_lattice = np.dot(np.linalg.inv(g), mesh_cartesian.T).T + xyz_index
    pbc = phase.model["pbc"]
    use = np.ones(mesh_cartesian.shape[0])

    for id, pbc_each in enumerate(pbc):
        if pbc_each:
            pass
        else:
            use *= np.where(
                (mesh_lattice[:, id] <= 1.0) & (mesh_lattice[:, id] >= 0.0),
                1,
                0,
            )

    mesh_cartesian = mesh_cartesian[np.where(use == 1)]

    # xyz of the current atoms in the phase within the max move radius
    xyz_cartesian = get_cartesian_xyz_under_pbc(
        xyz - xyz_index, g, translational_vectors
    )
    d = np.linalg.norm(xyz_cartesian, axis=-1) - closest_distance.reshape(
        -1, 1
    )
    use = np.where(d < max_move_radius)
    xyz_cartesian = xyz_cartesian[use] + np.dot(np.linalg.inv(g), xyz_index)
    closest_distance = np.tile(
        closest_distance, (translational_vectors.shape[0], 1)
    ).T[use]

    # distance between the current atoms and the mesh points to remove collision
    dist = distance.cdist(
        xyz_cartesian.reshape(-1, 3), mesh_cartesian.reshape(-1, 3)
    ).T  # shape = mesh, xyz

    collision = dist - (closest_distance * shortest_threshold)
    judge = np.where(collision > 0, 1, 0)
    judge = np.prod(judge, axis=-1)
    judge = np.where(judge == 1, 1, 0)

    mesh_cartesian = mesh_cartesian[np.where(judge != 0)]
    if mesh_cartesian.shape[0] == 0:
        mesh_xyz = xyz_index.reshape(-1, 3)
    else:
        mesh_xyz = np.dot(np.linalg.inv(g), mesh_cartesian.T).T + xyz_index

    return mesh_xyz


def clear_coordinates_to_move_without_collision_to_youngers(
    phase,
    atom_index,
    max_move_radius=0.1,
    longest_threshold=1.2,
    shortest_threshold=0.8,
):
    atom = phase["symbol"][atom_index]
    closest_distance = get_closest_distance_for_atom_type(phase, atom)[
        :atom_index
    ]

    g = phase.model["lattice_vector"].T
    xyz = phase.model["xyz"][:atom_index]
    xyz_index = phase["xyz"][atom_index]
    translational_vectors = get_translational_vectors_under_pbc(
        phase, np.amax(closest_distance) * longest_threshold
    )

    # make spherical mesh around the atom. origin  0,0,0
    mesh_cartesian = make_sphere_points(radius=max_move_radius)
    mesh_lattice = np.dot(np.linalg.inv(g), mesh_cartesian.T).T + xyz_index
    pbc = phase.model["pbc"]
    use = np.ones(mesh_cartesian.shape[0])

    for id, pbc_each in enumerate(pbc):
        if pbc_each:
            pass
        else:
            use *= np.where(
                (mesh_lattice[:, id] <= 1.0) & (mesh_lattice[:, id] >= 0.0),
                1,
                0,
            )

    mesh_cartesian = mesh_cartesian[np.where(use == 1)]

    # xyz of the current atoms in the phase within the max move radius
    xyz_cartesian = get_cartesian_xyz_under_pbc(
        xyz - xyz_index, g, translational_vectors
    )
    d = np.linalg.norm(xyz_cartesian, axis=-1) - closest_distance.reshape(
        -1, 1
    )
    use = np.where(d < max_move_radius)
    xyz_cartesian = xyz_cartesian[use] + np.dot(np.linalg.inv(g), xyz_index)
    closest_distance = np.tile(
        closest_distance, (translational_vectors.shape[0], 1)
    ).T[use]

    # distance between the current atoms and the mesh points to remove collision
    dist = distance.cdist(
        xyz_cartesian.reshape(-1, 3), mesh_cartesian.reshape(-1, 3)
    ).T  # shape = mesh, xyz

    collision = dist - (closest_distance * shortest_threshold)
    judge = np.where(collision > 0, 1, 0)
    judge = np.prod(judge, axis=-1)
    judge = np.where(judge == 1, 1, 0)

    mesh_cartesian = mesh_cartesian[np.where(judge != 0)]
    if mesh_cartesian.shape[0] == 0:
        mesh_xyz = xyz_index.reshape(-1, 3)
    else:
        mesh_xyz = np.dot(np.linalg.inv(g), mesh_cartesian.T).T + xyz_index

    return mesh_xyz


def clear_coordinates_to_add(
    phase,
    atom_type,
    longest_threshold=1.2,
    shortest_threshold=0.8,
    mesh_size=[0.3, 0.3, 0.3],
):
    closest_distance = get_closest_distance_for_atom_type(phase, atom_type)
    shortest_bond = np.amin(closest_distance)

    g = phase.model["lattice_vector"].T
    xyz = phase.model["xyz"]

    # mesh size is in angstrom. mesh is in lattice coordinates
    mesh_xyz = make_grid_points(phase, mesh_size)
    mesh_cartesian = np.dot(g, mesh_xyz.T).T

    # xyz coordinates in cartesian to get distance map
    translational_vectors = get_translational_vectors_under_pbc(
        phase, np.amax(closest_distance) * longest_threshold
    )
    xyz_current_cartesian = get_cartesian_xyz_under_pbc(
        xyz, g, translational_vectors
    )
    dist = distance.cdist(
        xyz_current_cartesian.reshape(-1, 3), mesh_cartesian.reshape(-1, 3)
    )  # shape = len(filtered current sites), len(xyz cartesian candidate sites)
    dist = dist.reshape(
        xyz_current_cartesian.shape[0], xyz_current_cartesian.shape[1], -1
    )  # shape = len(symbol), len(tv), len(xyz)

    # check collision of atoms, if one atom collides, the site should be removed
    collision = dist.T - (closest_distance * shortest_threshold)
    collision = collision.reshape(dist.shape[-1], -1)

    # remove grid points far from the current atoms
    isolate = shortest_bond * 2.0 * longest_threshold - dist.T
    isolate = isolate.reshape(dist.shape[-1], -1)

    judge = np.where((collision > 0) & (isolate > 1), 1, 0)
    judge = np.prod(judge, axis=-1)
    judge = np.where(judge == 1, 1, 0)

    mesh_xyz = mesh_xyz[np.where(judge > 0)]

    return mesh_xyz


def make_mesh_in_cartesian(phase, atom_index, max_move_radius=0.1):
    g = phase.model["lattice_vector"].T
    xyz = phase.model["xyz"]
    xyz_index = phase["xyz"][atom_index]

    # make spherical mesh around the atom
    mesh_cartesian = make_sphere_points(radius=max_move_radius) + np.dot(
        g, xyz_index
    )
    mesh_lattice = np.dot(np.linalg.inv(g), mesh_cartesian.T).T
    pbc = phase.model["pbc"]
    use = np.ones(mesh_cartesian.shape[0])

    for id, pbc_each in enumerate(pbc):
        if pbc_each:
            pass
        else:
            use *= np.where(
                (mesh_lattice[:, id] <= 1.0) & (mesh_lattice[:, id] >= 0.0),
                1,
                0,
            )
    mesh_cartesian = mesh_cartesian[np.where(use == 1)]
    return mesh_cartesian


def get_pair_potential_for_mesh(
    phase, atom_index, mesh_cartesian, longest_threshold
):
    xyz = phase.model["xyz"]
    g = phase.model["lattice_vector"].T

    ri = get_radii_for_all_sites(phase)
    rj = np.full(mesh_cartesian.shape[0], ri[atom_index])
    sigma = (rj[:, np.newaxis] + ri[np.newaxis, :]) / (
        2 ** (1 / 6)
    )  # collision distance
    longest = np.amax(ri + ri[atom_index]) * longest_threshold

    translational_vectors = get_translational_vectors_under_pbc(phase, longest)
    tv_0 = np.where(np.linalg.norm(translational_vectors, axis=-1) == 0.0)[0][
        0
    ]
    xyz_cartesian = get_cartesian_xyz_under_pbc(xyz, g, translational_vectors)

    r_vectors = (
        mesh_cartesian[:, np.newaxis, np.newaxis]
        - xyz_cartesian[np.newaxis, :, :]
    )
    r = np.linalg.norm(r_vectors, axis=-1)
    r[:, atom_index, tv_0] = np.nan

    # ratio between typical distance and actual distance
    sigma_r = sigma[:, :, np.newaxis] / r
    sigma_r[np.isnan(sigma_r)] = 0
    sigma_r = sigma_r.reshape(rj.size, -1)

    pair_potential = sigma_r**12.0 - sigma_r**6.0
    pair_potential = np.sum(pair_potential, axis=-1)
    return pair_potential


def get_charge_neutrality_for_mesh(
    phase, atom_index, mesh_cartesian, longest_threshold
):
    xyz = phase.model["xyz"]
    xyz = np.tile(xyz, (mesh_cartesian.shape[0], 1, 1))
    xyz[:, atom_index] = mesh_cartesian
    g = phase.model["lattice_vector"].T

    ri = get_radii_for_all_sites(phase)
    rj = np.full(mesh_cartesian.shape[0], ri[atom_index])
    sigma = rj[:, np.newaxis] + ri[np.newaxis, :]
    longest = np.amax(ri + ri[atom_index]) * longest_threshold

    translational_vectors = get_translational_vectors_under_pbc(phase, longest)

    xyz_cartesian = np.dot(g, xyz.reshape(-1, 3).T).T.reshape(xyz.shape)
    xyz_cartesian = (
        xyz_cartesian[:, :, np.newaxis]
        + translational_vectors[np.newaxis, np.newaxis, :]
    )  # shape: mesh, xyz, tv
    xyz_cartesian_tv0 = xyz_cartesian[:, :, 0]

    r_vectors = (
        xyz_cartesian[:, :, np.newaxis, :]
        - xyz_cartesian_tv0[:, np.newaxis, :, np.newaxis]
    )
    r = np.linalg.norm(r_vectors, axis=-1)  # shape: mesh, xyz, xyz, tv
    ri_all = np.tile(
        ri, (mesh_cartesian.shape[0], translational_vectors.shape[0], 1)
    )
    ri_all = np.transpose(ri_all, (0, 2, 1))
    sigma = (
        ri_all[:, :, np.newaxis, :] + ri[np.newaxis, np.newaxis, :, np.newaxis]
    )

    # effective bond weight for i = exp(1-(length_i/expected_length)^6)
    effective_bond_weight = np.exp(1 - (r / sigma) ** 6.0)
    effective_bond_weight[r == 0] = 0

    ci = phase.model["charge"]
    ci_all = np.tile(
        ci, (mesh_cartesian.shape[0], translational_vectors.shape[0], 1)
    )
    ci_all = np.transpose(ci_all, (0, 2, 1))
    cc1 = (
        ci_all[:, :, np.newaxis, :] * ci[np.newaxis, np.newaxis, :, np.newaxis]
    )
    cc2 = (
        np.abs(ci_all)[:, :, np.newaxis, :]
        + np.abs(ci)[np.newaxis, np.newaxis, :, np.newaxis]
    )

    use = np.where((cc1 < 0) | (cc2 == 0))
    ebw = np.zeros_like(effective_bond_weight)
    ebw[use] = effective_bond_weight[use]

    # coulomb potential with charge screening by the counter charge srounding the center atom
    effective_bond_number = np.sum(
        np.sum(ebw, axis=-1), axis=-1
    )  # shape: mesh, xyz
    no_bond = np.where(effective_bond_number < 0.01)
    effective_bond_number[no_bond] = np.nan
    charge_per_bond = ci / effective_bond_number
    charge_per_bond[no_bond] = 0.0

    fraction_of_charge = (
        ebw * (charge_per_bond)[:, :, np.newaxis, np.newaxis]
    )  # shape: mesh, xyz, xyz, tv
    received_charge = np.sum(
        fraction_of_charge, axis=-1
    )  # shape: mesh, xyz, xyz
    received_charge = np.sum(received_charge, axis=-2)  # shape: mesh, xyz

    charge_neutrality = np.abs(ci + received_charge)  # shape: mesh, xyz
    return charge_neutrality


def get_charge_neutrality_for_mesh_vesta(
    phase, atom_index, mesh_cartesian, longest_threshold
):
    xyz = phase.model["xyz"]
    g = phase.model["lattice_vector"].T

    ri = get_radii_for_all_sites(phase)
    sigma = ri + ri[atom_index]
    longest = np.amax(sigma) * longest_threshold
    translational_vectors = get_translational_vectors_under_pbc(phase, longest)

    xyz = np.dot(g, xyz.T).T
    translational_vectors = get_translational_vectors_under_pbc(
        phase, np.amax(sigma) * longest_threshold
    )
    r_vectors = (
        -mesh_cartesian[:, np.newaxis, np.newaxis]
        + xyz[np.newaxis, :, np.newaxis]
        + translational_vectors[np.newaxis, np.newaxis, :]
    )  # shape: mesh, xyz, tv
    r = np.linalg.norm(r_vectors, axis=-1)
    # r = r/sigma.reshape(-1, ri.size, 1) #normalization

    charge = phase.model["charge"]
    cc1 = (
        np.full(mesh_cartesian.shape[0], charge[atom_index])[:, np.newaxis]
        * charge[np.newaxis, :]
    ).reshape(mesh_cartesian.shape[0], np.size(charge), 1)
    r = np.where(cc1 >= 0, 0, r).reshape((mesh_cartesian.shape[0], -1))

    r_nan = np.where(r != 0, r, np.nan)
    r_min = np.nanmin(r_nan, axis=-1).reshape(-1, 1)
    r_av = np.nansum(
        r_nan * np.exp(1 - np.power(r_nan / r_min, 6)), axis=-1
    ) / np.nansum(np.exp(1 - np.power(r_nan / r_min, 6)), axis=-1)
    r_av = r_av.reshape(-1, 1)

    bond_weight = np.exp(1 - np.power(r_nan / r_av, 6))
    bond_weight[np.isnan(bond_weight)] = 0.0
    effective_bond_number = np.sum(bond_weight, axis=-1)
    bond_weight = bond_weight.reshape(
        (mesh_cartesian.shape[0], xyz.shape[0], translational_vectors.shape[0])
    )
    fraction_of_charge = (
        bond_weight
        / effective_bond_number.reshape(-1, 1, 1)
        * charge.reshape(1, -1, 1)
    )
    received_charge = np.sum(np.sum(fraction_of_charge, axis=-1), axis=-1)
    charge_neutrality = np.abs(charge[atom_index] + received_charge)

    return charge_neutrality


def get_distortion_index_for_mesh_vesta(
    phase, atom_index, mesh_cartesian, longest_threshold
):
    xyz = phase.model["xyz"]
    g = phase.model["lattice_vector"].T

    ri = get_radii_for_all_sites(phase)
    sigma = ri + ri[atom_index]
    longest = np.amax(sigma) * longest_threshold
    translational_vectors = get_translational_vectors_under_pbc(phase, longest)

    xyz = np.dot(g, xyz.T).T
    translational_vectors = get_translational_vectors_under_pbc(
        phase, np.amax(sigma) * longest_threshold
    )
    r_vectors = (
        -mesh_cartesian[:, np.newaxis, np.newaxis]
        + xyz[np.newaxis, :, np.newaxis]
        + translational_vectors[np.newaxis, np.newaxis, :]
    )  # shape: mesh, xyz, tv
    r = np.linalg.norm(r_vectors, axis=-1)
    r = r / sigma.reshape(-1, ri.size, 1)  # normalization
    r[r > longest_threshold] = 0.0

    charge = phase.model["charge"]
    charge_mesh = np.full(mesh_cartesian.shape[0], charge[atom_index])
    cc1 = (charge_mesh[:, np.newaxis] * charge[np.newaxis, :]).reshape(
        mesh_cartesian.shape[0], np.size(charge), 1
    )
    cc2 = (
        np.abs(charge_mesh)[:, np.newaxis] + np.abs(charge)[np.newaxis, :]
    ).reshape(mesh_cartesian.shape[0], np.size(charge), 1)
    r = np.where((cc1 < 0) | (cc2 == 0), r, 0)
    r = r.reshape((mesh_cartesian.shape[0], -1))

    r_nan = np.where(r != 0, r, np.nan)
    r_ave = np.nanmean(r_nan, axis=-1).reshape(-1, 1)
    distortion_index = np.abs(r_nan - r_ave) / r_ave
    distortion_index = np.nanmean(distortion_index, axis=-1)

    return distortion_index


def find_pair_potential_minimum_for_atom(
    phase, atom_index, max_move_radius=0.1, longest_threshold=2.0
):
    mesh_cartesian = make_mesh_in_cartesian(phase, atom_index, max_move_radius)
    pair_potential = get_pair_potential_for_mesh(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    minimum_xyz = mesh_cartesian[np.argmin(pair_potential)]

    g = phase.model["lattice_vector"].T
    minimum_xyz = np.dot(np.linalg.inv(g), minimum_xyz)

    return minimum_xyz


def find_charge_neutrality_minimum_for_atom(
    phase, atom_index, max_move_radius=0.1, longest_threshold=2.0
):
    mesh_cartesian = make_mesh_in_cartesian(phase, atom_index, max_move_radius)
    charge_neutrality = get_charge_neutrality_for_mesh(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    minimum_xyz = mesh_cartesian[np.argmin(np.sum(charge_neutrality, axis=-1))]
    g = phase.model["lattice_vector"].T
    minimum_xyz = np.dot(np.linalg.inv(g), minimum_xyz)
    return minimum_xyz


def find_charge_neutrality_minimum_for_atom_vesta(
    phase, atom_index, max_move_radius=0.1, longest_threshold=2.0
):
    mesh_cartesian = make_mesh_in_cartesian(phase, atom_index, max_move_radius)
    charge_neutrality = get_charge_neutrality_for_mesh_vesta(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    minimum_xyz = mesh_cartesian[np.argmin(charge_neutrality)]
    g = phase.model["lattice_vector"].T
    minimum_xyz = np.dot(np.linalg.inv(g), minimum_xyz)
    return minimum_xyz


def find_charge_neutrality_minimum_for_atom_without_collision(
    phase, atom_index, max_move_radius=0.1, longest_threshold=2.0
):
    mesh_cartesian = make_mesh_in_cartesian(phase, atom_index, max_move_radius)
    pair_potential = get_pair_potential_for_mesh(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    mesh_cartesian = mesh_cartesian[np.where(pair_potential < 0.0)]

    if np.size(mesh_cartesian) == 0:
        return None

    charge_neutrality = get_charge_neutrality_for_mesh(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    minimum_xyz = mesh_cartesian[np.argmin(charge_neutrality)]
    g = phase.model["lattice_vector"].T
    minimum_xyz = np.dot(np.linalg.inv(g), minimum_xyz)

    return minimum_xyz


def find_charge_neutrality_minimum_for_atom_without_collision_vesta(
    phase, atom_index, max_move_radius=0.1, longest_threshold=2.0
):
    mesh_cartesian = make_mesh_in_cartesian(phase, atom_index, max_move_radius)
    pair_potential = get_pair_potential_for_mesh(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    mesh_cartesian = mesh_cartesian[np.where(pair_potential < 0.0)]

    if np.size(mesh_cartesian) == 0:
        return None

    charge_neutrality = get_charge_neutrality_for_mesh_vesta(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    minimum_xyz = mesh_cartesian[np.argmin(charge_neutrality)]
    g = phase.model["lattice_vector"].T
    minimum_xyz = np.dot(np.linalg.inv(g), minimum_xyz)

    return minimum_xyz


def find_distortion_index_minimum_for_atom_without_collision_vesta(
    phase, atom_index, max_move_radius=0.1, longest_threshold=2.0
):
    mesh_cartesian = make_mesh_in_cartesian(phase, atom_index, max_move_radius)
    pair_potential = get_pair_potential_for_mesh(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    mesh_cartesian = mesh_cartesian[np.where(pair_potential < 0.0)]

    if np.size(mesh_cartesian) == 0:
        return None

    distortion_index = get_distortion_index_for_mesh_vesta(
        phase, atom_index, mesh_cartesian, longest_threshold
    )
    minimum_xyz = mesh_cartesian[np.nanargmin(distortion_index)]
    g = phase.model["lattice_vector"].T
    minimum_xyz = np.dot(np.linalg.inv(g), minimum_xyz)

    return minimum_xyz


def find_symmtric_xyz_under_fixed_lattice_vector(phase, symprec):
    lattice = phase["lattice", "vector"]
    positions = np.copy(phase.lattice_sites["xyz"])
    site_index = np.copy(phase.lattice_sites["site_index"])

    cell = (lattice, positions, site_index)
    dataset = spglib.get_symmetry_dataset(
        cell, symprec=symprec, angle_tolerance=-1.0
    )

    if dataset == None:
        return False, phase["xyz"]

    symop_r = dataset["rotations"]
    symop_t = dataset["translations"]

    xyz = positions[np.unique(site_index, return_index=True)[-1]]
    N = xyz.shape[0]
    Nt = symop_r.shape[0]
    site_index = np.arange(0, N, 1)
    xyz -= np.floor(xyz)

    xyz_all = np.transpose(np.dot(symop_r, xyz.T), (0, 2, 1))
    xyz_all += np.reshape(np.tile(symop_t, (1, N)), (-1, N, 3))
    xyz_all -= np.floor(xyz_all)
    xyz_all = xyz_all.reshape(-1, 3)

    xyz_all, use = phase.remove_duplicate(xyz_all)
    site_index = np.tile(site_index, Nt).flatten()[use]
    new_xyz = xyz_all[np.unique(site_index, return_index=True)[-1]]
    return True, new_xyz


def get_closest_distance_for_atom_type(phase, atom):
    info = ai()
    symbol, charge = info.parse_symbol(atom)
    rj = info.get_radius(symbol=symbol, charge=charge)
    ri = get_radii_for_all_sites(phase)
    closest_distance = ri + rj
    return closest_distance


def get_radii_for_all_sites(phase):
    symbols = phase.model["symbol"]
    radii = np.ones(np.array(symbols).size)

    symbols, index, inverse = np.unique(
        symbols, return_index=True, return_inverse=True
    )
    info = ai()
    for n, atom in enumerate(symbols):
        symbol, charge_str = info.parse_symbol(atom)
        try:
            charge = float(phase.model["charge"][index[n]])
        except:
            charge = charge_str
        radii[inverse == n] = info.get_radius(symbol=symbol, charge=charge)
    return radii


def get_translational_vectors_under_pbc(phase, longest_distance):
    lattice = phase.model["lattice"]
    pbc = phase.model["pbc"]
    g = phase.model["lattice_vector"].T

    extension = np.linalg.norm(np.sum(g, axis=0))
    vectors = (
        np.abs(
            np.floor(np.full(3, longest_distance) / lattice[:3]).astype(int)
            + 1
        )
        * pbc
    )
    cell_vectors = np.indices(vectors * 2 + 1).T.reshape(-1, 3) - vectors
    cell_vectors = cell_vectors.astype(float)

    cartesian_vectors = np.dot(g, cell_vectors.T).T
    vector_distance = np.linalg.norm(cartesian_vectors, axis=1)
    use = np.where(vector_distance <= longest_distance + extension)

    translational_vectors = np.round(cartesian_vectors[use], 7)
    return translational_vectors


def get_cartesian_xyz_under_pbc(xyz, g, translational_vectors):
    xyz = np.dot(g, xyz.T).T
    xyz = xyz[:, np.newaxis] + translational_vectors[np.newaxis, :]
    return xyz  # shape = len(symbol), len(tv), 3


def make_grid_points(phase, mesh_size):
    mesh_size = np.array(mesh_size).flatten()
    lattice = phase.model["lattice"][:3]
    grid_points = (lattice // mesh_size).astype(int)
    x = np.linspace(0.0, 1.0, grid_points[0] + 1)[:-1]
    y = np.linspace(0.0, 1.0, grid_points[1] + 1)[:-1]
    z = np.linspace(0.0, 1.0, grid_points[2] + 1)[:-1]
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False)
    xyz = np.concatenate(
        [
            xx[:, :, :, np.newaxis],
            yy[:, :, :, np.newaxis],
            zz[:, :, :, np.newaxis],
        ],
        axis=-1,
    ).reshape(-1, 3)
    return xyz


def make_sphere_points(radius):  # return cartesian
    base = np.linspace(-radius, radius, 25)
    xx, yy, zz = np.meshgrid(base, base, base, sparse=False)
    xyz = np.concatenate(
        [
            xx[:, :, :, np.newaxis],
            yy[:, :, :, np.newaxis],
            zz[:, :, :, np.newaxis],
        ],
        axis=-1,
    ).reshape(-1, 3)
    xyz = xyz[np.where(np.linalg.norm(xyz, axis=-1) < radius)]
    return xyz
