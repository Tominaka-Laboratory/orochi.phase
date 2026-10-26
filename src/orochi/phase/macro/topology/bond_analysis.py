import numpy as np

from ...database.atoms import AtomInformation as ai
from ...database.bond_valence import BondValence as bv
from ..analysis.site_relation import get_all_sites_relationship
from ..analysis.coordination import get_effective_coordination_number

# from sklearn.cluster import DBSCAN

###Definition####
# site neighbors: 0, none; 1, attractive; 2, repulsive; 3, neutral
# node neighbors: number of attractive, repulsive, neutral neighbors
# connectivity: corner, edge, face
# edge shared_atom_type: attractive, repulsive, neutral
# charge_charge: 1, attractive, 2, repulsive, 3, zero zero, 0, others


def get_node_edge_information_for_topology_analysis(phase):
    cng = get_coordination_network_graph(phase)

    graph = {}
    graph["node"] = {}
    graph["edge"] = {}

    # unify the equivalent nodes into the same number
    neighbors = cng["node"]["neighbors"]
    polyhedral_connectivity = cng["node"]["polyhedral_connectivity"]
    atomic_numbers = cng["node"]["atomic_number"]
    index = cng["node"]["index"]
    charge = cng["node"]["charge"]

    edge = cng["edge"]["edge"]
    shared_atom_type = cng["edge"]["shared_atom_type"]

    vectors = np.concatenate(
        [
            neighbors,
            polyhedral_connectivity,
            atomic_numbers.reshape(-1, 1),
            charge.reshape(-1, 1),
        ],
        axis=-1,
    )
    eq_matrix = (
        np.linalg.norm(
            vectors[np.newaxis, :, :] - vectors[:, np.newaxis, :], axis=-1
        )
        == 0
    )
    eq_index_all = np.argmax(eq_matrix, axis=-1)
    eq_index = np.unique(eq_index_all)
    counts = np.sum(eq_matrix, axis=-1)

    graph["node"]["index"] = index[eq_index]
    graph["node"]["atomic_numbers"] = atomic_numbers[eq_index]
    graph["node"]["charge"] = charge[eq_index]
    graph["node"]["counts_in_cell"] = counts[eq_index]
    graph["node"]["neighbor_shell"] = neighbors[eq_index]
    graph["node"]["polyhedral_connectivity"] = polyhedral_connectivity[
        eq_index
    ]

    # unify the equivalent nodes into the same number for edges
    if len(edge) == 0:
        return graph

    new_edge = np.empty((0, 2))
    for id in eq_index:
        ed = edge[edge[:, 0] == id]
        new_edge = np.append(new_edge, ed, axis=0)

    new_edge_eq_index = np.zeros_like(new_edge)
    for new_node_id, id in enumerate(eq_index):
        eq_node_numbers = index[eq_index_all == id]
        for eq_node in eq_node_numbers:
            new_edge_eq_index[new_edge == eq_node] = new_node_id

    new_edge = new_edge_eq_index
    vec = np.sort(new_edge, axis=-1)
    # vec = np.concatenate([edge.reshape(-1,2), shared_atom_type], axis=-1)

    eq_matrix = (
        np.linalg.norm(vec[np.newaxis, :, :] - vec[:, np.newaxis, :], axis=-1)
        == 0
    )
    eq_index = np.argmax(eq_matrix, axis=-1)
    eq_index = np.unique(eq_index)
    counts = np.sum(eq_matrix, axis=-1)

    graph["edge"]["pairs"] = new_edge[eq_index]
    # graph["edge"]["shared_atom_type"] = shared_atom_type[eq_index]
    graph["edge"]["counts_for_node"] = counts[eq_index]

    return graph


def get_coordination_network_graph(phase, bandwidth=0.1):

    model, info = get_all_sites_relationship(phase)
    r_c = np.where(model["cell_index"] == 0)

    # making node
    node = {}
    node["neighbors"] = np.empty((0, 2))
    node["polyhedral_connectivity"] = np.empty((0, 3))

    node["index"] = model["site_index_p1"][r_c]
    node["site"] = model["site_index"][r_c]
    node["atomic_number"] = model["atomic_number"][r_c]
    node["charge"] = model["charge"][r_c]
    node["multiplicity"] = model["multiplicity"][r_c]

    # making edge
    edge = {}
    edge["edge"] = np.empty((0, 2))
    edge["shared_atom_type"] = np.empty((0, 2))

    def bond_normalization_with_shortest_bond(bond):
        zero_to_100 = np.where(bond != 0, bond, 100.0)
        bond_min = np.min(zero_to_100, axis=-1).reshape(-1, 1)
        res = bond / bond_min
        return res

    attractive_neighbors = info["distance_ratio"] * np.where(
        info["charge_charge"] == 1, 1.0, 0.0
    )
    repulsive_neighbors = info["distance_ratio"] * np.where(
        info["charge_charge"] == 2, 1.0, 0.0
    )
    neutral_neighbors = info["distance_ratio"] * np.where(
        info["charge_charge"] == 3, 1.0, 0.0
    )

    attractive_neighbors = bond_normalization_with_shortest_bond(
        attractive_neighbors
    )
    repulsive_neighbors = bond_normalization_with_shortest_bond(
        repulsive_neighbors
    )
    neutral_neighbors = bond_normalization_with_shortest_bond(
        neutral_neighbors
    )

    site_uc = model["index"][r_c]
    charge_uc = model["charge"][r_c]
    for n, center1 in enumerate(site_uc):
        if charge_uc[n] != 0:
            clusters = one_dimensional_clustering_with_kernel_estimation(
                attractive_neighbors[center1], bandwidth
            )
            shell_1 = model["index"][np.where(clusters == 1)]

            clusters = one_dimensional_clustering_with_kernel_estimation(
                repulsive_neighbors[center1], bandwidth
            )
            shell_2 = model["index"][np.where(clusters == 1)]

        else:
            clusters = one_dimensional_clustering_with_kernel_estimation(
                neutral_neighbors[center1], bandwidth
            )
            shell_1 = model["index"][np.where(clusters == 1)]
            shell_2 = model["index"][np.where(clusters == 2)]

        neighbors = np.array([np.size(shell_1), np.size(shell_2)])
        node["neighbors"] = np.append(
            node["neighbors"], neighbors.reshape(-1, 2), axis=0
        )

        if np.size(shell_2) == 0:
            node["polyhedral_connectivity"] = np.append(
                node["polyhedral_connectivity"], np.array([[0, 0, 0]]), axis=0
            )
            continue

        connectivity = np.array([0, 0, 0])  # corner, edge, face
        for m, center2 in enumerate(shell_2):
            if charge_uc[n] != 0:
                clusters = one_dimensional_clustering_with_kernel_estimation(
                    attractive_neighbors[center2], bandwidth
                )
                shell_1_b = model["index"][np.where(clusters == 1)]

                clusters = one_dimensional_clustering_with_kernel_estimation(
                    repulsive_neighbors[center2], bandwidth
                )
                shell_2_b = model["index"][np.where(clusters == 1)]
            else:
                clusters = one_dimensional_clustering_with_kernel_estimation(
                    neutral_neighbors[center2], bandwidth
                )
                shell_1_b = model["index"][np.where(clusters == 1)]
                shell_2_b = model["index"][np.where(clusters == 2)]

            shared_1_shell = np.size(np.intersect1d(shell_1, shell_1_b))
            shared_2_shell = np.size(np.intersect1d(shell_2, shell_2_b))

            shared_atom_type = np.array([shared_1_shell, shared_2_shell])
            if np.sum(shared_atom_type) == 0:
                continue

            if shared_1_shell == 1:
                connectivity += np.array([1, 0, 0])
            elif shared_1_shell == 2:
                connectivity += np.array([0, 1, 0])
            elif shared_1_shell > 2:
                connectivity += np.array([0, 0, 1])

            center2_in_uc = model["site_index_p1"][center2]
            node_id_2 = np.where(site_uc == center2_in_uc)[0][0]
            edge["edge"] = np.append(
                edge["edge"], np.array([[n, node_id_2]]).astype(int), axis=0
            )
            edge["shared_atom_type"] = np.append(
                edge["shared_atom_type"],
                shared_atom_type.reshape(-1, 2),
                axis=0,
            )  # 1st shell, 2nd shell

        node["polyhedral_connectivity"] = np.append(
            node["polyhedral_connectivity"],
            connectivity.reshape(-1, 3),
            axis=0,
        )  # corner, edge, face

    graph = {}
    graph["node"] = node
    graph["edge"] = edge
    return graph


def get_coordination_shell(phase, order=6, bandwidth1=0.1, bandwidth2=0.05):

    model, info = get_all_sites_relationship(phase)
    r_c = np.where(model["cell_index"] == 0)

    # making node
    node = {}
    # node["polyhedral_connectivity"] = np.empty((0,3))
    unique, unique_id = np.unique(
        model["site_index"][r_c], return_index=True
    )

    # node["index"] = model["site_index_p1"][r_c][unique_id]
    node["site"] = model["site_index"][r_c][unique_id]
    node["atomic_number"] = model["atomic_number"][r_c][unique_id]
    node["charge"] = model["charge"][r_c][unique_id]
    node["multiplicity"] = model["multiplicity"][r_c][unique_id]

    attractive_neighbors = info["distance_ratio"] * np.where(
        info["charge_charge"] == 1, 1.0, 0.0
    )
    neutral_neighbors = info["distance_ratio"] * np.where(
        info["charge_charge"] == 3, 1.0, 0.0
    )

    shell = []

    node["ECON_Center"] = []
    node["ECON_Shell"] = []
    node["ECON_Extend"] = []
    node["Shell"] = []

    site_uc = model["index"][r_c][unique_id]
    charge_uc = model["charge"][r_c][unique_id]
    for n, center1 in enumerate(site_uc):
        if charge_uc[n] != 0:
            length1 = attractive_neighbors[center1]
        else:
            length1 = neutral_neighbors[center1]

        econ, indices = get_effective_coordination_number(
            length1, order=order, bandwidth=bandwidth1, cluster_number=0
        )
        node["ECON_Center"] += [econ]
        shell += [model["site_index"][indices]]

        length2 = info["distance_ratio"][indices][:, indices]
        econ, indices = get_effective_coordination_number(
            length2, order=order, bandwidth=bandwidth2, cluster_number=0
        )
        node["ECON_Shell"] += [econ / np.size(shell[-1])]

        Shell_Bond = np.zeros_like(length2).flatten()
        Shell_Bond[indices] = 1
        Shell_Bond = Shell_Bond.reshape(length2.shape)

        Share = Shell_Bond[:, np.newaxis] + Shell_Bond
        Share = np.sum(np.where(Share == 2, 1, 0), axis=-1)
        Share[Shell_Bond == 1] = 0
        np.fill_diagonal(Share, 0)

        count_zero = np.average(np.sum(Share == 0, axis=1))
        count_one = np.average(np.sum(Share == 1, axis=1))
        count_two = np.average(np.sum(Share == 2, axis=1))

        node["Shell"] += [[count_zero, count_one, count_two]]

    for n, center1 in enumerate(site_uc):
        shell2 = 0
        for id in shell[n]:
            shell2 += node["ECON_Center"][int(id)]
        shell2 /= np.size(shell[n])
        node["ECON_Extend"] += [shell2]

    graph = {}
    graph["node"] = node
    # graph["edge"] = edge
    return graph["node"]




# other iterms==================================================


def analyze_nearest_neighbors(
    phase, tolerance=[0.8, 1.4]
):  # should be maintained in future 20231012 Tominaka
    def get_all_sites():
        x = np.arange(-1, 2)
        xx, yy, zz = np.meshgrid(x, x, x)
        translational_vectors = np.concatenate(
            [xx.reshape(-1, 1).T, yy.reshape(-1, 1).T, zz.reshape(-1, 1).T]
        ).T
        base_sites = phase.lattice_sites

        vector_length = np.linalg.norm(translational_vectors, axis=-1)
        translational_vectors = translational_vectors[
            np.argsort(vector_length)
        ]
        model = phase.make_translated_model(base_sites, translational_vectors)
        model["index"] = np.arange(0, model["xyz"].shape[0], 1).astype(
            int
        )
        return model

    def get_actual_distance(
        model,
    ):  # xyz1: all atoms in the lattice (cell_index=0), xyz2: all the atoms of the adjacent cells
        xyz = model["xyz"][np.where(model["cell_index"] == 0)]
        g = model["lattice_vector"]
        xyz1 = np.dot(g, xyz.T).T

        xyz = model["xyz"]
        xyz2 = np.dot(g, xyz.T).T

        # axis0: atom index in the lattice; axis1: index in the all sites
        actual_distance = np.sqrt(
            np.sum(
                np.square(xyz1[:, np.newaxis] - xyz2[np.newaxis, :]), axis=-1
            )
        )
        return actual_distance

    def get_expected_distance(model):  # isolated atomic/ionic radius
        symbol1 = model["symbol"][np.where(model["cell_index"] == 0)]
        symbol2 = model["symbol"]

        # expected distance
        radius1 = np.ones(np.array(symbol1).size)
        radius2 = np.ones(np.array(symbol2).size)

        symbols, symbol_index = np.unique(symbol1, return_index=True)
        for n, atom in enumerate(symbols):
            info = ai()
            info.read(atom)
            info.charge = float(phase.par["charge"][symbol_index[n]])
            r = info.get_radius()

            index = np.where(symbol1 == atom)
            radius1[index] = r
            index = np.where(symbol2 == atom)
            radius2[index] = r

        expected_distance = radius1[:, np.newaxis] + radius2[np.newaxis, :]
        return expected_distance

    def get_charge_charge_info(model):
        charge1 = model["charge"][np.where(model["cell_index"] == 0)]
        charge2 = model["charge"]

        charge_charge = charge1[:, np.newaxis] * charge2[np.newaxis, :]
        charge_charge[np.where(charge_charge < 0)] = -1
        charge_charge[np.where(charge_charge > 0)] = 1

        return charge_charge

    def get_shannon_distance(model, coordination_numbers):
        symbol1 = model["symbol"][np.where(model["cell_index"] == 0)]
        charge1 = model["charge"][np.where(model["cell_index"] == 0)]
        symbol2 = model["symbol"]
        site_index1 = model["site_index"][
            np.where(model["cell_index"] == 0)
        ]
        site_index2 = model["site_index"]

        radius1 = np.zeros(np.array(symbol1).size)
        radius2 = np.zeros(np.array(symbol2).size)

        sn, index = np.unique(site_index1, return_index=True)

        for s in sn:
            n = np.where(s == site_index1)
            cn = float(coordination_numbers[n][0])
            if cn == 0:
                continue

            atom = symbol1[n][0]
            charge = float(charge1[n][0])
            info = ai()
            info.read(atom)
            info.charge = charge
            info.bond_number = cn

            r = info.get_shannon_radius()
            if r == 0.0:
                r = info.get_ionic_radius()

            radius1[n] = r
            m = np.where(s == site_index2)
            radius2[m] = r

        r = radius1[:, np.newaxis] + radius2[np.newaxis, :]
        return r

    def get_effective_charge(actual_distance, shortest_distance, attractive):
        Lav_exp = (
            np.exp(
                1.0
                - np.power(
                    actual_distance / shortest_distance.reshape(-1, 1), 6.0
                )
            )
            * attractive
        )
        Lav = np.sum(actual_distance * Lav_exp, axis=1) / np.sum(
            Lav_exp, axis=1
        )
        bond_weight = (
            np.exp(1.0 - np.power(actual_distance / Lav.reshape(-1, 1), 6.0))
            * attractive
        )
        bond_weight_sum = np.sum(bond_weight, axis=-1)

        r = np.where(bond_weight_sum != 0.0)
        bond_weight[r] /= bond_weight_sum[r]

        effective_charge = model["charge"] * bond_weight
        return effective_charge

    def get_coordination_information(actual_distance, model, tolerance):

        for key in ["attractive", "repulsive", "neutral"]:
            select = charge_charge_select[key]

            if np.sum(select) == 0:
                continue

            else:
                d = actual_distance * select
                d[d == 0.0] = 100.0
                shortest_bond = np.amin(d, axis=-1)

                within_shortest_bond = np.where(
                    actual_distance.T - shortest_bond * tolerance < 0.0,
                    1.0,
                    0.0,
                ).T
                use = select * within_shortest_bond * exclude_phase_pairs

                coordination_information[key] = {}
                coordination_number = np.sum(use, axis=-1)
                coordination_information[key][
                    "coordination_number"
                ] = coordination_number
                d1 = actual_distance * use

                if key in ["attractive", "repulsive"]:
                    d2 = (
                        get_shannon_distance(model, coordination_number)
                        * use
                    )
                else:
                    d2 = expected_distance * use

                coordination_information[key]["distance"] = []
                coordination_information[key]["expected_distance"] = []
                coordination_information[key]["ratio"] = []

                for id in range(d1.shape[0]):
                    actual = d1[id][np.nonzero(d1[id])]
                    expected = d2[id][np.nonzero(d2[id])]
                    coordination_information[key]["distance"] += [actual]
                    coordination_information[key]["expected_distance"] += [
                        expected
                    ]
                    coordination_information[key]["ratio"] += [
                        actual / expected
                    ]

                if key in ["attractive"]:

                    Lav_exp = (
                        np.exp(
                            1.0
                            - np.power(
                                d1 / np.amin(d, axis=1).reshape(-1, 1), 6.0
                            )
                        )
                        * use
                    )
                    Lav = np.sum(d1 * Lav_exp, axis=1) / np.sum(
                        Lav_exp, axis=1
                    )
                    bond_weight = (
                        np.exp(1.0 - np.power(d1 / Lav.reshape(-1, 1), 6.0))
                        * use
                    )
                    bond_weight_sum = np.sum(bond_weight, axis=-1)

                    neighbor_site_index = np.tile(
                        model["site_index"],
                        (np.size(phase.par["symbol"]), 1),
                    )

                    for site_index, value in enumerate(bond_weight_sum):
                        r2 = np.where(neighbor_site_index == site_index)
                        bond_weight[r2] /= value

                    effective_charge = charge2 * bond_weight
                    coordination_information[key]["effective_charge"] = (
                        np.sum(effective_charge, axis=-1) * -1.0
                    )

        return coordination_information

    info = {}
    model = get_all_sites()
    # info["edge1"], info["edge2"] = get_edges(model)
    info["actual_distance"] = get_actual_distance(model)
    info["expected_distance"] = get_expected_distance(model)
    info["charge"] = get_charge_charge_info(model)

    info["ratio"] = info["actual_distance"] / info["expected_distance"]
    info["neighbors"] = np.where(
        (info["ratio"] > tolerance[0]) & (info["ratio"] < tolerance[1]),
        1.0,
        0.0,
    )

    nearest_distance = np.amin(
        info["actual_distance"][info["actual_distance"] != 0]
    )
    info["2nd_neighbors"] = np.where(
        (info["actual_distance"] > nearest_distance * tolerance[1])
        & (info["actual_distance"] < nearest_distance * tolerance[1] * 2),
        1.0,
        0.0,
    )

    attractive = np.where(info["charge"] < 0, 1, 0)
    info["coordination_number"] = np.sum(
        info["neighbors"] * attractive, axis=-1
    )
    info["shannon_distance"] = get_shannon_distance(
        model, info["coordination_number"]
    )

    r = np.where(info["shannon_distance"] != 0.0)
    info["expected_distance"][r] = info["shannon_distance"][r]

    # shortest_distance = np.amin(actual_distance * neighbors * attractive, axis=-1)
    # effective_charge = get_effective_charge(actual_distance, shortest_distance, attractive)

    return model, info


def get_node_edge_information_for_topology_analysis_old(
    phase,
):  # other functions might be useful

    def topology_graph(cng):

        neighbors = cng["node"]["neighbors"]
        polyhedral_connectivity = cng["node"]["polyhedral_connectivity"]
        atomic_numbers = cng["node"]["atomic_number"]
        index = cng["node"]["index"]
        charge = cng["node"]["charge"]

        edge = cng["edge"]["edge"]
        shared_atom_type = cng["edge"]["shared_atom_type"]

        graph = {}
        graph["node"] = {}
        graph["edge"] = {}

        vectors = np.concatenate(
            [
                neighbors,
                polyhedral_connectivity,
                atomic_numbers.reshape(-1, 1),
                charge.reshape(-1, 1),
            ],
            axis=-1,
        )
        eq_matrix = (
            np.linalg.norm(
                vectors[np.newaxis, :, :] - vectors[:, np.newaxis, :], axis=-1
            )
            == 0
        )
        eq_index = np.argmax(eq_matrix, axis=-1)
        eq_index = np.unique(eq_index)
        counts = np.sum(eq_matrix, axis=-1)

        graph["node"]["index"] = index[eq_index]
        graph["node"]["atomic_numbers"] = atomic_numbers[eq_index]
        graph["node"]["charge"] = charge[eq_index]
        graph["node"]["counts_in_cell"] = counts[eq_index]
        graph["node"]["neighbor_type"] = neighbors[eq_index]
        graph["node"]["polyhedral_connectivity"] = polyhedral_connectivity[
            eq_index
        ]

        for n, id in enumerate(index):
            eqv = index[eq_matrix[n]]
            if eqv.size > 1:
                m = np.amin(eqv)
                if id == m:
                    for eq in eqv:
                        edge[edge == eq] = m

        if len(edge) == 0:
            return graph

        edge = np.sort(edge, axis=-1)
        vec = np.concatenate([edge.reshape(-1, 2), shared_atom_type], axis=-1)

        eq_matrix = (
            np.linalg.norm(
                vec[np.newaxis, :, :] - vec[:, np.newaxis, :], axis=-1
            )
            == 0
        )
        eq_index = np.argmax(eq_matrix, axis=-1)
        eq_index = np.unique(eq_index)
        counts = np.sum(eq_matrix, axis=-1)

        graph["edge"]["pairs"] = edge[eq_index]
        graph["edge"]["shared_atom_type"] = shared_atom_type[eq_index]
        graph["edge"]["counts_in_cell"] = counts[eq_index]
        return graph

    def graph_formation_space_group(model, info):
        id1 = model["site_index"][np.where(model["cell_index"] == 0)]
        e1, index = np.unique(id1, return_index=True)

        id2 = model["site_index"]
        edge2 = np.tile(id2.reshape(-1, 1), id1.shape[0]).T
        edge2[np.where(info["neighbors"] == 0)] = -1
        d = info["actual_distance"]

        def node_formation_space_ground(model, info):
            node = {}
            symb = model["symbol"][np.where(model["cell_index"] == 0)][
                index
            ]
            symbols = np.unique(symb)
            atomic_numbers = np.zeros(symb.size).astype(int)

            for atom in symbols:
                atom_info = ai()
                atom_info.read(atom)
                atomic_numbers[np.where(atom == symb)] = atom_info.number

            node["index"] = e1
            node["atomic_number"] = atomic_numbers
            node["charge"] = model["charge"][
                np.where(model["cell_index"] == 0)
            ][index]
            node["multiplicity"] = model["multiplicity"][
                np.where(model["cell_index"] == 0)
            ][index]
            return node

        def edge_formation_space_ground(model, info):
            edge = {}
            edge["edge"] = np.empty((0, 2))
            edge["counts"] = np.empty(0)
            edge["distance"] = np.empty(0)

            for id in index:
                e2 = edge2[id]
                e2, use, counts = np.unique(
                    e2, return_index=True, return_counts=True
                )
                r = np.where(e2 >= 0)
                e2 = e2[r]

                e1e2 = np.concatenate(
                    [
                        np.full(e2.size, e1[id]).reshape(-1, 1),
                        e2.reshape(-1, 1),
                    ],
                    axis=-1,
                )
                edge["edge"] = np.append(edge["edge"], e1e2, axis=0)
                edge["counts"] = np.append(edge["counts"], counts[r])
                edge["distance"] = np.append(edge["distance"], d[id][use][r])

            return edge

        graph = {}
        graph["node"] = node_formation_space_ground(model, info)
        graph["edge"] = edge_formation_space_ground(model, info)
        return graph

    def graph_formation_unit_cell(model, info):
        e1 = model["site_index_p1"][np.where(model["cell_index"] == 0)]
        id2 = model["site_index_p1"]
        edge2 = np.tile(id2.reshape(-1, 1), e1.shape[0]).T
        edge2[np.where(info["neighbors"] == 0)] = -1
        d = info["actual_distance"]

        id3 = model["cell_index"]
        edge3 = np.tile(id3.reshape(-1, 1), e1.shape[0]).T

        def node_formation_space_ground(model, info):
            node = {}
            symb = model["symbol"][np.where(model["cell_index"] == 0)]
            symbols = np.unique(symb)
            atomic_numbers = np.zeros(symb.size).astype(int)

            for atom in symbols:
                atom_info = ai()
                atom_info.read(atom)
                atomic_numbers[np.where(atom == symb)] = atom_info.number

            node["index"] = e1
            node["atomic_number"] = atomic_numbers
            node["charge"] = model["charge"][
                np.where(model["cell_index"] == 0)
            ]
            # node["multiplicity"] = model["multiplicity"][np.where(model["cell_index"] == 0)]
            return node

        def edge_formation_space_ground(model, info):
            edge = {}
            edge["edge"] = np.empty((0, 2))
            edge["counts"] = np.empty(0)
            edge["distance"] = np.empty(0)
            edge["cell"] = np.empty(0)

            for id, e2 in enumerate(edge2):
                e2, use, counts = np.unique(
                    e2, return_index=True, return_counts=True
                )
                r = np.where(e2 >= 0)
                e2 = e2[r]

                e1e2 = np.concatenate(
                    [
                        np.full(e2.size, e1[id]).reshape(-1, 1),
                        e2.reshape(-1, 1),
                    ],
                    axis=-1,
                )
                edge["edge"] = np.append(edge["edge"], e1e2, axis=0)
                edge["counts"] = np.append(edge["counts"], counts[r])
                edge["distance"] = np.append(edge["distance"], d[id][use][r])
                edge["cell"] = np.append(edge["cell"], edge3[id][use][r])
            return edge

        graph = {}
        graph["node"] = node_formation_space_ground(model, info)
        graph["edge"] = edge_formation_space_ground(model, info)
        return graph

    cng = get_coordination_network_graph(phase)
    tpg = topology_graph(cng)
    return tpg
