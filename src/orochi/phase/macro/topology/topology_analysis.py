from orochi.phase.macro.topology.bond_analysis import (
    get_node_edge_information_for_topology_analysis as topology,
)
from orochi.phase.macro.topology import bond_analysis
from orochi.phase.macro import typical_phase
import numpy as np


def cif_to_topology(cif_file, pbc=[True, True, True]):
    phase = typical_phase.bulk(cif_file)
    phase.par["pbc"] = pbc
    phase.run()

    phase.transform_into_standard_cell()
    phase.get_atom_information()
    phase.analyze_material_type()
    phase.get_formal_charge()

    graph = topology(phase)
    return graph


def cif_to_topology_2(
    cif_file, material_type: list, pbc=[True, True, True], max_atom=1000
):
    phase = typical_phase.bulk(cif_file)
    phase.par["pbc"] = pbc
    phase.run()

    if phase.model["xyz"].shape[0] > max_atom:
        return False, None

    # from orochi.phase.function import viewer

    # viewer.view_lattice(phase)
    phase.transform_into_standard_cell()
    phase.run()
    # viewer.view_lattice(phase)

    phase.get_atom_information()
    phase.analyze_material_type()

    if phase.par["material_type"] in material_type:
        phase.get_formal_charge()
        graph = topology(phase)
        node = graph["node"]
        use_information = {}
        use_information["primary_vector"] = np.append(
            node["neighbor_shell"], node["polyhedral_connectivity"], axis=-1
        )
        use_information["secondary_vector"] = np.append(
            np.sign(node["charge"].reshape(-1, 1)),
            node["counts_in_cell"].reshape(-1, 1)
            / np.sum(node["counts_in_cell"]),
            axis=-1,
        )

        edge = graph["edge"]
        if "pairs" in edge:
            use_information["link"] = np.append(
                edge["pairs"], edge["counts_for_node"].reshape(-1, 1), axis=-1
            )

        return True, use_information

    else:
        return False, None


def cif_to_topology_3(
    cif_file, material_type: list, pbc=[True, True, True], max_atom=100
):
    phase = typical_phase.bulk(cif_file)
    phase.par["pbc"] = pbc
    # phase.run()

    if phase.model["xyz"].shape[0] > max_atom:
        return False, None

    # from orochi.phase.function import viewer

    # viewer.view_lattice(phase)
    phase.transform_into_standard_cell()
    # phase.run()
    # viewer.view_lattice(phase)

    phase.get_atom_information()
    phase.analyze_material_type()

    if phase.par["material_type"] in material_type:
        phase.get_formal_charge()
        result = bond_analysis.get_coordination_shell(
            phase, order=6.0, bandwidth1=0.15, bandwidth2=0.15 / 2
        )

        number_of_node = np.size(result["site"])
        first_information = np.concatenate(
            [
                np.reshape(result["ECON_Center"], (number_of_node, 1)),
                np.reshape(result["ECON_Shell"], (number_of_node, 1)),
                np.reshape(result["ECON_Extend"], (number_of_node, 1)),
                np.reshape(result["Shell"], (number_of_node, -1)),
            ],
            axis=-1,
        )

        second_information = np.concatenate(
            [
                np.reshape(result["charge"], (number_of_node, 1)),
                np.reshape(
                    result["multiplicity"] / np.sum(result["multiplicity"]),
                    (number_of_node, 1),
                ),
            ],
            axis=-1,
        )
        print("1st information for node clustering")
        print(
            "effective coordination number, Ave ECON within shell atoms, ave ECON of shell atoms, Link within shell atoms (0,1,2 sharing)"
        )
        print(first_information)

        print("2nd information for crystal structure clustering")
        print("charge, proportion")
        print(second_information)
