import numpy as np
from ...phase import Phase
from typing import Dict
from numpy.typing import NDArray


# def get_site_relationship(phase: Phase, threshold: float = 1.4) -> Dict[str, NDArray[np.float64]]:
#     """
#     Analyzes the relationships between atomic sites in a crystal structure.

#     This function calculates and returns information about the distances, charge
#     relations, and coordination between atomic sites within a crystal structure
#     defined by a Phase object.  It calculates the actual distance between each pair
#     of sites, compares it to the expected distance (based on atomic radii),
#     and determines the charge relationship.  It then uses this information to
#     identify coordinating sites based on a distance threshold.

#     Parameters
#     ----------
#     phase : Phase
#         A Phase object containing the structural information, including atomic
#         coordinates ('xyz'), atomic radii ('radius'), and atomic charges ('charge').
#     threshold : float, optional
#         The distance ratio threshold for considering two sites as coordinating.
#         The ratio is calculated as actual distance / expected distance.
#         Default is 1.4.

#     Returns
#     -------
#     dict
#         A dictionary containing the following keys:
#         - "actual_distance": A NumPy array of pairwise distances between atoms.
#         - "expected_distance": A NumPy array of expected pairwise distances
#           (based on atomic radii).
#         - "charge_relation": A NumPy array indicating the product of charges for each
#           atom pair (positive for repulsion, negative for attraction, 0 for neutral).
#         - "coordinating": A dictionary with keys "attractive" and "neutral".
#           Each value is a NumPy array representing the coordination:
#           1 indicates coordination, 0 indicates no coordination.

#     """
#     info = {}
#     xyz = phase.model["xyz"]
#     g = phase.model["lattice_vector"]
#     xyz_cartesian = np.dot(g, xyz.T).T  # Convert to Cartesian coordinates

#     # Calculate pairwise distances
#     info["actual_distance"] = np.sqrt(
#         np.sum(np.square(xyz_cartesian[:, np.newaxis] - xyz_cartesian[np.newaxis, :]), axis=-1)
#     )

#     radius = phase.model["radius"]
#     charge = phase.model["charge"]
#     info["expected_distance"] = radius[:, np.newaxis] + radius[np.newaxis, :]
#     info["charge_relation"] = charge[:, np.newaxis] * charge[np.newaxis, :]

#     distance_ratio = info["actual_distance"] / info["expected_distance"]

#     # Identify coordinating sites based on distance and charge
#     info["coordinating"] = {}
#     info["coordinating"]["attractive"] = np.where(
#         (info["charge_relation"] < 0)
#         & (distance_ratio < threshold)
#         & (info["actual_distance"] > 0.0),
#         1.0,
#         0.0,
#     )
#     info["coordinating"]["neutral"] = np.where(
#         (info["charge_relation"] == 0)
#         & (distance_ratio < threshold)
#         & (info["actual_distance"] > 0.0),
#         1.0,
#         0.0,
#     )
#     return info




# def get_all_sites_relationship(phase):
#     def get_all_sites():
#         x = np.arange(-1, 2)
#         xx, yy, zz = np.meshgrid(x, x, x)
#         translational_vectors = np.concatenate(
#             [xx.reshape(-1, 1).T, yy.reshape(-1, 1).T, zz.reshape(-1, 1).T]
#         ).T
#         base_sites = phase.lattice_sites
#         base_sites["atomic_number"] = get_atomic_numbers_from_symbols(
#             base_sites["symbol"]
#         )

#         vector_length = np.linalg.norm(translational_vectors, axis=-1)
#         translational_vectors = translational_vectors[
#             np.argsort(vector_length)
#         ]
#         model = phase.make_translated_model(base_sites, translational_vectors)
#         model["index"] = np.arange(0, model["xyz"].shape[0], 1).astype(
#             int
#         )
#         return model

#     def get_atomic_numbers_from_symbols(symb):
#         symbols = np.unique(symb)
#         atomic_numbers = np.zeros(symb.size).astype(int)

#         for atom in symbols:
#             atom_info = ai()
#             atom_info.read(atom)
#             atomic_numbers[np.where(atom == symb)] = atom_info.number
#         return atomic_numbers

#     def get_actual_distance(model):
#         xyz = model["xyz"]
#         g = model["lattice_vector"]
#         xyz = np.dot(g, xyz.T).T
#         actual_distance = np.sqrt(
#             np.sum(np.square(xyz[:, np.newaxis] - xyz[np.newaxis, :]), axis=-1)
#         )
#         return actual_distance

#     def get_expected_distance(model):  # isolated atomic/ionic radius
#         symbol = model["symbol"]
#         charge = model["charge"]
#         radius = np.zeros(np.array(symbol).size)

#         symbols, symbol_index = np.unique(symbol, return_index=True)
#         charge = charge[symbol_index]
#         for n, atom in enumerate(symbols):
#             info = ai()
#             info.read(atom)
#             info.charge = float(charge[n])
#             radius[np.where(symbol == atom)] = info.get_radius()

#         expected_distance = radius[:, np.newaxis] + radius[np.newaxis, :]
#         return expected_distance

#     def get_charge_charge_info(model):
#         charge = model["charge"]
#         charge1 = np.tile(charge, (np.size(charge), 1))
#         charge2 = charge1.T
#         charge_charge = np.full_like(charge1, "neutral", dtype=object)

#         charge_charge[np.where(charge1 * charge2 < 0)] = "attractive"
#         charge_charge[np.where(charge1 * charge2 > 0)] = "repulsive"
#         charge_charge[np.where((charge1 == 0) & (charge2 == 0))] = "neutral"
#         return charge_charge

#     info = {}
#     model = phase.model #get_all_sites()
#     print(model["symbol"].shape)
#     info["actual_distance"] = get_actual_distance(model)
#     info["expected_distance"] = get_expected_distance(model)
#     info["charge_relation"] = get_charge_charge_info(model)
#     # info["distance_ratio"] = (
#     #     info["actual_distance"] / info["expected_distance"]
#     # )
#     return model, info
