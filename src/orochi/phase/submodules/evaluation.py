import numpy as np
from ..database.atoms import AtomInformation as ai

avogadro = 6.022 * 10**23  # mol-1
elemental_charge = 1.602 * 10**-19  # C
permittivity0 = 8.854 * 10**-12  # C2/Jm
bolzman = 1.38065 * 10**-23  # J/K

B = avogadro * elemental_charge**2.0 / (4.0 * np.pi * permittivity0)
epsilon = 120 * bolzman  # J; assuming a constant value

def site_scores(self, 
                polyhedral=True, 
                overlap=True, 
                nearest=True,
                longest_threshold=1.3, 
                include_coulomb=True,
                neutrality_mode='shannon', *args, **kwargs):
    """
    Master evaluation function that computes multiple site-wise physical 
    and chemical scores in a single pass.

    Parameters
    ----------
    longest_threshold : float, optional
        Multiplicative factor for Shannon radii to define the interaction cutoff.
    include_coulomb : bool, optional
        If True, adds the electrostatic potential (LJC). If False, calculates 
        only the distance-based pair potential (LJ).
    neutrality_mode : str, 'shannon' or 'vesta'
        'shannon': Base bond weights on the sum of Shannon radii.
        'vesta': Base bond weights on local average distances (VESTA definition).

    Returns
    -------
    dict
        A dictionary containing ndarrays for each requested metric.
    """

    if np.size(self["charge"]) != np.size(self["symbol"]):
        self.get_formal_charge()
        self()
    if self._state != "DONE":
        self()

    charge = self.model["charge"]
    site_indices = self.model["site_index"] 
    n_asu = np.max(site_indices) + 1

    # 1. Compute pairwise distance matrix and ionic radii once
    r, sigma, ri = _get_neighbor_distances(self, longest_threshold)
    results = {}

    # 2. Compute requested metrics reusing common r and sigma
    if polyhedral:
        results['distortion'], results['neutrality'], results['econ'] = _calc_polyhedral_scores(r, sigma, ri, charge, neutrality_mode)
    if overlap:
        results['overlap'] = _calc_overlap(r, sigma, ri, charge, include_coulomb)
    if nearest: # normalized nearest distance
        results['nearest'] = _calc_distance_ratio(r, sigma, ri)

    # 3. Average onto asymmetric unit (ASU) basis by site_index
    final_results = {}
    for key, p1_values in results.items():
        sums = np.bincount(site_indices, weights=p1_values, minlength=n_asu)
        counts = np.bincount(site_indices, minlength=n_asu)
        final_results[key] = np.divide(sums, counts, out=np.zeros_like(sums), where=counts!=0)

    return final_results


def _calc_overlap(r, sigma, ri, charge, include_coulomb=True):
    """
    Calculate the potential energy for each site based on Lennard-Jones (12-6) 
    and optionally Coulombic interactions.

    This function serves as a physical validity filter. The LJ term detects 
    atomic overlaps (repulsion), while the Coulomb term evaluates electrostatic 
    stability.

    """
    r_sq = np.where(r != 0, r, np.nan)
    # Adjust collision distance (sigma) so the potential minimum aligns with ri + rj
    s_r = (sigma.reshape(ri.size, ri.size, 1) / (2**(1/6))) / r_sq
    
    # Lennard-Jones (12-6) term
    potential = 4.0 * epsilon * (np.nan_to_num(s_r**12 - s_r**6))
    
    if include_coulomb:
        cc = (charge[:, np.newaxis] * charge[np.newaxis, :]).reshape(ri.size, ri.size, 1)
        potential += B * (cc / r_sq)
        
    return np.nansum(potential.reshape(ri.size, -1), axis=-1)

def _calc_distance_ratio(r, sigma, ri):
    """
    Calculate the distance ratio, r/sigma.

    """
    r_sq = np.where(r != 0, r, np.nan)
    s_r = r_sq/(sigma.reshape(ri.size, ri.size, 1))
        
    return np.nanmin(s_r.reshape(ri.size, -1), axis=-1)


def _calc_polyhedral_scores(r, sigma, ri, charge, mode='vesta'):
    """
    Compute site-wise polyhedral distortion, charge neutrality, and ECoN using bond weights.
    """
    # 1. Filter by charge relationship (attractive cc1 <= 0 or neutral)
    cc1 = (charge[:, np.newaxis] * charge[np.newaxis, :]).reshape(ri.size, ri.size, 1) # site, site, tv
    r_filtered = np.where(cc1 <= 0, r, 0)
    r_nan = np.where(r_filtered != 0, r_filtered, np.nan)

    # --- Compute bond weights according to mode ---
    if mode == 'shannon':
        # Based on ideal Shannon radii sum
        bond_weight = np.exp(1 - (r_nan / sigma.reshape(ri.size, ri.size, -1)) ** 6.0)
    elif mode == 'vesta':
        # Based on local average bond distance (VESTA definition)
        r_nan = r_nan.reshape(ri.size, -1)
        r_min = np.nanmin(r_nan, axis=-1)
        w_av = np.exp(1 - np.power(r_nan / r_min, 6))
        r_av = np.nansum(r_nan * w_av, axis=-1) / np.nansum(w_av, axis=-1)
        bond_weight = np.exp(1 - np.power(r_nan / r_av, 6))
        bond_weight = bond_weight.reshape(ri.size, ri.size, -1)

    # Zero out self-interaction (distance == 0)
    bond_weight[r_filtered == 0] = 0.0
    bond_weight = np.nan_to_num(bond_weight)
    bond_weight = bond_weight.reshape(ri.size, -1)
    r_filtered = r_filtered.reshape(ri.size, -1)

    # 3. Effective Coordination Number (ECoN)
    econ = np.sum(bond_weight, axis=-1)

    # 4. Weighted average distance (r_ave)
    r_weighted_sum = np.sum(bond_weight * r_filtered, axis=-1)
    r_ave = np.divide(r_weighted_sum, econ, out=np.zeros_like(econ), where=econ!=0)
    
    # 5. Weighted distortion index
    r_diff = np.abs(r_filtered - r_ave[:, np.newaxis])
    r_rel_diff = np.divide(r_diff, r_ave[:, np.newaxis], out=np.zeros_like(r_diff), where=r_ave[:, np.newaxis]!=0)
    distortion = np.sum(bond_weight * r_rel_diff, axis=-1) / np.where(econ!=0, econ, 1.0)

    # 6. Charge neutrality (received_charge)
    fraction_of_charge = (bond_weight / np.where(econ!=0, econ, 1.0).reshape(-1, 1) * charge.reshape(-1, 1))
    fraction_of_charge = fraction_of_charge.reshape(ri.size, ri.size, -1)
    received_charge = np.sum(np.sum(fraction_of_charge, axis=-1).T, axis=-1)
    neutrality = np.abs(charge + received_charge)
    return distortion, neutrality, econ


def _get_neighbor_distances(self, longest_threshold=1.3):
    """
    Compute actual distance matrix (r) and ideal distance matrix (sigma) for all sites.
    """
    ri = _get_radii_for_all_sites(self)
    sigma = ri[:, np.newaxis] + ri[np.newaxis, :] # shape = site, site
    
    g = self.model["lattice_vector"].T
    xyz = self.model["xyz"]
    xyz_cart = np.dot(g, xyz.T).T
    
    # Obtain translation vectors covering maximum interaction range
    max_sigma = np.amax(sigma)
    translational_vectors = _get_translational_vectors_under_pbc(self, max_sigma * longest_threshold)
    
    # Compute pairwise distances (xyz, xyz, tv)
    r_vectors = (-xyz_cart[:, np.newaxis, np.newaxis] + 
                 xyz_cart[np.newaxis, :, np.newaxis] + 
                 translational_vectors[np.newaxis, np.newaxis, :])
    r = np.linalg.norm(r_vectors, axis=-1) # shape = site, site, tv

    return r, sigma, ri

def _get_translational_vectors_under_pbc(self, longest_distance):
    """
    Generate Cartesian translational vectors for neighbor cell searching under PBC.

    This function calculates the integer cell offsets required to cover a 
    spherical search volume of 'longest_distance' and converts them into 
    Cartesian vectors using the lattice matrix. It respects the 'pbc' flags 
    defined in the model.

    Parameters
    ----------
    longest_distance : float
        The radius within which neighbor atoms are evaluated.

    Returns
    -------
    numpy.ndarray
        An array of Cartesian translation vectors (N_offsets, 3).
    """

    lattice = self.model["lattice"]
    pbc = self.model["pbc"]
    g = self.model["lattice_vector"].T

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

def _get_radii_for_all_sites(self):
    """
    Retrieve atomic or ionic radii for all sites, correctly handling mixed valence 
    by identifying unique (symbol, charge) pairs.
    """
    self()
    symbols = np.array(self.model["symbol"])
    charges = np.array(self.model["charge"])
    radii = np.zeros(symbols.size)

    # 1. Identify unique (symbol, charge) pairs
    combined = np.stack((symbols, charges), axis=-1)
    unique_pairs, inverse = np.unique(combined, axis=0, return_inverse=True)
    
    # 2. Retrieve radius for each unique (symbol, charge) pair
    info = ai()
    for n, pair in enumerate(unique_pairs):
        atom_symbol = pair
        try:
            atom_charge = int(float(pair[1]))
        except (ValueError, TypeError):
            atom_charge = 0

        radius = info.get_radius(symbol=atom_symbol[0], charge=atom_charge)
        # 3. Assign radius to all matching sites
        radii[inverse == n] = radius

    return radii
