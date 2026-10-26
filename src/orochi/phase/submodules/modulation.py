import numpy as np


def commensurate_cosine_ab_plane(phase):
    """
    Applies a simple cosine wave modulation to the z-coordinates relative to the ab-plane.
    
    This function adds a vertical displacement to the z-coordinate based on the x and y 
    positions. Unlike the thickness-preserving version, it does not adjust for the 
    surface normal, so the effective layer thickness may decrease on steep slopes.

    Parameters
    ----------
    phase : dict
        A dictionary containing the phase information.
        Expected keys:
            - "lattice": list or tuple, [a, b, c, alpha, beta, gamma]
            - "xyz": np.ndarray, shape (N, 3), fractional coordinates
            - ("modulation", "value"): list or tuple, [amplitude_angstrom, ...]
              * Only the first value (amplitude) is used.

    Returns
    -------
    np.ndarray
        The new modulated fractional coordinates with shape (N, 3).
    """
    # Unpack cell lengths (c is needed for normalization)
    a, b, c = phase["lattice"][:3]
    
    # Extract amplitude in Angstroms and convert to fractional
    # Assumes phase["modulation", "value"] is a tuple/list: [amplitude_angstrom, ...]
    amplitude_frac = phase["modulation", "value"][0] / c
    
    xyz = phase["xyz"]

    # Calculate modulation (simple vertical displacement)
    z_modulated = amplitude_frac * np.cos(2 * np.pi * xyz[:, 0]) * np.cos(2 * np.pi * xyz[:, 1])
    
    # Apply modulation
    xyz_modulated = xyz.copy()
    xyz_modulated[:, 2] += z_modulated
    
    return xyz_modulated

def commensurate_cosine_ab_plane_with_preserving_thickness(phase):
    """
    Applies a cosine wave modulation to the structure while preserving the 
    layer thickness by displacing atoms along the surface normal vector.

    Parameters
    ----------
    phase : dict
        A dictionary containing the phase information.
        Expected keys:
            - "lattice": list or tuple, [a, b, c, alpha, beta, gamma]
            - "xyz": np.ndarray, shape (N, 3), fractional coordinates
            - ("modulation", "value"): list or tuple, [amplitude_angstrom, center_z_frac]
              * center_z_frac is optional (default: 0.5)

    Returns
    -------
    np.ndarray
        The new modulated fractional coordinates with shape (N, 3).
    """
    # Unpack cell lengths
    a, b, c = phase["lattice"][:3]
    
    # Extract modulation parameters
    # Assumes phase["modulation", "value"] is a tuple/list: [amplitude_angstrom, center_z_frac]
    mod_params = phase["modulation", "value"]
    amp_angstrom = mod_params[0]
    
    # Use the second value as center_z if available, otherwise default to 0.5
    if len(mod_params) > 1:
        center_z_frac = mod_params[1]
    else:
        center_z_frac = 0.5
    
    # 1. Extract original fractional coordinates
    # Using specific variables for clarity
    xyz = phase["xyz"]
    u = xyz[:, 0] # x_frac
    v = xyz[:, 1] # y_frac
    w = xyz[:, 2] # z_frac
    
    # 2. Calculate the "vertical" distance of each atom from the center plane
    # This represents the layer half-thickness in Angstroms.
    d_thickness = (w - center_z_frac) * c
    
    # 3. Calculate gradients of the wave surface in Real Space
    # Surface function Z_surf = amp_angstrom * cos(2pi*u) * cos(2pi*v)
    
    # Pre-calculate common terms for derivative
    # d/dX (cos(2pi*u)) = -2pi/a * sin(2pi*u)
    common_factor = -amp_angstrom * 2 * np.pi
    
    # dZ/dX (Slope along a-axis in real space)
    dz_dx = (common_factor / a) * np.sin(2 * np.pi * u) * np.cos(2 * np.pi * v)
    
    # dZ/dY (Slope along b-axis in real space)
    dz_dy = (common_factor / b) * np.cos(2 * np.pi * u) * np.sin(2 * np.pi * v)
    
    # 4. Compute the Unit Normal Vector
    # The normal vector n = (-dz_dx, -dz_dy, 1) normalized.
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(u) # Z-component is always 1 before normalization
    
    # Normalization
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm
    
    # 5. Calculate new positions in Real Space
    # Base surface height at position (u, v) in Angstroms (relative to center)
    z_base_angstrom = amp_angstrom * np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v)
    
    # New X_real = (Original X on surface) + (Displacement along Normal_x)
    new_x_ang = (u * a) + (d_thickness * nx)
    new_y_ang = (v * b) + (d_thickness * ny)
    
    # New Z_real = (Center Z) + (Wave Height) + (Displacement along Normal_z)
    new_z_ang = (center_z_frac * c) + z_base_angstrom + (d_thickness * nz)
    
    # 6. Convert back to Fractional Coordinates
    new_frac_coords = np.zeros_like(xyz)
    new_frac_coords[:, 0] = new_x_ang / a
    new_frac_coords[:, 1] = new_y_ang / b
    new_frac_coords[:, 2] = new_z_ang / c
    
    return new_frac_coords