import numpy as np

def feasibility(self, 
                scores={}, 
                thresholds={'overlap': 100, 'distortion': 0.1, 'neutrality': 0.2},
                *args, **kwargs):
    """
    Analyze site_scores results and generate a natural-language diagnostic report,
    including comparison between Effective Coordination Number (ECoN) and typical coordination numbers.
    """
    problematic_sites = []

    # 1. Prepare typical coordination numbers (Typical CN) for comparison
    if "econ" in scores and np.size(self["cn"]) == np.shape(self["xyz"])[0]:
        typical_cns = self["cn"]
    else:
        typical_cns = None

    # 2. Evaluate scores for each site in the asymmetric unit (ASU)
    for i in range(len(self["symbol"])):
        issues = []
        symbol = self["symbol"][i]
        
        # Physical overlap (LJC Potential): hard constraint
        if "overlap" in scores and scores['overlap'][i] > thresholds['overlap']:
            issues.append(f"Atomic overlap (Score: {scores['overlap'][i]:.1f})")
            
        # Geometric distortion (Polyhedral Distortion)
        if "distortion" in scores and scores['distortion'][i] > thresholds['distortion']:
            issues.append(f"High polyhedral distortion (Index: {scores['distortion'][i]:.3f})")
            
        # Chemical charge balance (Charge Neutrality)
        if "neutrality" in scores and scores['neutrality'][i] > thresholds['neutrality']:
            issues.append(f"Charge imbalance (Delta: {scores['neutrality'][i]:.2f})")

        # 3. Coordination environment check (ECoN vs Typical CN)
        if "econ" in scores and typical_cns is not None:
            calculated_econ = scores['econ'][i]
            expected_cn = typical_cns[i]
            if abs(calculated_econ - expected_cn) > 1.0:
                status = "Under-coordinated" if calculated_econ < expected_cn else "Over-coordinated"
                issues.append(f"{status} (ECoN: {calculated_econ:.1f}, Expected: {expected_cn})")

        if issues:
            problematic_sites.append(f"- Site {i} ({symbol}): " + ", ".join(issues))

    # 4. Aggregate report
    if not problematic_sites:
        return "Structural validation: All atomic sites are physically, geometrically, and chemically reasonable based on site-wise potentials and coordination environments."

    return "Structural Issues Found:\n" + "\n".join(problematic_sites)

def structure(self, precision=3, *args, **kwargs):
    """
    Convert the Phase asymmetric-unit structure into a Markdown report.
    """
    # 1. Retrieve basic crystallographic information
    sg_symbol = self["space_group", "value"]
    sg_no = self["space_group", "number"]
    lattice = self["lattice"]
    abc = [round(v, precision) for v in lattice[:3]]
    angles = [round(v, precision) for v in lattice[3:]]
    pbc = self["pbc"]
    if np.sum(pbc) == 3:
        periodic_boundary = "Along all three axes"
    elif np.sum(pbc) == 0:
        periodic_boundary = "None"
    else:
        axes = np.array(["a", "b", "c"])
        pbc_axes = axes[pbc]
        periodic_boundary = "Along " + ", ".join(pbc_axes) + " axes"
    
    # 2. Build header information
    shape = self["shape_type"]
    if shape == "analytical":
        shape_info = "Analytical (Bulk structure with an envelope function)"
        shape_info += "\nSpherical particle diameters: " + ", ".join([f"{v:.{precision}f}" for v in self["sphere", "value"]])
    else:
        shape_info = f"Explicitly Modeled ({shape}) - Accounts for surface termination effects."

    lines = [
        f"### Structural Model (Asymmetric Unit)",
        f"- **Space Group**: {sg_symbol} (#{sg_no})",
        f"- **Lattice**: a={abc[0]}, b={abc[1]}, c={abc[2]} Å, alpha={angles[0]}, beta={angles[1]}, gamma={angles[2]}°",
        f"- **Periodic Boundary Conditions**: {periodic_boundary}",
        f"- **Particle Shape**: {shape_info}",
        ""
    ]
    
    # 3. Build atomic site table (including ADP and Site Symmetry)
    header = "| ID | Symbol | Wyckoff | Site Sym. | x (frac) | y (frac) | z (frac) | Occ | ADP ($U_{iso}$) | Charge |"
    sep    = "|----|--------|---------|-----------|---|---|---|-----|-----------------|--------|"
    lines.extend([header, sep])
    
    for i in range(len(self["symbol"])):
        xyz = [f"{v:.{precision}f}" for v in self["xyz"][i]]
        sym = self["symbol"][i]
        wyck = self["wyckoff_letters"][i]
        site_sym = self["site_symmetry_symbols"][i]
        occ = f"{self['occ'][i]:.2f}"
        adp = f"{self['adp'][i]:.{precision}f}"
        charge = f"{self['charge'][i]:.1f}"
        
        lines.append(f"| {i} | {sym} | {wyck} | {site_sym} | {xyz[0]} | {xyz[1]} | {xyz[2]} | {occ} | {adp} | {charge} |")
        
    return "\n".join(lines)