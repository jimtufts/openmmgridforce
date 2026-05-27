"""Internal-coordinate machinery for the hybrid entropy pipeline.

Tier 3 of hessian_plan.md. Provides:
  - find_rotatable_torsions(system, barrier_threshold_kjmol)
        Identify rotatable torsions from the FF (PeriodicTorsionForce).
        Dedupes Fourier-expansion terms by central bond.
  - compute_torsion_b_vector(positions_nm, indices)
        Blondel-Karplus per-torsion B-vector (∂φ/∂x_α) for a 4-tuple.
        Returns shape (3*N_total,) in units of 1/nm.
  - compute_pitzer_reduced_moment(masses_amu, B)
        Wilson G-matrix diagonal: I_r = 1 / Σ_α B²_α / m_α (amu·nm²).
  - project_out_torsions(H_mw, b_vectors_mw)
        QR projection of mass-weighted Hessian onto orthogonal complement.
  - vibrational_frequencies(H_projected, n_torsions, is_bound)
        Diagonalize and split rigid-body / torsion / vibrational eigvals.

Convention: positions in nm, masses in amu (daltons). I_r returned in amu·nm²
which is the natural unit when combining with kJ/(mol·rad²) curvature.
"""
import numpy as np

try:
    from openmm import PeriodicTorsionForce, HarmonicBondForce
    from openmm import unit as omm_unit
except ImportError:
    PeriodicTorsionForce = None
    HarmonicBondForce = None
    omm_unit = None

try:
    import gridforceplugin as _gfp
    _IsolatedBondedForce = _gfp.IsolatedBondedForce
except Exception:
    _IsolatedBondedForce = None


def _iter_torsion_params(force):
    """Yield (i, j, k, l, periodicity, phase_rad, k_kjmol) for any supported
    force class (stock PeriodicTorsionForce or plugin IsolatedBondedForce)."""
    if isinstance(force, PeriodicTorsionForce):
        for t in range(force.getNumTorsions()):
            i, j, k, l, n, phase, k_t = force.getTorsionParameters(t)
            yield (i, j, k, l, int(n),
                   float(phase.value_in_unit(omm_unit.radian)),
                   float(k_t.value_in_unit(omm_unit.kilojoule_per_mole)))
    elif _IsolatedBondedForce is not None and isinstance(force, _IsolatedBondedForce):
        for t in range(force.getNumTorsions()):
            i, j, k, l, n, phase_rad, k_kjmol = force.getTorsionParameters(t)
            yield (int(i), int(j), int(k), int(l),
                   int(n), float(phase_rad), float(k_kjmol))


def _iter_bond_params(force):
    """Yield (i, j) pairs from stock HarmonicBondForce or plugin IsolatedBondedForce."""
    if isinstance(force, HarmonicBondForce):
        for b in range(force.getNumBonds()):
            i, j, _, _ = force.getBondParameters(b)
            yield (int(i), int(j))
    elif _IsolatedBondedForce is not None and isinstance(force, _IsolatedBondedForce):
        for b in range(force.getNumBonds()):
            i, j, _, _ = force.getBondParameters(b)
            yield (int(i), int(j))


def _resolve_force_iter(source):
    """Yield force objects from either a System or a single Force.

    NOTE on SWIG downcasting: when a plugin Force subclass (e.g. IsolatedBondedForce)
    is added to a System and later retrieved via System.getForce(i), cross-module
    SWIG returns it as a base openmm.Force without the subclass identity, so
    isinstance() fails. Callers holding a direct Python reference to the plugin
    force should pass it here directly rather than the System.
    """
    if hasattr(source, 'getNumForces'):
        for fi in range(source.getNumForces()):
            yield source.getForce(fi)
    elif source is None:
        return
    elif isinstance(source, (list, tuple)):
        for f in source:
            yield f
    else:
        yield source


def find_rotatable_torsions(source, barrier_threshold_kjmol=30.0,
                              filter_ring_bonds=True):
    """Identify rotatable torsions, deduplicated by central bond (j, k).

    `source` is either an openmm.System or a Force (or list of Forces). Multiple
    Fourier terms on the same quartet sum into one entry per central bond.

    Ring bonds (central bond that is part of a ring) are filtered out by
    default because rigid rotation about a ring bond is not a valid 1D
    torsional coordinate. Detection: BFS from one endpoint excluding the
    candidate bond — if the other endpoint is reachable, it is a ring bond.
    """
    by_bond = {}  # (j, k) -> dict
    for force in _resolve_force_iter(source):
        for i, j, k, l, n, phase_rad, k_val in _iter_torsion_params(force):
            cb = (min(j, k), max(j, k))
            entry = by_bond.setdefault(cb, {
                'central_bond': cb,
                'representative_indices': (i, j, k, l),
                'fourier_terms': [],
                'total_barrier_kjmol': 0.0,
            })
            entry['fourier_terms'].append((n, phase_rad, k_val))
            entry['total_barrier_kjmol'] += 2.0 * abs(k_val)
    kept = [entry for entry in by_bond.values()
            if 0.1 < entry['total_barrier_kjmol'] < barrier_threshold_kjmol]
    if not filter_ring_bonds:
        return kept

    bond_list = extract_bond_list(source)
    adj_full = {}
    for a, b in bond_list:
        adj_full.setdefault(a, set()).add(b)
        adj_full.setdefault(b, set()).add(a)

    def _is_ring_bond(a, b):
        # BFS from b skipping the (a, b) edge; ring iff a reachable
        seen = {b}
        stack = [b]
        while stack:
            x = stack.pop()
            for y in adj_full.get(x, ()):
                if (x == b and y == a) or (x == a and y == b):
                    continue
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return a in seen

    return [e for e in kept if not _is_ring_bond(*e['central_bond'])]


def extract_bond_list(source):
    """Bond list (i, j) tuples extracted from any supported bonded force.

    `source` is either an openmm.System or a Force (or list of Forces)."""
    bonds = []
    for force in _resolve_force_iter(source):
        bonds.extend(_iter_bond_params(force))
    return bonds


def compute_torsion_b_vector(positions_nm, indices):
    """Blondel-Karplus B-vector for one torsion (∂φ/∂x_α).

    Args:
        positions_nm: (N, 3) atomic positions in nm.
        indices:      (i, j, k, l) tuple of atom indices.

    Returns:
        B: shape (3*N,) in 1/nm. Entries for atoms outside {i,j,k,l} are zero.

    Convention matches OpenMM's PeriodicTorsionForce dihedral angle.
    """
    pos = np.asarray(positions_nm)
    N = pos.shape[0]
    i, j, k, l = indices
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]
    m = np.cross(b1, b2)
    nv = np.cross(b2, b3)
    m_sq = float(np.dot(m, m))
    n_sq = float(np.dot(nv, nv))
    b2_sq = float(np.dot(b2, b2))
    if m_sq < 1e-20 or n_sq < 1e-20 or b2_sq < 1e-20:
        return np.zeros(3 * N)
    b2_norm = np.sqrt(b2_sq)
    G1 = m * (b2_norm / m_sq)
    G4 = nv * (-b2_norm / n_sq)
    alpha = float(np.dot(b1, b2)) / b2_sq
    beta = float(np.dot(b3, b2)) / b2_sq
    G2 = -(1.0 + alpha) * G1 + beta * G4
    G3 = alpha * G1 - (1.0 + beta) * G4
    B = np.zeros(3 * N)
    B[3*i:3*i+3] = G1
    B[3*j:3*j+3] = G2
    B[3*k:3*k+3] = G3
    B[3*l:3*l+3] = G4
    return B


def compute_pitzer_reduced_moment(masses_amu, B):
    """Wilson G-matrix diagonal reduced moment for ONE internal coordinate.

    I_r = 1 / G_ii  where  G_ii = Σ_α (∂φ/∂x_α)² / m_α

    NOTE: For a per-quartet dihedral B-vector this gives the reduced moment of
    *that 4-atom coordinate*, NOT the collective methyl-vs-methyl rotation
    used by Pitzer-Gwinn 1942. Use central_bond_reduced_moment for the
    physical Pitzer I_r of a rotatable bond.

    Args:
        masses_amu: (N,) atomic masses in amu (daltons).
        B:          (3*N,) B-vector in 1/nm.

    Returns:
        I_r in amu·nm² (multiply by 100 for amu·Å²).
    """
    masses_amu = np.asarray(masses_amu, dtype=float)
    inv_mass_3n = 1.0 / np.repeat(masses_amu, 3)
    G_ii = float(np.dot(B * B, inv_mass_3n))
    return 1.0 / G_ii if G_ii > 1e-30 else float('inf')


def _bfs_side(adj, start, blocked):
    """BFS through adj graph starting at start, never crossing blocked vertex."""
    seen = {start}
    queue = [start]
    while queue:
        v = queue.pop(0)
        for w in adj.get(v, ()):
            if w == blocked or w in seen:
                continue
            seen.add(w)
            queue.append(w)
    return seen


def central_bond_reduced_moment(positions_nm, masses_amu, central_bond,
                                  bond_list):
    """Pitzer (coupled-rotors) reduced moment of inertia for rotation about
    the j-k central bond:
        I_r = I_A * I_B / (I_A + I_B)
    where I_A = Σ_{α in side A} m_α * r⊥_α² (perpendicular distance from j-k axis).

    Args:
        positions_nm: (N, 3) atomic positions in nm.
        masses_amu:   (N,)   atomic masses in amu.
        central_bond: (j, k) tuple of atom indices.
        bond_list:    iterable of (i, j) bond tuples spanning the molecule.

    Returns:
        I_r in amu·nm² (multiply by 100 for amu·Å²).
    """
    pos = np.asarray(positions_nm, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    j, k = central_bond
    # Build adjacency from bond_list, excluding the central bond itself
    adj = {}
    for a, b in bond_list:
        if {a, b} == {j, k}:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    side_A = _bfs_side(adj, j, blocked=k)
    side_B = _bfs_side(adj, k, blocked=j)
    if not (side_A and side_B):
        return float('inf')

    axis = pos[k] - pos[j]
    axis /= np.linalg.norm(axis)

    def _moment(side):
        I = 0.0
        for alpha in side:
            v = pos[alpha] - pos[j]
            v_perp = v - np.dot(v, axis) * axis
            I += masses[alpha] * float(np.dot(v_perp, v_perp))
        return I

    I_A = _moment(side_A)
    I_B = _moment(side_B)
    if I_A == 0.0 and I_B == 0.0:
        return float('inf')
    if I_A == 0.0:
        return I_B
    if I_B == 0.0:
        return I_A
    return I_A * I_B / (I_A + I_B)


def project_out_torsions(H_mw, b_vectors_mw):
    """Project mass-weighted Cartesian Hessian onto subspace orthogonal
    to the torsion B-vectors.

    Args:
        H_mw:           (3N, 3N) mass-weighted Hessian (symmetric).
        b_vectors_mw:   (n_torsions, 3N) mass-weighted B-vectors (rows). Not
                        required to be orthonormal; QR handles it.

    Returns:
        H_projected: (3N, 3N) with the torsion subspace in its nullspace.
    """
    if b_vectors_mw is None or len(b_vectors_mw) == 0:
        return H_mw
    Q, _ = np.linalg.qr(np.asarray(b_vectors_mw).T)
    n_dof = H_mw.shape[0]
    P = np.eye(n_dof) - Q @ Q.T
    return P @ H_mw @ P


def vibrational_frequencies(H_projected, n_torsions, is_bound,
                             zero_tol_rel=1e-6):
    """Eigendecompose projected mass-weighted Hessian; classify modes.

    Args:
        H_projected:   (3N, 3N) projected mass-weighted Hessian.
        n_torsions:    number of torsion B-vectors projected out.
        is_bound:      True for bound state (rigid-body modes are vibrations now),
                       False for gas state (6 rigid-body modes are nullspace zeros).
        zero_tol_rel:  relative tolerance for the "zero" eigenvalue cutoff
                       (multiplied by max |eigenvalue|).

    Returns:
        dict with:
          'eigvals_mw':       all eigenvalues (sorted ascending)
          'frequencies_cm1':  signed frequencies for non-zero modes
          'omega_rad_s':      angular frequencies (rad/s) for positive modes
          'n_zero':           identified zero modes (rigid body + projected torsions)
          'n_imaginary':      modes with eigvalue below -tol
          'vibrational_omega_rad_s': positive non-zero, non-rigid frequencies
    """
    eigs = np.linalg.eigvalsh(0.5 * (H_projected + H_projected.T))
    eigs.sort()
    largest = max(np.max(np.abs(eigs)), 1.0)
    tol = zero_tol_rel * largest
    n_zero = int(np.sum(np.abs(eigs) < tol))
    n_neg = int(np.sum(eigs < -tol))

    # Mass-weighted eigvals are ω² in OpenMM units: kJ/(mol·amu·nm²) = ps^-2
    OMEGA2_TO_S2 = 1e24
    c_cm_ps = 2.99792458e-2
    freqs_cm1 = []
    for lam in eigs:
        if abs(lam) < tol:
            freqs_cm1.append(0.0)
            continue
        omega_ps = np.sqrt(abs(lam))
        freqs_cm1.append(np.sign(lam) * omega_ps / (2 * np.pi * c_cm_ps))

    # Vibrational modes: positive, above zero-tol, excluding the 6 rigid-body
    # zeros and n_torsions projected zeros (gas only)
    n_expected_zeros = n_torsions + (0 if is_bound else 6)
    positive_mask = eigs > tol
    positive_eigs = eigs[positive_mask]
    # Skip n_expected_zeros (already excluded by the threshold, but if some leaked
    # through as tiny positives, drop those too)
    vib_eigs = positive_eigs[max(0, n_expected_zeros - n_zero):]
    omega_rad_s = np.sqrt(vib_eigs * OMEGA2_TO_S2)

    return {
        'eigvals_mw': eigs,
        'frequencies_cm1': freqs_cm1,
        'n_zero': n_zero,
        'n_imaginary': n_neg,
        'n_expected_zeros': n_expected_zeros,
        'vibrational_omega_rad_s': omega_rad_s,
    }


# Helpful constant for callers that want to convert I_r in amu·nm² to amu·Å²
# (Å² == 100 nm² convention is wrong; 1 nm = 10 Å so 1 nm² = 100 Å²; thus
#  I[amu·Å²] = I[amu·nm²] * 100).
AMU_NM2_TO_AMU_ANG2 = 100.0
