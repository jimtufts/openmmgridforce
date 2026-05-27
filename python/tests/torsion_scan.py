"""1D torsion scan + FFT classifier + PG vs DVR entropy router.

Tier 4 of hessian_plan.md. The central decision point of the v2 pipeline:
classify each torsion's potential as cleanly n-fold symmetric (route to
Pitzer-Gwinn closed form) or asymmetric (route to numerical Fourier DVR).

Workflow per rotatable bond:
  1. Rigidly rotate the dependent half about the (j, k) axis to scan φ.
  2. Evaluate V(φ) at n_points evenly spaced angles using whatever forces
     are currently in the OpenMM Context (bonded + grid + NB + GBSA).
  3. FFT-classify: dominant n-fold component / total signal > purity_threshold?
  4. Route: PG if symmetric, DVR otherwise.
"""
import numpy as np
from scipy.interpolate import CubicSpline

try:
    from openmm import unit as omm_unit
    import openmm as _mm
except ImportError:
    omm_unit = None
    _mm = None

from internal_coords import _bfs_side
from entropy_methods import entropy_pitzer_gwinn, KB, HBAR, NA
from dvr_solvers import entropy_dvr

KJMOL_TO_J = 1000.0 / NA


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------
def _rodrigues_rotate(points_nm, axis_unit, angle_rad, pivot_nm):
    """Rotate (M, 3) points about an axis through pivot by angle_rad."""
    p = np.asarray(points_nm) - pivot_nm
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    k = np.asarray(axis_unit)
    return (p * c
            + np.cross(k, p) * s
            + k[None, :] * np.einsum('ij,j->i', p, k)[:, None] * (1 - c)) + pivot_nm


def _iupac_dihedral(p0, p1, p2, p3):
    """IUPAC dihedral angle 0-1-2-3 in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    m = np.cross(b1, b2)
    n = np.cross(b2, b3)
    b2u = b2 / np.linalg.norm(b2)
    sin_phi = float(np.dot(np.cross(m / np.linalg.norm(m), b2u),
                            n / np.linalg.norm(n)))
    cos_phi = float(np.dot(m, n) / (np.linalg.norm(m) * np.linalg.norm(n)))
    return np.arctan2(sin_phi, cos_phi)


def scan_torsion(context, torsion_indices, bond_list, n_points=64):
    """Scan V(φ) along a torsion using rigid rotation of the dependent half.

    Args:
        context:         OpenMM Context whose System has the energy functions
                         you want included (bonded + grid + NB + GBSA, etc.).
        torsion_indices: (i, j, k, l) representative quartet; (j, k) is the
                         rotor axis. l (and downstream atoms via BFS through
                         bond_list excluding j-k) gets rotated.
        bond_list:       iterable of (a, b) bond tuples (1-indexed or 0-indexed,
                         must match the context's atom indexing).
        n_points:        number of scan angles (uniform on [0, 2π)).

    Returns:
        phi_grid (n_points,) in radians
        V_grid   (n_points,) in kJ/mol
    """
    i, j, k, l = torsion_indices
    # BFS the bond graph excluding (j,k) — atoms on k's side rotate
    adj = {}
    for a, b in bond_list:
        if {a, b} == {j, k}:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    side_to_rotate = sorted(_bfs_side(adj, k, blocked=j))

    state = context.getState(getPositions=True)
    pos_orig = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(omm_unit.nanometer))
    N = pos_orig.shape[0]
    pivot = pos_orig[j].copy()
    axis = pos_orig[k] - pos_orig[j]
    axis /= np.linalg.norm(axis)
    phi_current = _iupac_dihedral(pos_orig[i], pos_orig[j], pos_orig[k], pos_orig[l])

    phi_grid = 2 * np.pi * np.arange(n_points) / n_points
    V_grid = np.zeros(n_points)
    for idx_phi, phi_target in enumerate(phi_grid):
        d_phi = phi_target - phi_current
        new_pos = pos_orig.copy()
        rotated = _rodrigues_rotate(
            pos_orig[list(side_to_rotate)], axis, d_phi, pivot)
        for ai, alpha in enumerate(side_to_rotate):
            new_pos[alpha] = rotated[ai]
        context.setPositions(new_pos * omm_unit.nanometer)
        E = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            omm_unit.kilojoule_per_mole)
        V_grid[idx_phi] = E

    # Restore original positions
    context.setPositions(pos_orig * omm_unit.nanometer)
    return phi_grid, V_grid


# ---------------------------------------------------------------------------
# FFT classifier
# ---------------------------------------------------------------------------
def classify_torsion_potential(phi_grid, V_grid_kjmol, purity_threshold=0.9,
                                 candidate_ns=(1, 2, 3, 4, 6)):
    """FFT classify V(φ) on a uniform grid as symmetric n-fold or asymmetric.

    Returns:
        dict with classification, V_b_kjmol (if symmetric), n, phase_rad, purity.
    """
    V = np.asarray(V_grid_kjmol, dtype=float)
    n_points = len(V)
    V_centered = V - V.mean()
    # rfft amplitudes; for a real signal of length N, the k-th amplitude has
    # contribution A_k = (2/N) * |FFT[k]| for 0 < k < N/2
    coefs = np.fft.rfft(V_centered) / (n_points / 2)
    amps = {n: abs(coefs[n]) for n in candidate_ns if n < len(coefs)}
    if not amps:
        return dict(classification='asymmetric', V_b_kjmol=None, n=None,
                    phase_rad=None, purity=0.0)
    dominant_n = max(amps, key=amps.get)
    dominant_amp = amps[dominant_n]
    total_amp = float(sum(abs(c) for c in coefs[1:]))
    purity = (dominant_amp / total_amp) if total_amp > 0 else 0.0
    if purity > purity_threshold:
        return dict(
            classification='symmetric',
            V_b_kjmol=2.0 * dominant_amp,   # FFT amplitude == V_b/2
            n=dominant_n,
            phase_rad=float(-np.angle(coefs[dominant_n])),
            purity=purity,
        )
    return dict(classification='asymmetric', V_b_kjmol=None, n=None,
                phase_rad=None, purity=purity)


# ---------------------------------------------------------------------------
# Entropy router
# ---------------------------------------------------------------------------
def torsion_entropy(phi_grid, V_grid_kjmol, I_r_kg_m2, T_K,
                     purity_threshold=0.9, sigma=1):
    """Route to Pitzer-Gwinn (symmetric n-fold) or Fourier DVR (asymmetric).

    Args:
        phi_grid:        (n_points,) angles in rad on a uniform [0, 2π) grid
        V_grid_kjmol:    (n_points,) potential in kJ/mol
        I_r_kg_m2:       reduced moment (kg·m²)
        T_K:             temperature (K)
        purity_threshold: FFT purity above which to use PG
        sigma:           rotational symmetry number for PG (default 1)

    Returns:
        (S_kB, method, characterization_dict)
    """
    char = classify_torsion_potential(phi_grid, V_grid_kjmol, purity_threshold)
    if char['classification'] == 'symmetric':
        V_b_J = char['V_b_kjmol'] * KJMOL_TO_J
        S = entropy_pitzer_gwinn(V_b_J, char['n'], I_r_kg_m2, T_K, sigma=sigma)
        return S, 'pitzer_gwinn', char
    # Asymmetric: build a periodic spline interpolant for the DVR
    phi_ext = np.concatenate(
        [phi_grid - 2 * np.pi, phi_grid, phi_grid + 2 * np.pi])
    V_ext = np.concatenate([V_grid_kjmol, V_grid_kjmol, V_grid_kjmol])
    spline = CubicSpline(phi_ext, V_ext)

    def V_func(phi):
        return float(spline(phi % (2 * np.pi))) * KJMOL_TO_J  # J

    S = entropy_dvr(V_func, I_r_kg_m2, T_K)
    return S, 'fourier_dvr', char
