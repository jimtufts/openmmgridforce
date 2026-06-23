#!/usr/bin/env python
"""
Desolvation Grid Generator for Grid-Based OBC Solvation.

This module generates grids that store HCT (Hawkins-Cramer-Truhlar) integral
contributions from receptor atoms, along with correction terms that allow
exact computation for any ligand atom radius.

Grid storage per point (2-bin configuration, float32):
- hct_probe: HCT sum using water probe (1.4 Å)
- N_lo, A_lo, B_lo: correction terms for small radii (< 0.12 nm)
- N_hi, A_hi, B_hi: correction terms for large radii (< 0.16 nm)
Total: 7 floats = 28 bytes per grid point

With derivatives (for tricubic/triquintic interpolation):
- hct_derivatives: 8 floats (tricubic) or 27 floats (triquintic) per point
- Correction grids remain values-only (trilinear) due to inherent discontinuities

At runtime, the exact HCT for any ligand radius R_i is:
    HCT(R_i) = HCT_probe + correction(R_i, N, A, B)

where the correction uses the analytical formula:
    ΔI = (1/R_i - 1/R_probe) × [N - 0.25×A×(1/R_i + 1/R_probe)] + B×ln(R_i/R_probe)
"""

import numpy as np
from typing import Tuple, Optional, Literal, Union
from dataclasses import dataclass, field
import time
from scipy.ndimage import gaussian_filter

# Import analytical derivative functions
HAS_DERIVATIVES = False
try:
    # Try relative import first (when used as a package)
    from .hct_analytical_derivatives import (
        compute_hct_derivs_tricubic,
        compute_hct_derivs_triquintic,
        TRICUBIC_NAMES,
        RASPA3_NAMES,
    )
    HAS_DERIVATIVES = True
except ImportError:
    try:
        # Fall back to absolute import (when used standalone)
        from hct_analytical_derivatives import (
            compute_hct_derivs_tricubic,
            compute_hct_derivs_triquintic,
            TRICUBIC_NAMES,
            RASPA3_NAMES,
        )
        HAS_DERIVATIVES = True
    except ImportError:
        pass

# Physical constants
DIELECTRIC_OFFSET = 0.009  # nm (0.09 Å)

# Default configuration
DEFAULT_PROBE_RADIUS = 0.14  # nm (water probe, 1.4 Å)
DEFAULT_R_THRESHOLDS = [0.12, 0.16]  # nm (H-like, heavy atoms)


def scale_derivatives_to_cell_fractional(derivs: np.ndarray, spacing: float) -> np.ndarray:
    """
    Scale derivatives from physical coordinates (nm) to cell-fractional coordinates.

    Triquintic/tricubic Hermite interpolation requires derivatives w.r.t. cell-fractional
    coordinates s ∈ [0,1], not physical coordinates x in nm.

    The relationship is: ∂U/∂s = ∂U/∂x × Δx (where Δx = grid spacing)

    Parameters
    ----------
    derivs : np.ndarray
        Derivatives in physical coordinates, shape (n_derivs,) or (n_derivs, ...)
    spacing : float
        Grid spacing in nm (uniform in all directions)

    Returns
    -------
    np.ndarray
        Derivatives scaled to cell-fractional coordinates
    """
    # For uniform spacing, all directions use the same dx
    dx = dy = dz = spacing

    # Create scaling factors for each RASPA3 derivative index
    # Order: f, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz, dxxy, dxxz, dxyy, dxyz, dyyz, dxzz, dyzz,
    #        dxxyy, dxxzz, dyyzz, dxxyz, dxyyz, dxyzz, dxxyyz, dxxyzz, dxyyzz, dxxyyzz
    scaling = np.array([
        1.0,                    # 0: f (no scaling)
        dx,                     # 1: dx
        dy,                     # 2: dy
        dz,                     # 3: dz
        dx*dx,                  # 4: dxx
        dx*dy,                  # 5: dxy
        dx*dz,                  # 6: dxz
        dy*dy,                  # 7: dyy
        dy*dz,                  # 8: dyz
        dz*dz,                  # 9: dzz
        dx*dx*dy,               # 10: dxxy
        dx*dx*dz,               # 11: dxxz
        dx*dy*dy,               # 12: dxyy
        dx*dy*dz,               # 13: dxyz
        dy*dy*dz,               # 14: dyyz
        dx*dz*dz,               # 15: dxzz
        dy*dz*dz,               # 16: dyzz
        dx*dx*dy*dy,            # 17: dxxyy
        dx*dx*dz*dz,            # 18: dxxzz
        dy*dy*dz*dz,            # 19: dyyzz
        dx*dx*dy*dz,            # 20: dxxyz
        dx*dy*dy*dz,            # 21: dxyyz
        dx*dy*dz*dz,            # 22: dxyzz
        dx*dx*dy*dy*dz,         # 23: dxxyyz
        dx*dx*dy*dz*dz,         # 24: dxxyzz
        dx*dy*dy*dz*dz,         # 25: dxyyzz
        dx*dx*dy*dy*dz*dz,      # 26: dxxyyzz
    ], dtype=np.float64)

    n_derivs = derivs.shape[0]
    scaled = derivs.copy()

    # Apply scaling to each derivative
    for i in range(min(n_derivs, 27)):
        if derivs.ndim == 1:
            scaled[i] *= scaling[i]
        else:
            scaled[i] *= scaling[i]

    return scaled


def compute_hct_integral_vectorized(r: np.ndarray, R_i: float, R_j: np.ndarray,
                                     scale_j: np.ndarray) -> np.ndarray:
    """
    Vectorized HCT integral computation.

    Computes HCT contribution from multiple receptor atoms j to a single point i.

    Parameters
    ----------
    r : np.ndarray
        Distances from point to each receptor atom, shape (n_rec,)
    R_i : float
        Offset radius of the probe/ligand atom
    R_j : np.ndarray
        Offset radii of receptor atoms, shape (n_rec,)
    scale_j : np.ndarray
        OBC scale factors of receptor atoms, shape (n_rec,)

    Returns
    -------
    np.ndarray
        HCT integral contributions, shape (n_rec,)
    """
    # Scaled radius of receptor atoms
    S_j = R_j * scale_j
    r_plus_Sj = r + S_j

    # Initialize output
    hct = np.zeros_like(r)

    # Skip very small distances
    valid = r > 1e-6

    # Check overlap condition
    no_overlap = R_i < r_plus_Sj
    valid = valid & no_overlap

    if not np.any(valid):
        return hct

    # Work only with valid pairs
    r_v = r[valid]
    S_j_v = S_j[valid]
    r_plus_Sj_v = r_plus_Sj[valid]

    r_inv = 1.0 / r_v

    # Compute l_ij (lower integration limit)
    r_minus_Sj = np.abs(r_v - S_j_v)
    use_Ri = R_i > r_minus_Sj
    l_ij = np.where(use_Ri, 1.0 / R_i, 1.0 / r_minus_Sj)

    # Compute u_ij (upper integration limit)
    u_ij = 1.0 / r_plus_Sj_v

    l_ij2 = l_ij * l_ij
    u_ij2 = u_ij * u_ij

    # HCT formula
    ratio = np.log(u_ij / l_ij)
    term = (l_ij - u_ij +
            0.25 * r_v * (u_ij2 - l_ij2) +
            0.5 * r_inv * ratio +
            0.25 * S_j_v * S_j_v * r_inv * (l_ij2 - u_ij2))

    # Tinker correction: atom i completely inside atom j
    inside = R_i < (S_j_v - r_v)
    term = np.where(inside, term + 2.0 * (1.0 / R_i - l_ij), term)

    hct[valid] = term
    return hct


@dataclass
class DesolvationGridData:
    """Container for desolvation grid data."""
    origin: np.ndarray              # Grid origin (x, y, z) in nm
    spacing: float                  # Grid spacing in nm
    counts: Tuple[int, int, int]    # Grid dimensions (nx, ny, nz)
    probe_radius: float             # Probe radius used (nm)
    r_thresholds: Tuple[float, ...] # R thresholds for correction bins

    # Grid data arrays - all float32
    hct_probe: np.ndarray       # Shape (nx, ny, nz)
    correction_N: np.ndarray    # Shape (n_bins, nx, ny, nz)
    correction_A: np.ndarray    # Shape (n_bins, nx, ny, nz)
    correction_B: np.ndarray    # Shape (n_bins, nx, ny, nz)

    # Optional derivative arrays for tricubic/triquintic interpolation
    # Shape (n_derivs, nx, ny, nz) where n_derivs is 8 (tricubic) or 27 (triquintic)
    hct_derivatives: Optional[np.ndarray] = None
    derivative_type: Optional[str] = None  # 'tricubic' or 'triquintic'

    @property
    def n_bins(self) -> int:
        return len(self.r_thresholds)

    @property
    def n_derivatives(self) -> int:
        """Number of derivatives per point (0, 8, or 27)."""
        if self.hct_derivatives is None:
            return 0
        return self.hct_derivatives.shape[0]

    @property
    def has_derivatives(self) -> bool:
        """Whether this grid has stored derivatives."""
        return self.hct_derivatives is not None

    @property
    def memory_bytes(self) -> int:
        """Total memory in bytes."""
        base = (self.hct_probe.nbytes + self.correction_N.nbytes +
                self.correction_A.nbytes + self.correction_B.nbytes)
        if self.hct_derivatives is not None:
            base += self.hct_derivatives.nbytes
        return base

    @property
    def floats_per_point(self) -> int:
        """Number of floats stored per grid point."""
        base = 1 + 3 * self.n_bins
        if self.hct_derivatives is not None:
            # Derivatives replace the single hct_probe value
            base = self.n_derivatives + 3 * self.n_bins
        return base

    def save(self, filepath: str):
        """Save grid to numpy npz file."""
        save_dict = dict(
            origin=self.origin,
            spacing=self.spacing,
            counts=np.array(self.counts),
            probe_radius=self.probe_radius,
            r_thresholds=np.array(self.r_thresholds),
            hct_probe=self.hct_probe,
            correction_N=self.correction_N,
            correction_A=self.correction_A,
            correction_B=self.correction_B,
        )
        if self.hct_derivatives is not None:
            save_dict['hct_derivatives'] = self.hct_derivatives
            save_dict['derivative_type'] = self.derivative_type
        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load(cls, filepath: str) -> 'DesolvationGridData':
        """Load grid from numpy npz file."""
        data = np.load(filepath)
        kwargs = dict(
            origin=data['origin'],
            spacing=float(data['spacing']),
            counts=tuple(data['counts']),
            probe_radius=float(data['probe_radius']),
            r_thresholds=tuple(data['r_thresholds']),
            hct_probe=data['hct_probe'],
            correction_N=data['correction_N'],
            correction_A=data['correction_A'],
            correction_B=data['correction_B'],
        )
        if 'hct_derivatives' in data:
            kwargs['hct_derivatives'] = data['hct_derivatives']
            kwargs['derivative_type'] = str(data['derivative_type'])
        return cls(**kwargs)


def generate_desolvation_grid(
    rec_positions: np.ndarray,
    rec_radii: np.ndarray,
    rec_scales: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    counts: Tuple[int, int, int],
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    r_thresholds: Tuple[float, ...] = None,
    dtype: np.dtype = np.float32,
    verbose: bool = True,
    compute_derivatives: Optional[Literal['tricubic', 'triquintic']] = None,
    smooth_corrections: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> DesolvationGridData:
    """
    Generate a desolvation grid for the receptor.

    Parameters
    ----------
    rec_positions : np.ndarray
        Receptor atom positions, shape (n_atoms, 3), in nm
    rec_radii : np.ndarray
        Receptor intrinsic radii, shape (n_atoms,), in nm
    rec_scales : np.ndarray
        Receptor OBC scale factors, shape (n_atoms,)
    origin : np.ndarray
        Grid origin (x, y, z) in nm
    spacing : float
        Grid spacing in nm
    counts : tuple
        Grid dimensions (nx, ny, nz)
    probe_radius : float
        Probe radius for HCT calculation (nm), default 0.14 (water)
    r_thresholds : tuple
        R thresholds for correction bins (offset radii in nm)
        Default: (0.12, 0.16) for H-like and heavy atoms
    dtype : np.dtype
        Data type for grid arrays, default float32
    verbose : bool
        Print progress information
    compute_derivatives : str or None
        If 'tricubic', compute 8 derivatives per point for tricubic interpolation.
        If 'triquintic', compute 27 derivatives per point for triquintic interpolation.
        If None (default), compute only values (for trilinear/bspline interpolation).
        Note: Correction grids always use values only (trilinear) due to inherent
        discontinuities at regime boundaries.
    smooth_corrections : float or tuple of floats, optional
        If provided, apply Gaussian smoothing to correction grids (N, A, B).
        If a single float, use as sigma in grid units (e.g., 1.0 = 1 grid spacing).
        If a tuple of 3 floats, use as (sigma_x, sigma_y, sigma_z).
        This can reduce force discontinuities at cell boundaries at the cost of
        some accuracy in the correction formula.

    Returns
    -------
    DesolvationGridData
        Grid object containing HCT and correction data
    """
    if r_thresholds is None:
        r_thresholds = tuple(DEFAULT_R_THRESHOLDS)

    # Validate derivative computation option
    if compute_derivatives is not None:
        if not HAS_DERIVATIVES:
            raise ImportError(
                "HCT analytical derivatives module not available. "
                "Run generate_hct_derivatives.py first."
            )
        if compute_derivatives not in ('tricubic', 'triquintic'):
            raise ValueError(
                f"compute_derivatives must be 'tricubic', 'triquintic', or None, "
                f"got {compute_derivatives!r}"
            )

    nx, ny, nz = counts
    n_bins = len(r_thresholds)
    n_rec = len(rec_positions)
    n_points = nx * ny * nz

    # Number of derivatives per point
    if compute_derivatives == 'tricubic':
        n_derivs = 8
    elif compute_derivatives == 'triquintic':
        n_derivs = 27
    else:
        n_derivs = 0

    # Precompute receptor parameters
    rec_off = np.maximum(rec_radii - DIELECTRIC_OFFSET, 1e-6).astype(np.float64)
    rec_S = rec_off * rec_scales
    R_probe_off = probe_radius - DIELECTRIC_OFFSET

    # Allocate arrays
    hct_probe = np.zeros((nx, ny, nz), dtype=dtype)
    correction_N = np.zeros((n_bins, nx, ny, nz), dtype=dtype)
    correction_A = np.zeros((n_bins, nx, ny, nz), dtype=dtype)
    correction_B = np.zeros((n_bins, nx, ny, nz), dtype=dtype)

    # Allocate derivative array if needed
    if n_derivs > 0:
        hct_derivatives = np.zeros((n_derivs, nx, ny, nz), dtype=dtype)
    else:
        hct_derivatives = None

    if verbose:
        print(f"Generating desolvation grid:")
        print(f"  Grid size: {nx} x {ny} x {nz} = {n_points:,} points")
        print(f"  Receptor atoms: {n_rec:,}")
        print(f"  Probe radius: {probe_radius} nm")
        print(f"  R thresholds: {r_thresholds}")
        print(f"  Data type: {dtype}")
        if compute_derivatives:
            print(f"  Derivatives: {compute_derivatives} ({n_derivs} per point)")
        else:
            print(f"  Derivatives: none (values only)")
        floats_per_pt = (n_derivs if n_derivs > 0 else 1) + 3 * n_bins
        mem_mb = n_points * floats_per_pt * np.dtype(dtype).itemsize / 1e6
        print(f"  Estimated memory: {mem_mb:.1f} MB ({floats_per_pt} floats/point)")

    start_time = time.time()

    # Generate grid points
    x_coords = origin[0] + np.arange(nx) * spacing
    y_coords = origin[1] + np.arange(ny) * spacing
    z_coords = origin[2] + np.arange(nz) * spacing

    # Select derivative function
    if compute_derivatives == 'tricubic':
        compute_derivs_func = compute_hct_derivs_tricubic
    elif compute_derivatives == 'triquintic':
        compute_derivs_func = compute_hct_derivs_triquintic
    else:
        compute_derivs_func = None

    # Process in slices for memory efficiency
    for ix in range(nx):
        if verbose and (ix % max(1, nx // 10) == 0 or ix == nx - 1):
            elapsed = time.time() - start_time
            if ix > 0:
                eta = elapsed / ix * (nx - ix)
                print(f"  Progress: {ix+1}/{nx} ({100*(ix+1)/nx:.0f}%) - ETA: {eta:.0f}s")
            else:
                print(f"  Progress: {ix+1}/{nx} ({100*(ix+1)/nx:.0f}%)")

        x = x_coords[ix]

        for iy in range(ny):
            y = y_coords[iy]

            for iz in range(nz):
                z = z_coords[iz]
                point = np.array([x, y, z])

                # Compute displacement from receptor atoms to grid point
                # Note: dx = point - rec_pos (grid point minus receptor atom)
                diff = point - rec_positions  # Shape (n_rec, 3)
                r = np.sqrt(np.sum(diff**2, axis=1))

                # Compute HCT with probe radius (vectorized)
                # Note: compute_hct_integral_vectorized expects r as distances
                hct_vals = compute_hct_integral_vectorized(r, R_probe_off, rec_off, rec_scales)
                hct_probe[ix, iy, iz] = np.sum(hct_vals)

                # Compute derivatives if requested
                if compute_derivs_func is not None:
                    # Sum derivatives from all receptor atoms
                    deriv_sum = np.zeros(n_derivs, dtype=np.float64)

                    for j in range(n_rec):
                        if r[j] < 1e-6:
                            continue  # Skip if grid point is at receptor atom

                        # Check if contribution is non-zero (same conditions as HCT)
                        r_plus_Sj = r[j] + rec_S[j]
                        if R_probe_off >= r_plus_Sj:
                            continue  # Overlap condition

                        # Compute derivatives using chain rule approach
                        # dx, dy, dz are displacements from receptor atom to grid point
                        dx, dy, dz = diff[j]
                        derivs = compute_derivs_func(dx, dy, dz, rec_S[j], R_probe_off)
                        deriv_sum += derivs

                    # Scale from physical coordinates (nm) to cell-fractional coordinates
                    # Required for Lekien-Marsden tricubic/triquintic interpolation
                    deriv_sum = scale_derivatives_to_cell_fractional(deriv_sum, spacing)
                    hct_derivatives[:, ix, iy, iz] = deriv_sum.astype(dtype)

                # Compute correction terms for each bin
                crossover = np.abs(r - rec_S)

                for b, thresh in enumerate(r_thresholds):
                    # Atoms in R-dependent regime for both probe and this threshold
                    mask = (crossover < thresh) & (crossover < R_probe_off) & (r > 1e-6)

                    if np.any(mask):
                        r_m = r[mask]
                        S_m = rec_S[mask]

                        correction_N[b, ix, iy, iz] = np.sum(mask)
                        correction_A[b, ix, iy, iz] = np.sum(r_m - S_m**2 / r_m)
                        correction_B[b, ix, iy, iz] = np.sum(0.5 / r_m)

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Completed in {elapsed:.1f}s")

    # Apply Gaussian smoothing to correction grids if requested
    if smooth_corrections is not None:
        if verbose:
            print(f"  Applying Gaussian smoothing to correction grids (sigma={smooth_corrections})...")

        # Convert sigma to tuple if scalar
        if isinstance(smooth_corrections, (int, float)):
            sigma = (float(smooth_corrections),) * 3
        else:
            sigma = tuple(smooth_corrections)

        # Smooth each bin's correction grids
        for b in range(n_bins):
            correction_N[b] = gaussian_filter(correction_N[b].astype(np.float64), sigma=sigma).astype(dtype)
            correction_A[b] = gaussian_filter(correction_A[b].astype(np.float64), sigma=sigma).astype(dtype)
            correction_B[b] = gaussian_filter(correction_B[b].astype(np.float64), sigma=sigma).astype(dtype)

        if verbose:
            print(f"  Smoothing complete")

    return DesolvationGridData(
        origin=origin.astype(np.float64),
        spacing=spacing,
        counts=counts,
        probe_radius=probe_radius,
        r_thresholds=r_thresholds,
        hct_probe=hct_probe,
        correction_N=correction_N,
        correction_A=correction_A,
        correction_B=correction_B,
        hct_derivatives=hct_derivatives,
        derivative_type=compute_derivatives,
    )


def interpolate_grid(grid: DesolvationGridData, position: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trilinear interpolation of grid values at a position.

    Returns
    -------
    tuple
        (hct_probe, N_bins, A_bins, B_bins) interpolated values
    """
    # Convert position to grid coordinates
    grid_pos = (position - grid.origin) / grid.spacing

    # Get integer indices
    i0 = int(np.floor(grid_pos[0]))
    j0 = int(np.floor(grid_pos[1]))
    k0 = int(np.floor(grid_pos[2]))

    # Clamp to grid bounds
    i0 = max(0, min(i0, grid.counts[0] - 2))
    j0 = max(0, min(j0, grid.counts[1] - 2))
    k0 = max(0, min(k0, grid.counts[2] - 2))

    # Fractional parts
    fx = grid_pos[0] - i0
    fy = grid_pos[1] - j0
    fz = grid_pos[2] - k0

    # Clamp fractions
    fx = max(0, min(1, fx))
    fy = max(0, min(1, fy))
    fz = max(0, min(1, fz))

    # Trilinear weights
    w = np.array([
        (1-fx) * (1-fy) * (1-fz),
        (1-fx) * (1-fy) * fz,
        (1-fx) * fy * (1-fz),
        (1-fx) * fy * fz,
        fx * (1-fy) * (1-fz),
        fx * (1-fy) * fz,
        fx * fy * (1-fz),
        fx * fy * fz,
    ])

    # Indices for 8 corners
    corners = [
        (i0, j0, k0), (i0, j0, k0+1), (i0, j0+1, k0), (i0, j0+1, k0+1),
        (i0+1, j0, k0), (i0+1, j0, k0+1), (i0+1, j0+1, k0), (i0+1, j0+1, k0+1),
    ]

    # Interpolate HCT
    hct = sum(w[c] * grid.hct_probe[corners[c]] for c in range(8))

    # Interpolate correction terms for each bin
    N_bins = np.zeros(grid.n_bins)
    A_bins = np.zeros(grid.n_bins)
    B_bins = np.zeros(grid.n_bins)

    for b in range(grid.n_bins):
        N_bins[b] = sum(w[c] * grid.correction_N[b][corners[c]] for c in range(8))
        A_bins[b] = sum(w[c] * grid.correction_A[b][corners[c]] for c in range(8))
        B_bins[b] = sum(w[c] * grid.correction_B[b][corners[c]] for c in range(8))

    return hct, N_bins, A_bins, B_bins


def get_hct_from_grid(grid: DesolvationGridData, position: np.ndarray, R_i: float) -> float:
    """
    Get corrected HCT value for a ligand atom.

    Parameters
    ----------
    grid : DesolvationGridData
        The desolvation grid
    position : np.ndarray
        Atom position in nm
    R_i : float
        Intrinsic radius of the ligand atom in nm

    Returns
    -------
    float
        Corrected HCT value
    """
    hct_probe, N_bins, A_bins, B_bins = interpolate_grid(grid, position)

    R_i_off = R_i - DIELECTRIC_OFFSET
    R_probe_off = grid.probe_radius - DIELECTRIC_OFFSET

    # Find appropriate bin (first threshold >= R_i_off)
    bin_idx = 0
    for idx, thresh in enumerate(grid.r_thresholds):
        if thresh >= R_i_off:
            bin_idx = idx
            break
    else:
        bin_idx = grid.n_bins - 1

    N = N_bins[bin_idx]
    A = A_bins[bin_idx]
    B = B_bins[bin_idx]

    # Always apply correction - N, A, B all go to zero together
    # in regions far from receptor, so correction naturally vanishes
    delta = 1.0/R_i_off - 1.0/R_probe_off
    sigma = 1.0/R_i_off + 1.0/R_probe_off
    correction = delta * (N - 0.25 * A * sigma) + B * np.log(R_i_off/R_probe_off)
    return hct_probe + correction


if __name__ == '__main__':
    # Quick test
    print("Desolvation Grid Generator")
    print("=" * 60)

    # Create a simple test case
    rec_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, 0.3, 0.0],
    ])
    rec_radii = np.array([0.17, 0.17, 0.15])
    rec_scales = np.array([0.72, 0.72, 0.79])

    origin = np.array([-0.5, -0.5, -0.5])
    spacing = 0.1
    counts = (11, 11, 11)

    # Test without derivatives
    print("\n--- Test 1: Values only ---")
    grid = generate_desolvation_grid(
        rec_positions, rec_radii, rec_scales,
        origin, spacing, counts,
        verbose=True
    )

    print(f"\nGrid created:")
    print(f"  Memory: {grid.memory_bytes / 1e3:.1f} KB")
    print(f"  Floats per point: {grid.floats_per_point}")
    print(f"  Has derivatives: {grid.has_derivatives}")

    # Test interpolation
    test_pos = np.array([0.15, 0.15, 0.0])
    hct = get_hct_from_grid(grid, test_pos, 0.17)
    print(f"\nHCT at {test_pos} for R=0.17: {hct:.4f}")

    # Test with tricubic derivatives
    if HAS_DERIVATIVES:
        print("\n" + "=" * 60)
        print("--- Test 2: Tricubic derivatives ---")
        grid_tri = generate_desolvation_grid(
            rec_positions, rec_radii, rec_scales,
            origin, spacing, counts,
            verbose=True,
            compute_derivatives='tricubic'
        )

        print(f"\nGrid created:")
        print(f"  Memory: {grid_tri.memory_bytes / 1e3:.1f} KB")
        print(f"  Floats per point: {grid_tri.floats_per_point}")
        print(f"  Has derivatives: {grid_tri.has_derivatives}")
        print(f"  Derivative type: {grid_tri.derivative_type}")
        print(f"  N derivatives: {grid_tri.n_derivatives}")

        # Show derivatives at a sample point
        ix, iy, iz = 5, 5, 5
        print(f"\nDerivatives at grid point ({ix},{iy},{iz}):")
        for i, name in enumerate(TRICUBIC_NAMES):
            print(f"  {name}: {grid_tri.hct_derivatives[i, ix, iy, iz]:.6e}")

        # Validate with finite differences
        print("\nFinite difference validation of derivatives:")
        h = 1e-6
        x = origin[0] + ix * spacing
        y = origin[1] + iy * spacing
        z = origin[2] + iz * spacing

        # Get analytical derivative from grid
        f_val = grid_tri.hct_derivatives[0, ix, iy, iz]
        fx_val = grid_tri.hct_derivatives[1, ix, iy, iz]

        # Compute numerical derivative using pairwise HCT
        rec_off = np.maximum(rec_radii - DIELECTRIC_OFFSET, 1e-6)
        R_probe_off = grid.probe_radius - DIELECTRIC_OFFSET

        def compute_hct_at(pos):
            diff = pos - rec_positions
            r = np.sqrt(np.sum(diff**2, axis=1))
            hct_vals = compute_hct_integral_vectorized(r, R_probe_off, rec_off, rec_scales)
            return np.sum(hct_vals)

        pos = np.array([x, y, z])
        hct_px = compute_hct_at(pos + np.array([h, 0, 0]))
        hct_mx = compute_hct_at(pos - np.array([h, 0, 0]))
        fx_fd = (hct_px - hct_mx) / (2 * h)

        print(f"  fx: analytical={fx_val:.6e}, FD={fx_fd:.6e}, diff={abs(fx_val - fx_fd):.2e}")

        # Test triquintic
        print("\n" + "=" * 60)
        print("--- Test 3: Triquintic derivatives ---")
        grid_quint = generate_desolvation_grid(
            rec_positions, rec_radii, rec_scales,
            origin, spacing, counts,
            verbose=True,
            compute_derivatives='triquintic'
        )

        print(f"\nGrid created:")
        print(f"  Memory: {grid_quint.memory_bytes / 1e3:.1f} KB")
        print(f"  Floats per point: {grid_quint.floats_per_point}")
        print(f"  N derivatives: {grid_quint.n_derivatives}")
    else:
        print("\n[Derivative computation not available - run generate_hct_derivatives.py first]")
