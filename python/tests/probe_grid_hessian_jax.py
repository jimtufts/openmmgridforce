"""JAX autodiff reference for pure-GRID IsolatedGBSAForce.

Replaces the receptor HCT pairwise loop with a cubic B-spline
interpolation of the plugin's DesolvationGrid, matching the CUDA
tricubic_bspline path exactly. Then autodiffs to get true reference
forces (jax.grad) and Hessian (jax.hessian) with no FD step-size.

Verification protocol:
  1. Compare JAX pure-GRID energy to plugin pure-GRID energy.
     If they match to ~1e-4, the JAX energy function faithfully
     reproduces the plugin's GRID energy expression.
  2. Compare JAX force (jax.grad) to plugin force.
     If they match, the plugin's grid-mode forces are the analytical
     gradient of the JAX energy (already validated but worth checking).
  3. Compare JAX Hessian (jax.hessian) to plugin analytical Hessian.
     Non-trivial delta here is the real analytical assembly bug.
"""
import numpy as np
import openmm as mm
from openmm import unit
import gridforceplugin as gfp

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# ---- OBC / GB constants (must match plugin) ------------------------------
DIELECTRIC_OFFSET = 0.009
OBC_ALPHA, OBC_BETA, OBC_GAMMA = 1.0, 0.8, 4.85
SOLUTE_DIEL, SOLVENT_DIEL = 1.0, 78.5
PREFACTOR = -138.935456 * (1.0/SOLUTE_DIEL - 1.0/SOLVENT_DIEL)
SURFACE_TENSION = 2.25936  # DEFAULT_SA_SURFACE_TENSION from IsolatedGBSAForce.h
BORN_MAX = 50.0


# ---- Test system (same as gbsa_grid_section) -----------------------------
np.random.seed(42)
n_rec, n_lig = 40, 12
rec_pos = np.random.randn(n_rec, 3) * 0.4
lig_pos = np.random.randn(n_lig, 3) * 0.15 + np.array([0.9, 0.0, 0.0])
rec_q = np.random.randn(n_rec) * 0.3
lig_q = np.random.randn(n_lig) * 0.3
rec_r = 0.12 + np.random.rand(n_rec) * 0.06
lig_r = 0.12 + np.random.rand(n_lig) * 0.06
rec_s = 0.7 + np.random.rand(n_rec) * 0.2
lig_s = 0.7 + np.random.rand(n_lig) * 0.2
PROBE_RADIUS = 0.14
R_THRESHOLDS = [0.12, 0.16]

all_pos = np.vstack([rec_pos, lig_pos])
lo = all_pos.min(axis=0) - 0.6
hi = all_pos.max(axis=0) + 0.6
SP = 0.15
counts = tuple(int(np.ceil((hi[d] - lo[d]) / SP)) + 1 for d in range(3))


def compute_receptor_baseline():
    R = rec_r.astype(np.float64); S = rec_s.astype(np.float64)
    p = rec_pos.astype(np.float64)
    R_off = R - DIELECTRIC_OFFSET
    hct = np.zeros(n_rec)
    for i in range(n_rec):
        for j in range(n_rec):
            if i == j: continue
            r = float(np.linalg.norm(p[i] - p[j]))
            Sj = (R[j] - DIELECTRIC_OFFSET) * S[j]
            r_plus_Sj = r + Sj
            if R_off[i] >= r_plus_Sj: continue
            r_minus_Sj = abs(r - Sj)
            l = 1.0/R_off[i] if R_off[i] > r_minus_Sj else 1.0/r_minus_Sj
            u = 1.0 / r_plus_Sj
            l2, u2 = l*l, u*u
            term = (l - u + 0.25*r*(u2 - l2) + 0.5*(1.0/r)*np.log(u/l)
                    + 0.25*Sj*Sj*(1.0/r)*(l2 - u2))
            if R_off[i] < (Sj - r):
                term += 2.0 * (1.0/R_off[i] - l)
            hct[i] += term
    psi = 0.5 * R_off * hct
    tanh_val = np.tanh(OBC_ALPHA*psi - OBC_BETA*psi**2 + OBC_GAMMA*psi**3)
    denom = 1.0/R_off - tanh_val / R
    return np.minimum(np.where(denom > 0, 1.0/denom, R), BORN_MAX)


def autogen_grid(with_derivs=False):
    f_gen = gfp.GBSAGridForce()
    f_gen.setNumAtoms(1)
    f_gen.setAtomParameters(0, 0.0, float(lig_r[0]), float(lig_s[0]))
    f_gen.setInterpolationMethod(0)
    f_gen.setAutoGenerateGrid(True); f_gen.setUseKDEGeneration(True)
    if with_derivs:
        f_gen.setComputeGridDerivatives(True)
    f_gen.setReceptorPositions(rec_pos.flatten().tolist())
    f_gen.setReceptorRadii(rec_r.tolist())
    f_gen.setReceptorScaleFactors(rec_s.tolist())
    f_gen.setGridOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
    f_gen.setGridCounts(int(counts[0]), int(counts[1]), int(counts[2]))
    f_gen.setGridSpacing(float(SP))
    f_gen.setProbeRadius(PROBE_RADIUS); f_gen.setRThresholds(R_THRESHOLDS)
    f_gen.setParticles([0]); f_gen.addParticleGroup('bootstrap', [0])
    s = mm.System(); s.addParticle(12.0); s.addForce(f_gen)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
    ctx.getState(getEnergy=True)
    cg = f_gen.getDesolvationGrid()
    del ctx
    return cg


R_rec_baseline = compute_receptor_baseline()
grid = autogen_grid(with_derivs=False)
grid_wd = autogen_grid(with_derivs=True)
nx, ny, nz = counts
origin = np.array([lo[0], lo[1], lo[2]], dtype=np.float64)

# --- Triquintic hermite reference (JAX) -----------------------------------
# Matrix parsed once from TriquinticCoefficients.cuh via
# _extract_triquintic_matrix.py.
TQM = np.load(
    "/home/jtufts/src/p312/openmmgridforce/python/tests/triquintic_matrix.npy"
).astype(np.float64)  # (216, 216)

# Extract with-derivs grid buffers, reshape to (nDerivs, nx, ny, nz).
def _reshape_grid(buf_flat, n_derivs, nx, ny, nz):
    a = np.asarray(buf_flat, dtype=np.float64)
    assert a.size == n_derivs * nx * ny * nz, (
        f"grid buffer size {a.size} != {n_derivs}*{nx}*{ny}*{nz}")
    return a.reshape(n_derivs, nx, ny, nz)

hct_derivs_np = _reshape_grid(grid_wd.getHctProbe(), 27, nx, ny, nz)
_corrN = np.asarray(grid_wd.getCorrectionN(), dtype=np.float64)
_corrA = np.asarray(grid_wd.getCorrectionA(), dtype=np.float64)
_corrB = np.asarray(grid_wd.getCorrectionB(), dtype=np.float64)
_numPoints = nx * ny * nz
_numBins = len(R_THRESHOLDS)
if _corrN.size == _numBins * 27 * _numPoints:
    corr_mode = "binned_kde"
    shape5 = (_numBins, 27, nx, ny, nz)
    corrN_bkde = _corrN.reshape(shape5)
    corrA_bkde = _corrA.reshape(shape5)
    corrB_bkde = _corrB.reshape(shape5)
else:
    corr_mode = "trilinear"
    numBins_g = _corrN.size // _numPoints
    corrN_tl = _corrN.reshape(numBins_g, nx, ny, nz)
    corrA_tl = _corrA.reshape(numBins_g, nx, ny, nz)
    corrB_tl = _corrB.reshape(numBins_g, nx, ny, nz)
print(f"triquintic ref: corr_mode={corr_mode}  corrN.size={_corrN.size}")

TQM_j = jnp.array(TQM)
hct_derivs_j = jnp.array(hct_derivs_np)
if corr_mode == "binned_kde":
    corrN_bkde_j = jnp.array(corrN_bkde)
    corrA_bkde_j = jnp.array(corrA_bkde)
    corrB_bkde_j = jnp.array(corrB_bkde)
else:
    corrN_tl_j = jnp.array(corrN_tl)
    corrA_tl_j = jnp.array(corrA_tl)
    corrB_tl_j = jnp.array(corrB_tl)


def _corner_stencil(grid_derivs_4d, ix, iy, iz):
    """Return X[d,c] where c enumerates the 8 corners in CUDA order:
    c=0..7 → (cx,cy,cz) = (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),
                          (1,0,1),(0,1,1),(1,1,1).
    Then flatten to X[d*8 + c] to match TRIQUINTIC_COEFFICIENTS layout.
    """
    cx = jnp.array([0, 1, 0, 1, 0, 1, 0, 1])
    cy = jnp.array([0, 0, 1, 1, 0, 0, 1, 1])
    cz = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    gxi = jnp.clip(ix + cx, 0, nx - 1)
    gyi = jnp.clip(iy + cy, 0, ny - 1)
    gzi = jnp.clip(iz + cz, 0, nz - 1)
    # gather grid_derivs[:, gxi, gyi, gzi]  → shape (27, 8)
    Xdc = grid_derivs_4d[:, gxi, gyi, gzi]
    return Xdc.reshape(-1)  # (216,)


def _poly_value(a216, ffx, ffy, ffz):
    """Σ_ijk a[i + 6j + 36k] * ffx^i * ffy^j * ffz^k, i,j,k ∈ [0..5]."""
    px = jnp.array([ffx**p for p in range(6)])
    py = jnp.array([ffy**p for p in range(6)])
    pz = jnp.array([ffz**p for p in range(6)])
    # outer product of powers, contract with a reshaped to (k,j,i) axes
    a_kji = a216.reshape(6, 6, 6)  # k in axis 0, j in axis 1, i in axis 2
    return jnp.einsum('kji,i,j,k->', a_kji, px, py, pz)


def _trilinear_value(grid_bin_4d, bin_idx, ix, iy, iz, ffx, ffy, ffz):
    """Trilinear interpolation on a single-bin selected 3D grid slice."""
    g = grid_bin_4d[bin_idx]  # (nx, ny, nz)
    ix1 = jnp.clip(ix + 1, 0, nx - 1)
    iy1 = jnp.clip(iy + 1, 0, ny - 1)
    iz1 = jnp.clip(iz + 1, 0, nz - 1)
    c000 = g[ix, iy, iz]
    c100 = g[ix1, iy, iz]
    c010 = g[ix, iy1, iz]
    c110 = g[ix1, iy1, iz]
    c001 = g[ix, iy, iz1]
    c101 = g[ix1, iy, iz1]
    c011 = g[ix, iy1, iz1]
    c111 = g[ix1, iy1, iz1]
    return ((1-ffx)*(1-ffy)*(1-ffz)*c000 + ffx*(1-ffy)*(1-ffz)*c100 +
            (1-ffx)*ffy*(1-ffz)*c010 + ffx*ffy*(1-ffz)*c110 +
            (1-ffx)*(1-ffy)*ffz*c001 + ffx*(1-ffy)*ffz*c101 +
            (1-ffx)*ffy*ffz*c011 + ffx*ffy*ffz*c111)


def grid_hct_triquintic(pos, R_i_off):
    """JAX-differentiable triquintic hermite eval of the HCT probe grid +
    correction grids at pos, matching gbsaGridForce.cu:681-953 for
    method=3 (triquintic_hermite)."""
    R_probe_off = PROBE_RADIUS - DIELECTRIC_OFFSET
    inv_Ri = 1.0 / R_i_off
    inv_Rp = 1.0 / R_probe_off
    delta = inv_Ri - inv_Rp
    sigma = inv_Ri + inv_Rp
    logTerm = jnp.log(R_i_off / R_probe_off)
    dCorr_dN = delta
    dCorr_dA = -0.25 * delta * sigma
    dCorr_dB = logTerm

    invSp = 1.0 / SP
    fx_r = (pos[0] - origin[0]) * invSp
    fy_r = (pos[1] - origin[1]) * invSp
    fz_r = (pos[2] - origin[2]) * invSp
    ix = jnp.floor(fx_r).astype(jnp.int32)
    iy = jnp.floor(fy_r).astype(jnp.int32)
    iz = jnp.floor(fz_r).astype(jnp.int32)
    ffx = fx_r - ix
    ffy = fy_r - iy
    ffz = fz_r - iz

    # HCT probe via triquintic hermite
    X_hct = _corner_stencil(hct_derivs_j, ix, iy, iz)
    a_hct = 0.125 * (TQM_j @ X_hct)
    hct_val = _poly_value(a_hct, ffx, ffy, ffz)

    if corr_mode == "binned_kde":
        # Bin selection: first b with rThresholds[b] >= R_i_off; else last.
        R_thr = jnp.array(R_THRESHOLDS)
        cond = (R_thr >= R_i_off).astype(jnp.int32)
        # index of first True; if none, use last (numBins-1)
        any_hit = jnp.sum(cond) > 0
        first_true = jnp.argmax(cond)  # 0 if none, else first True
        binIdx = jnp.where(any_hit, first_true, len(R_THRESHOLDS) - 1)
        X_N = _corner_stencil(corrN_bkde_j[binIdx], ix, iy, iz)
        X_A = _corner_stencil(corrA_bkde_j[binIdx], ix, iy, iz)
        X_B = _corner_stencil(corrB_bkde_j[binIdx], ix, iy, iz)
        a_N = 0.125 * (TQM_j @ X_N)
        a_A = 0.125 * (TQM_j @ X_A)
        a_B = 0.125 * (TQM_j @ X_B)
        corr_val = (dCorr_dN * _poly_value(a_N, ffx, ffy, ffz) +
                    dCorr_dA * _poly_value(a_A, ffx, ffy, ffz) +
                    dCorr_dB * _poly_value(a_B, ffx, ffy, ffz))
    else:
        R_thr = jnp.array(R_THRESHOLDS)
        cond = (R_thr >= R_i_off).astype(jnp.int32)
        any_hit = jnp.sum(cond) > 0
        first_true = jnp.argmax(cond)
        binIdx = jnp.where(any_hit, first_true, corrN_tl_j.shape[0] - 1)
        n = _trilinear_value(corrN_tl_j, binIdx, ix, iy, iz, ffx, ffy, ffz)
        a = _trilinear_value(corrA_tl_j, binIdx, ix, iy, iz, ffx, ffy, ffz)
        b = _trilinear_value(corrB_tl_j, binIdx, ix, iy, iz, ffx, ffy, ffz)
        corr_val = dCorr_dN * n + dCorr_dA * a + dCorr_dB * b

    return hct_val + corr_val

hctProbe_np    = np.array(grid.getHctProbe(),    dtype=np.float64)
correctionN_np = np.array(grid.getCorrectionN(), dtype=np.float64)
correctionA_np = np.array(grid.getCorrectionA(), dtype=np.float64)
correctionB_np = np.array(grid.getCorrectionB(), dtype=np.float64)

# With useKDEGeneration=True and no ComputeGridDerivatives call, plugin sets
# useKDECorrections=True and hasBinnedKDEDerivatives=False. In that case
# corrOffset=0 regardless of atom radius (see gbsaGridForce.cu:131). All
# atoms share a single correction grid of size nx*ny*nz.
assert hctProbe_np.size == nx * ny * nz, (
    f"hctProbe size {hctProbe_np.size} != {nx*ny*nz}")

hctProbe = jnp.array(hctProbe_np).reshape(nx, ny, nz)
corrN    = jnp.array(correctionN_np[:nx*ny*nz]).reshape(nx, ny, nz)
corrA    = jnp.array(correctionA_np[:nx*ny*nz]).reshape(nx, ny, nz)
corrB    = jnp.array(correctionB_np[:nx*ny*nz]).reshape(nx, ny, nz)


# ---- Cubic B-spline basis functions (match CUDA bspline_basis*) ---------
def bspline_basis(f):
    """Cubic B3 basis at fractional coord f in [0,1). Returns 4-vector.
    Matches bspline_basis0..3 in gbsaGridForce.cu.
    """
    om = 1.0 - f
    b0 = (om**3) / 6.0
    b1 = (4.0 - 6.0*f*f + 3.0*(f**3)) / 6.0
    b2 = (4.0 - 6.0*om*om + 3.0*(om**3)) / 6.0
    b3 = (f**3) / 6.0
    return jnp.stack([b0, b1, b2, b3])


def grid_hct_bspline(pos, R_i_off):
    """Cubic B-spline interpolation of gridHctProbe + correction*coefs at pos.
    Reproduces CUDA method=1 path (tricubic_bspline) in the KDE-single-bin
    case (useKDECorrections=True, hasBinnedKDEDerivatives=False, corrOffset=0).
    """
    R_probe_off = PROBE_RADIUS - DIELECTRIC_OFFSET
    inv_Ri = 1.0 / R_i_off
    inv_Rp = 1.0 / R_probe_off
    delta = inv_Ri - inv_Rp
    sigma = inv_Ri + inv_Rp
    logTerm = jnp.log(R_i_off / R_probe_off)
    dCorr_dN = delta
    dCorr_dA = -0.25 * delta * sigma
    dCorr_dB = logTerm

    invSp = 1.0 / SP
    fx = (pos[0] - origin[0]) * invSp
    fy = (pos[1] - origin[1]) * invSp
    fz = (pos[2] - origin[2]) * invSp
    ix = jnp.floor(fx).astype(jnp.int32)
    iy = jnp.floor(fy).astype(jnp.int32)
    iz = jnp.floor(fz).astype(jnp.int32)
    ffx = fx - ix
    ffy = fy - iy
    ffz = fz - iz

    bx = bspline_basis(ffx)
    by = bspline_basis(ffy)
    bz = bspline_basis(ffz)

    # 4x4x4 stencil gather with clip-to-boundary
    # gxi = clip(ix-1+i, 0, nx-1), etc.
    ii = jnp.arange(4)
    gxi = jnp.clip(ix - 1 + ii, 0, nx - 1)
    gyi = jnp.clip(iy - 1 + ii, 0, ny - 1)
    gzi = jnp.clip(iz - 1 + ii, 0, nz - 1)

    # combined[i,j,k] = hctProbe[gxi[i],gyi[j],gzi[k]]
    #                  + dCorr_dN * corrN[...] + dCorr_dA * corrA[...]
    #                  + dCorr_dB * corrB[...]
    def gather(arr):
        return arr[gxi[:, None, None], gyi[None, :, None], gzi[None, None, :]]
    H = gather(hctProbe)
    N = gather(corrN)
    A = gather(corrA)
    B = gather(corrB)
    combined = H + dCorr_dN * N + dCorr_dA * A + dCorr_dB * B

    # tensor product accumulation
    w = bx[:, None, None] * by[None, :, None] * bz[None, None, :]
    val = jnp.sum(w * combined)
    return val


def hct_term(R_i_off, pos_i, pos_j, S_j):
    dx = pos_i - pos_j
    r = jnp.sqrt(jnp.dot(dx, dx))
    r = jnp.maximum(r, 1e-10)
    r_plus_S = r + S_j
    r_minus_S = jnp.abs(r - S_j)
    l = jnp.where(R_i_off > r_minus_S, 1.0/R_i_off,
                  1.0/jnp.maximum(r_minus_S, 1e-10))
    u = 1.0 / r_plus_S
    term = (l - u + 0.25*r*(u**2 - l**2)
            + 0.5/r*jnp.log(u/l)
            + 0.25*S_j**2/r*(l**2 - u**2))
    term = jnp.where(R_i_off < (S_j - r), term + 2.0*(1.0/R_i_off - l), term)
    term = jnp.where(R_i_off < r_plus_S, term, 0.0)
    return term


def born_obc(R, hct):
    R_off = R - DIELECTRIC_OFFSET
    psi = 0.5 * R_off * hct
    ta = OBC_ALPHA*psi - OBC_BETA*psi**2 + OBC_GAMMA*psi**3
    tv = jnp.tanh(ta)
    d = 1.0/R_off - tv/R
    br = jnp.where(d > 0, 1.0/d, R)
    return jnp.minimum(br, BORN_MAX)


def still_energy_pair(q_i, q_j, R_i, R_j, r2):
    RiRj = R_i * R_j
    f_gb = jnp.sqrt(r2 + RiRj * jnp.exp(-r2 / (4.0 * RiRj)))
    return PREFACTOR * q_i * q_j / f_gb


def pure_grid_energy_hermite(lig_pos_flat, lig_q, lig_r_arr, lig_s_arr,
                              include_sa=True):
    """Same as pure_grid_energy but uses triquintic hermite for the receptor
    HCT lookup. JAX reference for the plugin's method=3 path."""
    N_l = lig_q.shape[0]
    lig_pos = lig_pos_flat.reshape(N_l, 3)
    lig_R_off = lig_r_arr - DIELECTRIC_OFFSET
    lig_S = lig_R_off * lig_s_arr

    lig_born_list = []
    for i in range(N_l):
        hct = 0.0
        for j in range(N_l):
            if j == i: continue
            hct += hct_term(lig_R_off[i], lig_pos[i], lig_pos[j], lig_S[j])
        hct += grid_hct_triquintic(lig_pos[i], lig_R_off[i])
        lig_born_list.append(born_obc(lig_r_arr[i], hct))
    lig_born = jnp.stack(lig_born_list)

    E = 0.0
    for i in range(N_l):
        E += 0.5 * PREFACTOR * lig_q[i]**2 / lig_born[i]
        for j in range(i+1, N_l):
            dx = lig_pos[i] - lig_pos[j]
            r2 = jnp.dot(dx, dx)
            E += still_energy_pair(lig_q[i], lig_q[j],
                                   lig_born[i], lig_born[j], r2)
    if include_sa:
        for i in range(N_l):
            Rsolv = lig_r_arr[i] + PROBE_RADIUS
            ratio = lig_r_arr[i] / lig_born[i]
            E += SURFACE_TENSION * 4.0 * jnp.pi * Rsolv**2 * ratio**6
    return E


def pure_grid_energy(lig_pos_flat, lig_q, lig_r_arr, lig_s_arr,
                    include_sa=True):
    N_l = lig_q.shape[0]
    lig_pos = lig_pos_flat.reshape(N_l, 3)
    lig_R_off = lig_r_arr - DIELECTRIC_OFFSET
    lig_S = lig_R_off * lig_s_arr

    # Ligand Born radii: HCT_lig-lig pairwise + grid receptor descreening
    lig_born_list = []
    for i in range(N_l):
        hct = 0.0
        for j in range(N_l):
            if j == i: continue
            hct += hct_term(lig_R_off[i], lig_pos[i], lig_pos[j], lig_S[j])
        hct += grid_hct_bspline(lig_pos[i], lig_R_off[i])
        lig_born_list.append(born_obc(lig_r_arr[i], hct))
    lig_born = jnp.stack(lig_born_list)

    # GB self + pair
    E = 0.0
    for i in range(N_l):
        E += 0.5 * PREFACTOR * lig_q[i]**2 / lig_born[i]
        for j in range(i+1, N_l):
            dx = lig_pos[i] - lig_pos[j]
            r2 = jnp.dot(dx, dx)
            E += still_energy_pair(lig_q[i], lig_q[j],
                                   lig_born[i], lig_born[j], r2)

    # SA (ACE): SA_i = tension * 4π * (R_i + probe)² * (R_i / R_born_i)^6
    if include_sa:
        for i in range(N_l):
            Rsolv = lig_r_arr[i] + PROBE_RADIUS
            ratio = lig_r_arr[i] / lig_born[i]
            E += SURFACE_TENSION * 4.0 * jnp.pi * Rsolv**2 * ratio**6
    return E


# ---- Plugin pure-GRID system --------------------------------------------
HERMITE_METHODS = {gfp.INTERP_TRICUBIC_HERMITE, gfp.INTERP_TRIQUINTIC_HERMITE}

def build_plugin_pure_grid(include_sa=True, use_charges=True,
                            interp_method=None):
    if interp_method is None:
        interp_method = gfp.INTERP_TRICUBIC_BSPLINE
    q_use = lig_q if use_charges else np.zeros_like(lig_q)
    system = mm.System()
    for _ in range(n_lig):
        system.addParticle(12.0)
    f = gfp.IsolatedGBSAForce()
    f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
    f.setSoluteDielectric(SOLUTE_DIEL); f.setSolventDielectric(SOLVENT_DIEL)
    f.setIncludeSurfaceArea(include_sa)
    f.setReceptorMode(gfp.IsolatedGBSAForce.GRID)
    f.setNumAtoms(n_lig)
    for i in range(n_lig):
        f.setAtomParameters(i, float(q_use[i]), float(lig_r[i]),
                            float(lig_s[i]))
    grid_used = grid_wd if interp_method in HERMITE_METHODS else grid
    f.setDesolvationGrid(grid_used)
    f.setInterpolationMethod(interp_method)
    f.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]),
                                    float(rec_s[i]))
    f.setReceptorPositions(rec_pos.flatten().tolist())
    f.setReceptorBornRadiiBaseline([float(x) for x in R_rec_baseline])
    f.addParticleGroup("lig", list(range(n_lig)))
    system.addForce(f)
    return system, f


# ---- Run cross-check ----------------------------------------------------
def run_case(include_sa, use_charges, label, interp_method=None):
    interp_name = {
        gfp.INTERP_TRICUBIC_BSPLINE:   "tricubic_bspline",
        gfp.INTERP_TRIQUINTIC_HERMITE: "triquintic_hermite",
    }.get(interp_method or gfp.INTERP_TRICUBIC_BSPLINE, str(interp_method))
    print(f"\n############# case: {label}  interp={interp_name} "
          f"(include_sa={include_sa}, use_charges={use_charges}) ############")
    sys_, force = build_plugin_pure_grid(include_sa=include_sa,
                                          use_charges=use_charges,
                                          interp_method=interp_method)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    x0 = np.array(lig_pos, dtype=np.float64).flatten()
    ctx.setPositions(x0.reshape(-1, 3) * unit.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    E_plugin = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F_plugin = np.array(st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)).flatten()
    H_plugin = np.array(force.computeHessian(ctx))
    if H_plugin.ndim == 1:
        n_ = int(round(np.sqrt(H_plugin.size)))
        H_plugin = H_plugin.reshape(n_, n_)
    del ctx

    lp = jnp.array(x0, dtype=jnp.float64)
    lq_ = jnp.array(lig_q if use_charges else np.zeros_like(lig_q),
                    dtype=jnp.float64)
    lr_ = jnp.array(lig_r, dtype=jnp.float64)
    ls_ = jnp.array(lig_s, dtype=jnp.float64)

    jax_energy_fn = (pure_grid_energy_hermite
                     if interp_method == gfp.INTERP_TRIQUINTIC_HERMITE
                     else pure_grid_energy)
    E_jax = float(jax_energy_fn(lp, lq_, lr_, ls_, include_sa=include_sa))
    grad_jax = jax.grad(jax_energy_fn, argnums=0)(lp, lq_, lr_, ls_,
                                                   include_sa)
    F_jax = -np.array(grad_jax)
    H_jax = np.array(jax.hessian(jax_energy_fn, argnums=0)(lp, lq_, lr_, ls_,
                                                            include_sa))

    frob_p = float(np.linalg.norm(H_plugin))
    frob_j = float(np.linalg.norm(H_jax))
    frob_d = float(np.linalg.norm(H_plugin - H_jax))
    max_d  = float(np.max(np.abs(H_plugin - H_jax)))
    rel = frob_d / max(frob_p, 1e-10)
    print(f"  E_plugin={E_plugin:+.6f}  E_jax={E_jax:+.6f}  "
          f"dE={E_plugin-E_jax:+.4e}  rel={abs(E_plugin-E_jax)/max(abs(E_plugin),1e-10):.2e}")
    print(f"  ||F_p||={np.linalg.norm(F_plugin):.4e}  ||F_j||={np.linalg.norm(F_jax):.4e}  "
          f"||dF||={np.linalg.norm(F_plugin - F_jax):.4e}  "
          f"rel={np.linalg.norm(F_plugin-F_jax)/max(np.linalg.norm(F_plugin),1e-10):.2e}")
    print(f"  ||H_p||={frob_p:.4e}  ||H_j||={frob_j:.4e}  "
          f"||dH||={frob_d:.4e}  max|dH|={max_d:.4e}  rel={rel:.2e}")
    return H_plugin, H_jax

for interp in (gfp.INTERP_TRICUBIC_BSPLINE, gfp.INTERP_TRIQUINTIC_HERMITE):
    run_case(True,  True,  "SA_on__q_on",  interp_method=interp)
    run_case(False, True,  "SA_off_q_on",  interp_method=interp)
    run_case(True,  False, "SA_on__q_off", interp_method=interp)
    run_case(False, False, "SA_off_q_off", interp_method=interp)
H_plugin, H_jax = run_case(True, True, "SA_on__q_on",
                            interp_method=gfp.INTERP_TRICUBIC_BSPLINE)




# ---- Direct diagnostic: JAX vs plugin grid Hessian per atom ------------
# Compute jax.hessian(grid_hct_bspline)(x_i, R_i_off) for each ligand
# atom, then diff against what the plugin's Hessian assembly writes as
# gridHCTHessian * dE_dHCT for that atom's diagonal block.
#
# We can back out `plugin_gridHCTHessian * dE_dHCT[i]` for atom i by:
#     H_plugin_diag_block_i - [everything else in the diagonal block]
# For q=0, N_lig=1, everything else in the diagonal collapses because
# there are no ligand-ligand pair terms and no coupling M off-diagonals.
# So H_plugin(3x3) - (J^T M J diag part) - (SA_ratio term via J) = pure
# gridHCTHessian * dE_dHCT.
#
# We do the same JAX autodiff, then diff. If they disagree, gridHCTHessian
# (or its scaling into dE_dHCT) is the bug.

def build_plugin_pure_grid_N1(include_sa, use_charges):
    q_use = lig_q if use_charges else np.zeros_like(lig_q)
    system = mm.System()
    system.addParticle(12.0)
    f = gfp.IsolatedGBSAForce()
    f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
    f.setSoluteDielectric(SOLUTE_DIEL); f.setSolventDielectric(SOLVENT_DIEL)
    f.setIncludeSurfaceArea(include_sa)
    f.setReceptorMode(gfp.IsolatedGBSAForce.GRID)
    f.setNumAtoms(1)
    f.setAtomParameters(0, float(q_use[0]), float(lig_r[0]), float(lig_s[0]))
    f.setDesolvationGrid(grid)
    f.setInterpolationMethod(gfp.INTERP_TRICUBIC_BSPLINE)
    f.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]),
                                    float(rec_s[i]))
    f.setReceptorPositions(rec_pos.flatten().tolist())
    f.setReceptorBornRadiiBaseline([float(x) for x in R_rec_baseline])
    f.addParticleGroup("lig", [0])
    system.addForce(f)
    return system, f


def jax_single_atom_grid_energy(pos, q, R_intr, S, include_sa):
    """Pure single-atom pure-GRID energy: no ligand pair terms."""
    R_off = R_intr - DIELECTRIC_OFFSET
    hct = grid_hct_bspline(pos, R_off)
    R_born = born_obc(R_intr, hct)
    E = 0.5 * PREFACTOR * q**2 / R_born
    if include_sa:
        Rsolv = R_intr + PROBE_RADIUS
        E += SURFACE_TENSION * 4.0 * jnp.pi * Rsolv**2 * (R_intr / R_born)**6
    return E


print("\n" + "="*70)
print("N_lig=1 diagnostic: pure grid H_hct2 chain")
print("="*70)
for include_sa in (True, False):
    for use_charges in (True, False):
        if not include_sa and not use_charges:
            continue  # trivial
        sys_, force = build_plugin_pure_grid_N1(include_sa, use_charges)
        ctx = mm.Context(sys_, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName('CUDA'),
                         {'Precision': 'double'})
        x0 = np.array(lig_pos[0], dtype=np.float64)
        ctx.setPositions(x0.reshape(1, 3) * unit.nanometer)
        st = ctx.getState(getEnergy=True, getForces=True)
        E_p = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        F_p = np.array(st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)).flatten()
        H_p = np.array(force.computeHessian(ctx)).reshape(3, 3)
        del ctx

        q_val = float(lig_q[0]) if use_charges else 0.0
        E_j = float(jax_single_atom_grid_energy(
            jnp.array(x0), q_val, float(lig_r[0]), float(lig_s[0]),
            include_sa))
        F_j = -np.array(jax.grad(jax_single_atom_grid_energy, argnums=0)(
            jnp.array(x0), q_val, float(lig_r[0]), float(lig_s[0]),
            include_sa))
        H_j = np.array(jax.hessian(jax_single_atom_grid_energy, argnums=0)(
            jnp.array(x0), q_val, float(lig_r[0]), float(lig_s[0]),
            include_sa))

        lbl = f"SA={include_sa} q={use_charges}"
        print(f"\n  --- {lbl} ---")
        print(f"    E:  plugin={E_p:+.4e}  jax={E_j:+.4e}  "
              f"dE={E_p-E_j:+.3e}")
        print(f"    F:  ||dF||={np.linalg.norm(F_p-F_j):.3e}  "
              f"rel={np.linalg.norm(F_p-F_j)/max(np.linalg.norm(F_p),1e-10):.2e}")
        print(f"    H_plugin:\n{H_p}")
        print(f"    H_jax:\n{H_j}")
        print(f"    dH = H_p - H_j:\n{H_p - H_j}")
        print(f"    ||dH||_F={np.linalg.norm(H_p-H_j):.3e}  "
              f"max|dH|={np.max(np.abs(H_p-H_j)):.3e}  "
              f"rel={np.linalg.norm(H_p-H_j)/max(np.linalg.norm(H_p),1e-10):.2e}")

print("=== Energy ===")
print(f"  plugin: {E_plugin:.6f}")
print(f"  jax:    {E_jax:.6f}")
print(f"  diff:   {E_plugin - E_jax:+.6f}  rel={abs(E_plugin-E_jax)/abs(E_plugin):.2e}")

print("\n=== Forces (kJ/mol/nm) ===")
d = F_plugin - F_jax
print(f"  ||F_plugin||: {np.linalg.norm(F_plugin):.4e}")
print(f"  ||F_jax||:    {np.linalg.norm(F_jax):.4e}")
print(f"  ||dF||_F:     {np.linalg.norm(d):.4e}  max|dF|={np.max(np.abs(d)):.4e}  "
      f"rel={np.linalg.norm(d)/max(np.linalg.norm(F_plugin),1e-10):.2e}")

print("\n=== Hessian (kJ/mol/nm^2) ===")
d = H_plugin - H_jax
frob_p = float(np.linalg.norm(H_plugin))
frob_j = float(np.linalg.norm(H_jax))
frob_d = float(np.linalg.norm(d))
max_d  = float(np.max(np.abs(d)))
print(f"  ||H_plugin||_F: {frob_p:.4e}")
print(f"  ||H_jax||_F:    {frob_j:.4e}")
print(f"  ||H_plugin - H_jax||_F: {frob_d:.4e}  max|dH|={max_d:.4e}  "
      f"rel={frob_d/max(frob_p,1e-10):.2e}")

# Symmetry check
print("\n=== Symmetry ===")
print(f"  ||H_plugin - H_plugin.T||_F: {np.linalg.norm(H_plugin - H_plugin.T):.4e}")
print(f"  ||H_jax    - H_jax.T   ||_F: {np.linalg.norm(H_jax - H_jax.T):.4e}")

# -------------------------------------------------------------
# Block-level localization: reshape to (N, 3, N, 3) and diff
# per (i,j) atom-pair 3x3 sub-block.
# -------------------------------------------------------------
N = n_lig
Hp = H_plugin.reshape(N, 3, N, 3)
Hj = H_jax.reshape(N, 3, N, 3)
D = Hp - Hj

# Per-atom-pair block Frobenius and max|Δ|
block_fro = np.linalg.norm(D, axis=(1, 3))       # (N, N)
block_max = np.max(np.abs(D), axis=(1, 3))       # (N, N)
block_p_fro = np.linalg.norm(Hp, axis=(1, 3))    # for rel context

# Split diagonal vs off-diagonal
diag = np.diag(block_fro)
offdiag = block_fro.copy()
np.fill_diagonal(offdiag, 0.0)

print("\n=== Block localization ===")
print(f"  total ||dH||_F         = {np.linalg.norm(D):.4e}")
print(f"  diagonal blocks (Frobenius sum): {np.sqrt(np.sum(diag**2)):.4e}")
print(f"  offdiag  blocks (Frobenius sum): {np.sqrt(np.sum(offdiag**2)):.4e}")

# Worst blocks
flat = block_fro.flatten()
idx_sorted = np.argsort(flat)[::-1][:12]
print("\n  top 12 worst blocks (i,j):  ||dH_ij||_F   max|dH_ij|   ||H_plugin_ij||_F")
for k in idx_sorted:
    i, j = k // N, k % N
    print(f"    ({i:2d},{j:2d})   {block_fro[i,j]:.4e}   {block_max[i,j]:.4e}   "
          f"{block_p_fro[i,j]:.4e}")

# Diagonal-only vs off-diagonal-only view
print("\n  block Frobenius matrix (12x12):")
np.set_printoptions(precision=2, suppress=True, linewidth=160)
print(block_fro)

# xx/yy/zz/off checks on diagonal blocks
print("\n  Per-diagonal-block xx,yy,zz and (xy+xz+yz):")
print(f"    {'atom':4s} {'dxx':>10s} {'dyy':>10s} {'dzz':>10s} {'dxy':>10s} {'dxz':>10s} {'dyz':>10s}")
for i in range(N):
    dxx = D[i, 0, i, 0]; dyy = D[i, 1, i, 1]; dzz = D[i, 2, i, 2]
    dxy = D[i, 0, i, 1]; dxz = D[i, 0, i, 2]; dyz = D[i, 1, i, 2]
    print(f"    {i:4d} {dxx:+10.3f} {dyy:+10.3f} {dzz:+10.3f} "
          f"{dxy:+10.3f} {dxz:+10.3f} {dyz:+10.3f}")

# Is the delta ~ a scalar-times-Hjax_diag structure? (i.e., a missing prefactor)
diag_H_plugin = np.array([np.trace(Hp[i, :, i, :]) for i in range(N)])
diag_H_jax    = np.array([np.trace(Hj[i, :, i, :]) for i in range(N)])
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(np.abs(diag_H_jax) > 1e-3,
                     diag_H_plugin / diag_H_jax, np.nan)
print("\n  Per-atom trace ratio plugin / jax on diagonal 3x3:")
for i in range(N):
    print(f"    atom {i:2d}   trace(Hp)={diag_H_plugin[i]:+.4e}   "
          f"trace(Hj)={diag_H_jax[i]:+.4e}   ratio={ratio[i]:+.4f}")
