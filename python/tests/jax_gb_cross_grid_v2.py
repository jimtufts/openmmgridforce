#!/usr/bin/env python
"""JAX prototype v2: grid GB cross term on FULL receptor + Astex dock6 poses.

Improvements over v1:
  - fully vectorized (jax.vmap) so N_R = full receptor (~thousands) fits
  - loads real docked poses from Astex mol2 (100 poses spanning the pocket)
  - grid box auto-sized around the pose ensemble (not just crystal)
  - reports per-pose and aggregate error statistics
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DIELECTRIC_OFFSET = 0.009
OBC_ALPHA = 1.0
OBC_BETA = 0.8
OBC_GAMMA = 4.85
PREFACTOR = -138.935456 * (1.0 / 1.0 - 1.0 / 78.5)

# -------- HCT / OBC2 (vectorized) --------------------------------------------

def hct_pair_matrix(R_i_off_arr, pos_i_arr, pos_j_arr, S_j_arr):
    """Return HCT contributions to each atom i from each atom j.

    R_i_off_arr : (N_i,)
    pos_i_arr   : (N_i, 3)
    pos_j_arr   : (N_j, 3)
    S_j_arr     : (N_j,)
    Returns (N_i, N_j) matrix of hct terms. Diagonal (if any) is caller's job.
    """
    dx = pos_i_arr[:, None, :] - pos_j_arr[None, :, :]   # (Ni, Nj, 3)
    r = jnp.sqrt(jnp.sum(dx * dx, axis=-1))              # (Ni, Nj)
    r = jnp.maximum(r, 1e-10)
    r_plus_S = r + S_j_arr[None, :]
    r_minus_S = jnp.abs(r - S_j_arr[None, :])
    R_i_off = R_i_off_arr[:, None]
    l = jnp.where(R_i_off > r_minus_S,
                  1.0 / R_i_off,
                  1.0 / jnp.maximum(r_minus_S, 1e-10))
    u = 1.0 / r_plus_S
    term = l - u + 0.25*r*(u**2 - l**2) + 0.5/r*jnp.log(u/l) \
           + 0.25*S_j_arr[None, :]**2/r*(l**2 - u**2)
    term = jnp.where(R_i_off < (S_j_arr[None, :] - r),
                     term + 2.0*(1.0/R_i_off - l), term)
    term = jnp.where(R_i_off < r_plus_S, term, 0.0)
    return term


def born_obc(R, hct):
    R_off = R - DIELECTRIC_OFFSET
    psi = 0.5 * R_off * hct
    ta = OBC_ALPHA*psi - OBC_BETA*psi**2 + OBC_GAMMA*psi**3
    tv = jnp.tanh(ta)
    d = 1.0/R_off - tv/R
    br = jnp.where(d > 0, 1.0/d, R)
    return jnp.minimum(br, 50.0)


# -------- receptor / ligand Born radii ---------------------------------------

def receptor_born_radii_isolated(rec_pos, rec_r, rec_s):
    """Frozen receptor Born radii — computed WITHOUT ligand present.

    HCT sum over receptor atoms only, mask out i==j.
    """
    R_off = rec_r - DIELECTRIC_OFFSET
    S = R_off * rec_s
    hct_ij = hct_pair_matrix(R_off, rec_pos, rec_pos, S)  # (Nr, Nr)
    N_r = rec_pos.shape[0]
    hct_ij = hct_ij.at[jnp.arange(N_r), jnp.arange(N_r)].set(0.0)
    hct_i = hct_ij.sum(axis=1)
    return jax.vmap(born_obc)(rec_r, hct_i)


def receptor_born_radii_with_ligand(rec_pos, rec_r, rec_s,
                                     lig_pos, lig_r, lig_s):
    """Receptor Born radii with ligand present (fully-coupled reference)."""
    R_off_r = rec_r - DIELECTRIC_OFFSET
    S_r = R_off_r * rec_s
    R_off_l = lig_r - DIELECTRIC_OFFSET
    S_l = R_off_l * lig_s
    # receptor-receptor (mask diag)
    hct_rr = hct_pair_matrix(R_off_r, rec_pos, rec_pos, S_r)
    N_r = rec_pos.shape[0]
    hct_rr = hct_rr.at[jnp.arange(N_r), jnp.arange(N_r)].set(0.0)
    hct_r_from_r = hct_rr.sum(axis=1)
    # receptor-ligand
    hct_rl = hct_pair_matrix(R_off_r, rec_pos, lig_pos, S_l)
    hct_r_from_l = hct_rl.sum(axis=1)
    return jax.vmap(born_obc)(rec_r, hct_r_from_r + hct_r_from_l)


def ligand_born_radii_with_receptor(lig_pos, lig_r, lig_s,
                                     rec_pos, rec_r, rec_s):
    """Context-dependent ligand Born radii."""
    R_off_l = lig_r - DIELECTRIC_OFFSET
    S_l = R_off_l * lig_s
    R_off_r = rec_r - DIELECTRIC_OFFSET
    S_r = R_off_r * rec_s
    # ligand-ligand (mask diag)
    hct_ll = hct_pair_matrix(R_off_l, lig_pos, lig_pos, S_l)
    N_l = lig_pos.shape[0]
    hct_ll = hct_ll.at[jnp.arange(N_l), jnp.arange(N_l)].set(0.0)
    hct_l_from_l = hct_ll.sum(axis=1)
    # ligand-receptor
    hct_lr = hct_pair_matrix(R_off_l, lig_pos, rec_pos, S_r)
    hct_l_from_r = hct_lr.sum(axis=1)
    return jax.vmap(born_obc)(lig_r, hct_l_from_l + hct_l_from_r)


# -------- G2 grid: build and lookup ------------------------------------------

def phi_gb_at_points(query_pts, R_query,
                     rec_pos, rec_q, rec_R_frozen,
                     chunk=4096):
    """Sum_j q_j / f_gb(|r - r_j|, R_query, R_j_frozen).

    query_pts : (M, 3)      grid nodes or ligand positions
    R_query   : scalar or (M,) probe/ligand radius
    Returns (M,).
    """
    R_query_arr = jnp.broadcast_to(jnp.asarray(R_query),
                                    (query_pts.shape[0],))

    def one_chunk(q_pts, R_q):
        dx = q_pts[:, None, :] - rec_pos[None, :, :]      # (M, Nr, 3)
        r2 = jnp.sum(dx * dx, axis=-1)                    # (M, Nr)
        RiRj = R_q[:, None] * rec_R_frozen[None, :]       # (M, Nr)
        f_gb = jnp.sqrt(r2 + RiRj * jnp.exp(-r2 / (4.0 * RiRj)))
        return jnp.sum(rec_q[None, :] / f_gb, axis=1)

    M = query_pts.shape[0]
    out = np.zeros(M)
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        out[start:end] = np.asarray(
            jax.jit(one_chunk)(query_pts[start:end], R_query_arr[start:end]))
    return jnp.array(out)


def build_g2_grid(rec_pos, rec_q, rec_R_frozen,
                   box_lo, box_hi, spacing, probe_radii,
                   chunk=4096):
    nx = int(np.ceil((box_hi[0] - box_lo[0]) / spacing)) + 1
    ny = int(np.ceil((box_hi[1] - box_lo[1]) / spacing)) + 1
    nz = int(np.ceil((box_hi[2] - box_lo[2]) / spacing)) + 1
    n_probes = len(probe_radii)
    n_nodes = nx * ny * nz
    print(f"  G2 dimensions: {n_probes} probes x {nx}x{ny}x{nz} = "
          f"{n_probes*n_nodes:,} nodes ({n_probes*n_nodes*8/1e6:.1f} MB float64)")
    origin = jnp.array(box_lo)
    xs = origin[0] + spacing * jnp.arange(nx)
    ys = origin[1] + spacing * jnp.arange(ny)
    zs = origin[2] + spacing * jnp.arange(nz)
    xx, yy, zz = jnp.meshgrid(xs, ys, zs, indexing='ij')
    nodes = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    Phi = np.zeros((n_probes, n_nodes))
    for k, R_probe in enumerate(probe_radii):
        Phi[k] = np.asarray(phi_gb_at_points(nodes, float(R_probe),
                                              rec_pos, rec_q, rec_R_frozen,
                                              chunk=chunk))
    return (jnp.array(Phi).reshape(n_probes, nx, ny, nz),
            dict(nx=nx, ny=ny, nz=nz, origin=origin, spacing=spacing,
                 probe_radii=jnp.array(probe_radii)))


def _trilinear(field, pos, meta):
    """field: (nx, ny, nz). pos: (3,). Returns scalar."""
    fx = (pos[0] - meta['origin'][0]) / meta['spacing']
    fy = (pos[1] - meta['origin'][1]) / meta['spacing']
    fz = (pos[2] - meta['origin'][2]) / meta['spacing']
    ix = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, meta['nx'] - 2)
    iy = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, meta['ny'] - 2)
    iz = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, meta['nz'] - 2)
    tx = fx - ix; ty = fy - iy; tz = fz - iz
    c00 = field[ix, iy, iz]         * (1 - tx) + field[ix + 1, iy, iz]         * tx
    c01 = field[ix, iy, iz + 1]     * (1 - tx) + field[ix + 1, iy, iz + 1]     * tx
    c10 = field[ix, iy + 1, iz]     * (1 - tx) + field[ix + 1, iy + 1, iz]     * tx
    c11 = field[ix, iy + 1, iz + 1] * (1 - tx) + field[ix + 1, iy + 1, iz + 1] * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


def phi_gb_lookup(pos, R, Phi_stack, meta):
    """Lookup Phi_GB^R at (pos, R) with linear-in-R interp."""
    probes = meta['probe_radii']
    R_c = jnp.clip(R, probes[0], probes[-1])
    kL = jnp.clip(jnp.searchsorted(probes, R_c, side='right') - 1,
                  0, len(probes) - 2)
    kU = kL + 1
    tR = (R_c - probes[kL]) / (probes[kU] - probes[kL])
    phi_lo = _trilinear(Phi_stack[kL], pos, meta)
    phi_hi = _trilinear(Phi_stack[kU], pos, meta)
    return phi_lo * (1 - tR) + phi_hi * tR


# -------- cross energies -----------------------------------------------------

def cross_energy_atomistic_vec(lig_pos, lig_q, R_i,
                                 rec_pos, rec_q, R_j):
    """Exact atomistic cross energy with given Born radii.

    lig_pos: (Nl, 3), lig_q: (Nl,), R_i: (Nl,)
    rec_pos: (Nr, 3), rec_q: (Nr,), R_j: (Nr,)
    """
    dx = lig_pos[:, None, :] - rec_pos[None, :, :]      # (Nl, Nr, 3)
    r2 = jnp.sum(dx * dx, axis=-1)                       # (Nl, Nr)
    RiRj = R_i[:, None] * R_j[None, :]                   # (Nl, Nr)
    f_gb = jnp.sqrt(r2 + RiRj * jnp.exp(-r2 / (4.0 * RiRj)))
    return jnp.sum(PREFACTOR * lig_q[:, None] * rec_q[None, :] / f_gb)


def cross_energy_grid_vec(lig_pos, lig_q, R_i, Phi_stack, meta):
    phi_at_atoms = jax.vmap(lambda p, R: phi_gb_lookup(p, R, Phi_stack, meta))(
        lig_pos, R_i)
    return jnp.sum(PREFACTOR * lig_q * phi_at_atoms)


# -------- driver -------------------------------------------------------------

def load_prmtop_params(prmtop, inpcrd):
    import openmm as mm
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile
    from openmm import unit
    prm = AmberPrmtopFile(prmtop)
    ic = AmberInpcrdFile(inpcrd)
    system = prm.createSystem(nonbondedMethod=mm.app.NoCutoff,
                              implicitSolvent=mm.app.OBC2)
    gb = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.GBSAOBCForce):
            gb = f
            break
    n = system.getNumParticles()
    pos = np.array(ic.positions.value_in_unit(unit.nanometer))
    q = np.array([gb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(n)])
    r = np.array([gb.getParticleParameters(i)[1].value_in_unit(unit.nanometer)         for i in range(n)])
    s = np.array([gb.getParticleParameters(i)[2]                                       for i in range(n)])
    return pos, q, r, s


def load_dock6_mol2(mol2_file):
    """Returns list of (N, 3) arrays in nm."""
    if mol2_file.endswith('.gz'):
        import gzip
        content = gzip.open(mol2_file, 'rt').read()
    else:
        content = open(mol2_file).read()
    blocks = content.split('@<TRIPOS>MOLECULE')
    poses = []
    for block in blocks[1:]:
        if '@<TRIPOS>ATOM' not in block:
            continue
        atom_start = block.index('@<TRIPOS>ATOM') + len('@<TRIPOS>ATOM')
        remaining = block[atom_start:].lstrip('\n')
        end_idx = remaining.find('@<TRIPOS>')
        if end_idx < 0:
            end_idx = len(remaining)
        atom_lines = remaining[:end_idx].strip().split('\n')
        coords = []
        for line in atom_lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                except (ValueError, IndexError):
                    continue
        if coords:
            poses.append(np.array(coords) * 0.1)  # Angstrom -> nm
    return poses


def main():
    pdb = os.environ.get('PDB', '1r1h')
    astex = '/home/dminh/backup/AstexDiv_xtal'
    lig_prmtop = f'{astex}/1-build/{pdb}/ligand.prmtop'
    lig_inpcrd = f'{astex}/3-grids/{pdb}/ligand.trans.inpcrd'
    rec_prmtop = f'{astex}/1-build/{pdb}/receptor.prmtop'
    rec_inpcrd = f'{astex}/3-grids/{pdb}/receptor.trans.inpcrd'
    poses_file = f'{astex}/4-UCSF_dock6/{pdb}/xtal_plus_dock6_scored.mol2'
    max_poses = int(os.environ.get('MAX_POSES', '30'))

    print(f'PDB: {pdb}')
    l_pos_native, l_q, l_r, l_s = load_prmtop_params(lig_prmtop, lig_inpcrd)
    r_pos, r_q, r_r, r_s = load_prmtop_params(rec_prmtop, rec_inpcrd)
    N_l = len(l_q); N_r = len(r_q)
    print(f'Ligand atoms: {N_l}   Receptor atoms: {N_r} (full)')

    poses = load_dock6_mol2(poses_file)
    print(f'Loaded {len(poses)} docked poses; using first {max_poses}')
    poses = poses[:max_poses]
    # Sanity check: all poses must have N_l atoms
    poses = [p for p in poses if p.shape[0] == N_l]
    print(f'  {len(poses)} poses match N_l={N_l}\n')

    # JAX arrays
    lq_j = jnp.array(l_q)
    lr_j = jnp.array(l_r)
    ls_j = jnp.array(l_s)
    rp_j = jnp.array(r_pos)
    rq_j = jnp.array(r_q)
    rr_j = jnp.array(r_r)
    rs_j = jnp.array(r_s)

    # 1. Frozen receptor Born radii (once, offline)
    print('Computing frozen receptor Born radii (full receptor, no ligand)...',
          flush=True)
    t0 = time.time()
    R_j_frozen = receptor_born_radii_isolated(rp_j, rr_j, rs_j)
    R_j_frozen.block_until_ready()
    print(f'  done in {time.time()-t0:.1f}s. '
          f'R_j range {float(R_j_frozen.min()):.3f}-{float(R_j_frozen.max()):.3f} nm\n',
          flush=True)

    # 2. Grid box around pose ensemble + margin
    all_pos = np.vstack(poses)
    margin = 0.5  # nm
    box_lo = (all_pos.min(axis=0) - margin).tolist()
    box_hi = (all_pos.max(axis=0) + margin).tolist()
    box_size = [box_hi[i] - box_lo[i] for i in range(3)]
    spacing = float(os.environ.get('SPACING', '0.05'))
    print(f'Grid box: {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} nm, '
          f'spacing {spacing} nm')

    # Get a preliminary sense of ligand R_i range from the crystal pose
    print('Sizing radius axis from crystal pose ligand R_i...', flush=True)
    R_i_crystal = ligand_born_radii_with_receptor(
        jnp.array(l_pos_native), lr_j, ls_j, rp_j, rr_j, rs_j)
    R_lo = float(R_i_crystal.min()) * 0.85
    R_hi = float(R_i_crystal.max()) * 1.15
    n_probes = int(os.environ.get('N_PROBES', '6'))
    probe_radii = np.linspace(R_lo, R_hi, n_probes)
    print(f'  R_i range on crystal: {float(R_i_crystal.min()):.3f}-{float(R_i_crystal.max()):.3f} nm')
    print(f'  probe axis: {n_probes} slices from {R_lo:.3f} to {R_hi:.3f} nm\n')

    # 3. Build G2
    print('Building G2 (Phi_GB^R at each node x probe, full receptor)...',
          flush=True)
    t0 = time.time()
    Phi_stack, meta = build_g2_grid(rp_j, rq_j, R_j_frozen,
                                     box_lo, box_hi, spacing, probe_radii,
                                     chunk=1024)
    print(f'  done in {time.time()-t0:.1f}s\n', flush=True)

    # 4. Evaluate over all poses
    print(f'{"pose":>5s}  {"exact_full":>12s}  {"exact_froz":>12s}  '
          f'{"grid":>12s}  {"grid-froz":>12s}  {"froz-full":>12s}  '
          f'{"min_dR_i":>10s}')
    print()
    residuals_grid_minus_froz = []
    residuals_froz_minus_full = []
    residuals_total = []
    all_full = []
    all_froz = []
    all_grid = []
    for pk, pose in enumerate(poses):
        lp_j = jnp.array(pose)
        R_i_here = ligand_born_radii_with_receptor(lp_j, lr_j, ls_j,
                                                    rp_j, rr_j, rs_j)
        R_j_here = receptor_born_radii_with_ligand(rp_j, rr_j, rs_j,
                                                    lp_j, lr_j, ls_j)
        E_full = float(cross_energy_atomistic_vec(lp_j, lq_j, R_i_here,
                                                   rp_j, rq_j, R_j_here))
        E_froz = float(cross_energy_atomistic_vec(lp_j, lq_j, R_i_here,
                                                   rp_j, rq_j, R_j_frozen))
        E_grid = float(cross_energy_grid_vec(lp_j, lq_j, R_i_here,
                                              Phi_stack, meta))
        # Diagnostic: how different are frozen vs fully-coupled R_j?
        min_dR = float(jnp.min(R_j_here - R_j_frozen))
        f = 1 / 4.184
        d_g_f = (E_grid - E_froz) * f
        d_fz_f = (E_froz - E_full) * f
        residuals_grid_minus_froz.append(d_g_f)
        residuals_froz_minus_full.append(d_fz_f)
        residuals_total.append((E_grid - E_full) * f)
        all_full.append(E_full * f)
        all_froz.append(E_froz * f)
        all_grid.append(E_grid * f)
        if pk < 15 or pk % 5 == 0:
            print(f'  {pk:>3d}  {E_full*f:>+12.3f}  {E_froz*f:>+12.3f}  '
                  f'{E_grid*f:>+12.3f}  {d_g_f:>+12.4f}  {d_fz_f:>+12.4f}  '
                  f'{min_dR:>10.4f}')

    r_gf = np.array(residuals_grid_minus_froz)
    r_ff = np.array(residuals_froz_minus_full)
    r_tot = np.array(residuals_total)

    def stats(x, label):
        return (f'{label}: mean={np.mean(x):+.4f}, RMS={np.sqrt(np.mean(x**2)):.4f}, '
                f'max|={np.max(np.abs(x)):.4f}, min={np.min(x):+.4f}, max={np.max(x):+.4f}')

    print()
    print('=== error decomposition across all poses (kcal/mol) ===')
    print(f'  {stats(r_gf,  "grid - exact(frozen R_j) [interp err]")}')
    print(f'  {stats(r_ff,  "exact(frozen) - exact(full) [§3 err] ")}')
    print(f'  {stats(r_tot, "grid - exact(full) [TOTAL]          ")}')

    # Fit linear corrections and write JSON summary.
    all_full_a = np.array(all_full)
    all_froz_a = np.array(all_froz)
    all_grid_a = np.array(all_grid)
    m_fg, c_fg = np.polyfit(all_grid_a, all_full_a, 1)
    r_fg = float(np.corrcoef(all_grid_a, all_full_a)[0, 1])
    m_ff, c_ff = np.polyfit(all_froz_a, all_full_a, 1)
    r_ff_corr = float(np.corrcoef(all_froz_a, all_full_a)[0, 1])
    print()
    print('=== linear correction fits (kcal/mol) ===')
    print(f'  E_full = {m_fg:.4f} * E_grid + {c_fg:+.4f}   Pearson r = {r_fg:.4f}')
    print(f'  E_full = {m_ff:.4f} * E_frozen + {c_ff:+.4f}  Pearson r = {r_ff_corr:.4f}')

    out_json = os.environ.get('OUTPUT_JSON')
    if out_json:
        import json
        summary = {
            'pdb': pdb,
            'N_lig': int(N_l),
            'N_rec': int(N_r),
            'n_poses': len(poses),
            'linear_fit_full_vs_grid': {
                'slope': float(m_fg),
                'intercept': float(c_fg),
                'pearson_r': r_fg,
            },
            'linear_fit_full_vs_frozen': {
                'slope': float(m_ff),
                'intercept': float(c_ff),
                'pearson_r': r_ff_corr,
            },
            'per_pose': {
                'exact_full': all_full_a.tolist(),
                'exact_frozen_Rj': all_froz_a.tolist(),
                'grid': all_grid_a.tolist(),
            },
            'error_stats_kcal': {
                'grid_minus_frozen_mean': float(np.mean(r_gf)),
                'grid_minus_frozen_rms': float(np.sqrt(np.mean(r_gf ** 2))),
                'frozen_minus_full_mean': float(np.mean(r_ff)),
                'frozen_minus_full_rms': float(np.sqrt(np.mean(r_ff ** 2))),
                'grid_minus_full_mean': float(np.mean(r_tot)),
                'grid_minus_full_rms': float(np.sqrt(np.mean(r_tot ** 2))),
            },
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f'\nwrote {out_json}')


if __name__ == '__main__':
    main()
