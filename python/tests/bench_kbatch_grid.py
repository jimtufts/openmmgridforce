"""K-batch minimization test WITH the full production grid stack:
IsolatedBondedForce + IsolatedNonbondedForce + IsolatedGBSAForce PAIRWISE
+ IsolatedGBSAForce GRID (receptor-descreening) + GridForce (LJr / LJa /
ELE bspline-arcsinh receptor grid) + IsolatedSiteForce (per-group site
restraint to break rigid-body degeneracy).

Checks:
  1. K=1 baseline vs L-BFGS from a docking-pose start.
  2. K-batch minimize on K replicas seeded with different perturbations
     of the K=1 minimum; each replica should reach the same energy.
  3. Full-H path vs block-diagonal path give identical minima.
"""
import os, sys, gc, time
import numpy as np
import openmm
import openmm.unit as unit
import gridforceplugin as gfp

sys.path.insert(0, '/home/jtufts/src/p312/algdock')
from AlGDock.mwe.utils import (
    extract_bonded_params, extract_nonbonded_params, extract_obc_params,
    build_isolated_bonded_force, build_isolated_nonbonded_force,
    build_isolated_gbsa_force, build_isolated_gbsa_force_grid,
    load_positions,
    compute_grid_geometry, create_grid_force_from_file,
)

PRMDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prmtopcrd')
LIGAND_PRMTOP  = os.path.join(PRMDIR, 'ligand.prmtop')
LIGAND_INPCRD  = os.path.join(PRMDIR, 'ligand.trans.inpcrd')
RECEPTOR_PRMTOP = os.path.join(PRMDIR, 'receptor.prmtop')
RECEPTOR_INPCRD = os.path.join(PRMDIR, 'receptor.trans.inpcrd')

GRID_SPACING  = 0.05
ARCSINH_SCALE = 1000.0
GRID_MARGIN   = 1.0
SITE_MAX_R    = 1.5

K = 4
TOL = 1e-3
MAX_ITER = 500

bonded_params  = extract_bonded_params(LIGAND_PRMTOP)
nb_params      = extract_nonbonded_params(LIGAND_PRMTOP)
lig_obc_params = extract_obc_params(LIGAND_PRMTOP)
rec_obc_params = extract_obc_params(RECEPTOR_PRMTOP)
N = bonded_params['n_atoms']
masses = np.array(bonded_params['masses'])
pos_lig = load_positions(LIGAND_INPCRD)
rec_positions = load_positions(RECEPTOR_INPCRD)

site_center = pos_lig.mean(axis=0)
grid_origin, grid_counts, grid_spacing_tuple = compute_grid_geometry(
    [pos_lig], margin=GRID_MARGIN, spacing=GRID_SPACING,
    site_center=site_center, site_max_R=SITE_MAX_R)
grid_origin = np.array(grid_origin)
grid_hcorner = grid_origin + (np.array(grid_counts) - 1) * np.array(grid_spacing_tuple)
grid_center = 0.5 * (grid_origin + grid_hcorner)
eff_R = min(float(0.5 * (grid_hcorner - grid_origin).min()), SITE_MAX_R)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'minimize_full_system', 'grid_cache',
                         'deriv0_pref3_asinh1000')
grid_files = {gk: os.path.join(CACHE_DIR, f'{gk}.grid') for gk in ('ljr','lja','charge')}
for f in grid_files.values():
    if not os.path.exists(f):
        raise SystemExit(f"missing grid file: {f}. Run minimize_full_system.py first.")

POSES_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'minimize_full_system',
                         'minimize_full_system_poses.npz')
d = np.load(POSES_NPZ); x0 = d['x0']; del d


def build_system(K_):
    system = openmm.System()
    for g in range(K_):
        for i in range(N): system.addParticle(masses[i])
    system.addForce(build_isolated_bonded_force(bonded_params, K_, N))
    system.addForce(build_isolated_nonbonded_force(nb_params, K_, N))
    system.addForce(build_isolated_gbsa_force(lig_obc_params, K_, N))
    system.addForce(build_isolated_gbsa_force_grid(
        lig_obc_params, rec_obc_params, rec_positions, K_, N,
        grid_origin=grid_origin, grid_counts=grid_counts,
        grid_spacing=GRID_SPACING,
        interpolation_method=1, bspline_prefilter_order=3))
    for name, gkey in [('LJr','ljr'),('LJa','lja'),('ELE','charge')]:
        gf = create_grid_force_from_file(
            grid_files[gkey], K_, N, gkey, name=name,
            interp_method=gfp.INTERP_TRICUBIC_BSPLINE,
            bspline_prefilter_order=3,
            arcsinh_scale=ARCSINH_SCALE, tile_memory_mb=0)
        system.addForce(gf)
    site = gfp.IsolatedSiteForce()
    site.setNumAtoms(N)
    site.setSiteCenter(float(grid_center[0]), float(grid_center[1]), float(grid_center[2]))
    site.setMaxRadius(eff_R)
    site.setForceConstant(10000.0)
    site.setAtomMasses([float(m) for m in masses])
    for g in range(K_):
        site.addParticleGroup(f'lig{g}', list(range(g*N, g*N + N)))
    system.addForce(site)
    return system


def make_ctx(system):
    return openmm.Context(system, openmm.VerletIntegrator(0.001),
                          openmm.Platform.getPlatformByName('CUDA'),
                          {'Precision': 'double'})


def snap(ctx):
    st = ctx.getState(getEnergy=True, getForces=True, getPositions=True)
    E = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = np.array(st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer))
    x = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return E, F, x


# ----- K=1 reference minimum from a docking pose -----
print(f"Ligand N={N} atoms, K={K}, grid={grid_counts}")
print("\nK=1 LE minimize from x0...")
ctx = make_ctx(build_system(1))
ctx.setPositions(x0 * unit.nanometer)
openmm.LocalEnergyMinimizer.minimize(ctx, 1e-4, 2000)
E_le, F_le, x_min = snap(ctx)
RMS_le = float(np.sqrt(np.mean(np.sum(F_le**2, axis=1))))
print(f"  LE: E={E_le:+.4f}  RMS={RMS_le:.2e}")
del ctx; gc.collect()

print("\nK=1 Newton minimize (from LE min for tight polish)...")
ctx = make_ctx(build_system(1))
ctx.setPositions(x_min * unit.nanometer)
gfp.NewtonMinimizer().minimize(ctx, TOL, MAX_ITER)
E_k1, F_k1, x_k1 = snap(ctx)
RMS_k1 = float(np.sqrt(np.mean(np.sum(F_k1**2, axis=1))))
print(f"  Newton (K=1): E={E_k1:+.4f}  RMS={RMS_k1:.2e}")
del ctx; gc.collect()

# ----- K-batch: seed replicas with different small perturbations of the K=1 min -----
rng = np.random.default_rng(20260708)
starts = [x_k1 + 0.001 * (1 + g) * rng.standard_normal(x_k1.shape) for g in range(K)]

def run_kbatch(block_diagonal):
    system = build_system(K)
    ctx = make_ctx(system)
    ctx.setPositions(np.concatenate(starts, axis=0) * unit.nanometer)
    E0, _, _ = snap(ctx)
    t0 = time.time()
    m = gfp.NewtonMinimizer()
    m.setKBatchBlockDiagonal(block_diagonal)
    m.minimize(ctx, TOL, MAX_ITER)
    dt = time.time() - t0
    E1, F1, x1 = snap(ctx)
    del ctx; gc.collect()
    return E0, E1, F1, x1, dt

print("\nK-batch Newton minimize (all replicas in one call)...")
print("  Full-H path:")
E0f, E1f, Ff, xf, dt_full = run_kbatch(False)
print(f"    Combined E: {E0f:+.3f} -> {E1f:+.3f}   time={dt_full:.2f}s")
print("  Block-diagonal path:")
E0b, E1b, Fb, xb, dt_bd = run_kbatch(True)
print(f"    Combined E: {E0b:+.3f} -> {E1b:+.3f}   time={dt_bd:.2f}s")
print(f"  Full vs BD final-E diff: {E1f - E1b:+.4e} kJ/mol")
print(f"  Full vs BD final-x max diff: {np.abs(xf - xb).max():.3e} nm")

print(f"\n{'replica':<8s} {'E_full':>10s}  {'E_bd':>10s}  {'RMS_full':>10s}  {'RMSD_vs_x_min':>15s}")
print("-" * 70)
all_ok = True
for g in range(K):
    x_g_full = xf[g*N:(g+1)*N]
    x_g_bd   = xb[g*N:(g+1)*N]
    F_g_full = Ff[g*N:(g+1)*N]
    RMS_g = float(np.sqrt(np.mean(np.sum(F_g_full**2, axis=1))))
    rmsd_g = float(np.sqrt(np.mean(np.sum((x_g_full - x_min)**2, axis=1))))
    ctx_solo = make_ctx(build_system(1))
    ctx_solo.setPositions(x_g_full * unit.nanometer)
    E_g_full, _, _ = snap(ctx_solo); del ctx_solo; gc.collect()
    ctx_solo = make_ctx(build_system(1))
    ctx_solo.setPositions(x_g_bd * unit.nanometer)
    E_g_bd, _, _ = snap(ctx_solo); del ctx_solo; gc.collect()
    dE = E_g_full - E_k1
    if abs(dE) > 1.0 or RMS_g > 1.0: all_ok = False
    print(f"lig{g:<5d} {E_g_full:>+10.4f}  {E_g_bd:>+10.4f}  {RMS_g:>10.2e}  {rmsd_g:>15.4e}")

print("\n" + ("OK: all replicas match K=1 minimum within 1 kJ/mol"
              if all_ok else "FAIL: at least one replica did not converge to K=1 minimum"))
