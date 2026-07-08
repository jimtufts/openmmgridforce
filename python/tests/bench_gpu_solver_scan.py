"""Scan solver wall time across K to find the CPU-vs-GPU Cholesky crossover.

Uses the same production stack (bonded + NB + GBSA PAIRWISE + GBSA GRID +
receptor grids + site restraint).  Each K value seeds K replicas near the
K=1 minimum, runs both CPU and GPU LM-Cholesky, and reports wall time and
final combined energy for each.
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
LIGAND_PRMTOP   = os.path.join(PRMDIR, 'ligand.prmtop')
LIGAND_INPCRD   = os.path.join(PRMDIR, 'ligand.trans.inpcrd')
RECEPTOR_PRMTOP = os.path.join(PRMDIR, 'receptor.prmtop')
RECEPTOR_INPCRD = os.path.join(PRMDIR, 'receptor.trans.inpcrd')

GRID_SPACING  = 0.05
ARCSINH_SCALE = 1000.0
GRID_MARGIN   = 1.0
SITE_MAX_R    = 1.5

TOL = 1e-3
MAX_ITER = 200
K_VALUES = [4, 8, 16, 32]

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


# One K=1 minimum, reused as seed
print("K=1 baseline...")
ctx = make_ctx(build_system(1))
ctx.setPositions(x0 * unit.nanometer)
openmm.LocalEnergyMinimizer.minimize(ctx, 1e-4, 2000)
st = ctx.getState(getPositions=True, getEnergy=True)
x_min = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
E_k1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
print(f"  K=1 E={E_k1:+.4f}")
del ctx; gc.collect()

rng = np.random.default_rng(20260708)


def time_solver(K_, solver):
    seeds = [x_min + 0.001 * (1 + g) * rng.standard_normal(x_min.shape) for g in range(K_)]
    ctx = make_ctx(build_system(K_))
    ctx.setPositions(np.concatenate(seeds, axis=0) * unit.nanometer)
    st = ctx.getState(getEnergy=True)
    E0 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    m = gfp.NewtonMinimizer()
    m.setInnerSolver(solver)
    t0 = time.time()
    m.minimize(ctx, TOL, MAX_ITER)
    dt = time.time() - t0
    st = ctx.getState(getEnergy=True)
    E1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx; gc.collect()
    return dt, E1

# Warm-up CUDA JIT
print("warm-up...")
_ = time_solver(4, gfp.NewtonMinimizer.LMCholesky)
_ = time_solver(4, gfp.NewtonMinimizer.GPULMCholesky)

print(f"\n{'K':>4s}  {'n':>6s}  {'CPU_time':>10s}  {'GPU_time':>10s}  {'GPU/CPU':>10s}  "
      f"{'E_cpu':>12s}  {'E_gpu':>12s}")
print("-" * 90)
for K_ in K_VALUES:
    n = 3 * K_ * N
    dt_cpu, E_cpu = time_solver(K_, gfp.NewtonMinimizer.LMCholesky)
    dt_gpu, E_gpu = time_solver(K_, gfp.NewtonMinimizer.GPULMCholesky)
    ratio = dt_gpu / dt_cpu if dt_cpu > 0 else float('inf')
    print(f"{K_:>4d}  {n:>6d}  {dt_cpu:>9.2f}s  {dt_gpu:>9.2f}s  {ratio:>9.2f}x  "
          f"{E_cpu:>+12.3f}  {E_gpu:>+12.3f}")
