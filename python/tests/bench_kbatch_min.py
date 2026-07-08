"""K-batch minimization accuracy test.

Builds a system with K replicas of the same ligand (bonded + intra-lig NB +
intra-lig GBSA PAIRWISE), seeds each replica with a different perturbation of
the K=1 minimum, runs NewtonMinimizer once on the K-replica system, and
checks that each replica reaches the same per-replica energy that a K=1
minimize from the same seed would produce.

If the K-batch code path silently drops any per-group Hessian contribution,
different replicas will fall to different (non-)minima and this test will
fail.
"""
import os, sys, gc
import numpy as np
import openmm
import openmm.unit as unit
import gridforceplugin as gfp

sys.path.insert(0, '/home/jtufts/src/p312/algdock')
from AlGDock.mwe.utils import (
    extract_bonded_params, extract_nonbonded_params, extract_obc_params,
    build_isolated_bonded_force, build_isolated_nonbonded_force,
    build_isolated_gbsa_force,
    load_positions,
)

PRMDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prmtopcrd')
LIGAND_PRMTOP = os.path.join(PRMDIR, 'ligand.prmtop')
LIGAND_INPCRD = os.path.join(PRMDIR, 'ligand.trans.inpcrd')

K = 4
TOL = 1e-3
MAX_ITER = 500

bonded_params  = extract_bonded_params(LIGAND_PRMTOP)
nb_params      = extract_nonbonded_params(LIGAND_PRMTOP)
lig_obc_params = extract_obc_params(LIGAND_PRMTOP)
N = bonded_params['n_atoms']
masses = np.array(bonded_params['masses'])
pos_lig = load_positions(LIGAND_INPCRD)


def build_system_k1():
    """K=1 baseline system, exactly one replica."""
    system = openmm.System()
    for i in range(N): system.addParticle(masses[i])
    system.addForce(build_isolated_bonded_force(bonded_params, 1, N))
    system.addForce(build_isolated_nonbonded_force(nb_params, 1, N))
    system.addForce(build_isolated_gbsa_force(lig_obc_params, 1, N))
    return system


def build_system_kbatch():
    """K replicas of the same ligand template.  Particle indices for
    replica g are [g*N, g*N+1, ..., g*N+N-1] (matches utils helpers)."""
    system = openmm.System()
    for g in range(K):
        for i in range(N):
            system.addParticle(masses[i])
    system.addForce(build_isolated_bonded_force(bonded_params, K, N))
    system.addForce(build_isolated_nonbonded_force(nb_params, K, N))
    system.addForce(build_isolated_gbsa_force(lig_obc_params, K, N))
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


# ----- K=1 reference minimum -----
print(f"Ligand N={N} atoms, K={K}")
print("\nK=1 baseline minimize (LE)...")
ctx = make_ctx(build_system_k1())
ctx.setPositions(pos_lig * unit.nanometer)
openmm.LocalEnergyMinimizer.minimize(ctx, 1e-6, 5000)
E_le, F_le, x_min = snap(ctx)
RMS_le = float(np.sqrt(np.mean(np.sum(F_le**2, axis=1))))
print(f"  LE: E={E_le:+.4f}  RMS={RMS_le:.2e}")
del ctx; gc.collect()

# ----- K=1 Newton reference -----
print("\nK=1 Newton minimize...")
ctx = make_ctx(build_system_k1())
ctx.setPositions(pos_lig * unit.nanometer)
gfp.NewtonMinimizer().minimize(ctx, TOL, MAX_ITER)
E_k1, F_k1, x_k1 = snap(ctx)
RMS_k1 = float(np.sqrt(np.mean(np.sum(F_k1**2, axis=1))))
print(f"  Newton (K=1): E={E_k1:+.4f}  RMS={RMS_k1:.2e}")
del ctx; gc.collect()

# ----- K-batch: seed replicas with different perturbations -----
rng = np.random.default_rng(20260708)
starts = []
for g in range(K):
    # different perturbation magnitude per replica, so any per-group Hessian
    # confusion shows as different final energies
    scale = 0.001 * (1 + g)   # 0.001 nm ... 0.004 nm
    starts.append(x_min + scale * rng.standard_normal(x_min.shape))

def run_kbatch(block_diagonal):
    system = build_system_kbatch()
    ctx = make_ctx(system)
    ctx.setPositions(np.concatenate(starts, axis=0) * unit.nanometer)
    E0, _, _ = snap(ctx)
    import time as _t; t0 = _t.time()
    m = gfp.NewtonMinimizer()
    m.setKBatchBlockDiagonal(block_diagonal)
    m.minimize(ctx, TOL, MAX_ITER)
    dt = _t.time() - t0
    E1, F1, x1 = snap(ctx)
    del ctx; gc.collect()
    return E0, E1, F1, x1, dt

print("\nK-batch Newton minimize (all replicas in one call)...")
print("  Full-H path:")
E0f, E1f, F_batch_after, x_batch, dt_full = run_kbatch(False)
print(f"    Combined E: {E0f:+.3f} -> {E1f:+.3f}   time={dt_full:.2f}s")
print("  Block-diagonal path:")
E0b, E1b, F_batch_bd,  x_batch_bd, dt_bd = run_kbatch(True)
print(f"    Combined E: {E0b:+.3f} -> {E1b:+.3f}   time={dt_bd:.2f}s")
print(f"  Full vs BD final-E diff: {E1f - E1b:+.4e} kJ/mol")
print(f"  Full vs BD final-x max diff: {np.abs(x_batch - x_batch_bd).max():.3e} nm")

# ----- Compare per-replica -----
print(f"\n{'replica':<8s} {'E_final':>10s}  {'RMS':>10s}  {'|F|_max':>10s}  "
      f"{'RMSD_vs_x_min':>15s}")
print("-" * 65)
all_ok = True
for g in range(K):
    x_g = x_batch[g*N:(g+1)*N]
    F_g = F_batch_after[g*N:(g+1)*N]
    RMS_g = float(np.sqrt(np.mean(np.sum(F_g**2, axis=1))))
    Fmax_g = float(np.max(np.linalg.norm(F_g, axis=1)))
    rmsd_g = float(np.sqrt(np.mean(np.sum((x_g - x_min)**2, axis=1))))
    # Per-replica energy via IsolatedBondedForce group API (rough check)
    # Compute per-replica-energy by evaluating each replica alone
    ctx_solo = make_ctx(build_system_k1())
    ctx_solo.setPositions(x_g * unit.nanometer)
    E_g, _, _ = snap(ctx_solo)
    del ctx_solo; gc.collect()
    dE = E_g - E_k1
    marker = "" if (abs(dE) < 1.0 and RMS_g < 1.0) else "  << mismatch"
    if abs(dE) > 1.0 or RMS_g > 1.0:
        all_ok = False
    print(f"{'lig'+str(g):<8s} {E_g:>+10.4f}  {RMS_g:>10.2e}  {Fmax_g:>10.2e}  "
          f"{rmsd_g:>15.4e}{marker}")

print("\n" + ("OK: all replicas match K=1 minimum within 1 kJ/mol"
              if all_ok else "FAIL: at least one replica did not converge to K=1 minimum"))
