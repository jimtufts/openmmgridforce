"""Validate the plugin's assembled Hessian against a JAX autodiff reference.

For the ligand-only system (bonded + intra-lig NB + intra-lig GBSA
PAIRWISE), compute:
  - Plugin analytical H: BondedHessian + IsolatedNonbondedForce +
    IsolatedGBSAForce PAIRWISE
  - JAX autodiff H: jax.hessian(gas_total_energy)

Compare entry-by-entry. If they match, the negative diagonals we see at
the LE min are physical; if they differ, we have a plugin Hessian bug.
"""
import os, sys, gc
import numpy as np
import openmm
import openmm.unit as unit
import gridforceplugin as gfp

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/jtufts/src/p312/algdock')
from AlGDock.mwe.utils import (
    extract_bonded_params, extract_nonbonded_params, extract_obc_params,
    build_isolated_bonded_force, build_isolated_nonbonded_force,
    build_isolated_gbsa_force, load_positions,
)
from AlGDock.mwe.jax_gas_reference import gas_total_energy, build_params_from_prmtop

PRMDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prmtopcrd')
LIGAND_PRMTOP = os.path.join(PRMDIR, 'ligand.prmtop')
LIGAND_INPCRD = os.path.join(PRMDIR, 'ligand.trans.inpcrd')

bonded_params  = extract_bonded_params(LIGAND_PRMTOP)
nb_params      = extract_nonbonded_params(LIGAND_PRMTOP)
lig_obc_params = extract_obc_params(LIGAND_PRMTOP)
N = bonded_params['n_atoms']
n3 = 3 * N
masses = np.array(bonded_params['masses'])
pos_lig = load_positions(LIGAND_INPCRD)


def build_system():
    """Returns (system, iso_bnd, iso_nb, iso_gbsa) — keep force handles for
    downstream computeHessian since SWIG downcast on getForce doesn't
    work."""
    K = 1
    system = openmm.System()
    for i in range(N): system.addParticle(masses[i])
    iso_bnd  = build_isolated_bonded_force(bonded_params, K, N)
    iso_nb   = build_isolated_nonbonded_force(nb_params, K, N)
    iso_gbsa = build_isolated_gbsa_force(lig_obc_params, K, N)
    system.addForce(iso_bnd)
    system.addForce(iso_nb)
    system.addForce(iso_gbsa)
    return system, iso_bnd, iso_nb, iso_gbsa


def make_ctx():
    system, iso_bnd, iso_nb, iso_gbsa = build_system()
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName('CUDA'),
                         {'Precision': 'double'})
    return ctx, iso_bnd, iso_nb, iso_gbsa


# LE min from raw inpcrd
ctx, iso_bnd, iso_nb, iso_gbsa = make_ctx()
ctx.setPositions(pos_lig * unit.nanometer)
openmm.LocalEnergyMinimizer.minimize(ctx, 1e-6, 5000)
st = ctx.getState(getPositions=True, getEnergy=True, getForces=True)
x_min = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
E_min = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
F_min = np.array(st.getForces(asNumpy=True).value_in_unit(
    unit.kilojoule_per_mole / unit.nanometer))
RMS_min = float(np.sqrt(np.mean(np.sum(F_min**2, axis=1))))
print(f"LE min: E={E_min:+.4f}  RMS={RMS_min:.2e}\n")

# === Plugin analytical Hessian ===
system = ctx.getSystem()
bh = gfp.BondedHessian(); bh.initialize(system, ctx)
H_bh_stock = np.array(bh.computeHessian(ctx)).reshape(n3, n3)  # legacy path
H_bnd_iso  = np.array(iso_bnd.computeHessian(ctx, 0)).reshape(n3, n3)
H_nb       = np.array(iso_nb.computeHessian(ctx)).reshape(n3, n3)
H_gbsa     = np.array(iso_gbsa.computeHessian(ctx)).reshape(n3, n3)
H_plugin = H_bnd_iso + H_nb + H_gbsa

print("Plugin Hessian diagonal:")
print(f"  BondedHessian (stock)    : min={H_bh_stock.diagonal().min():+.3e}  max={H_bh_stock.diagonal().max():+.3e}  n_neg={int((H_bh_stock.diagonal()<0).sum())}/{n3}")
print(f"  IsolatedBondedForce      : min={H_bnd_iso.diagonal().min():+.3e}  max={H_bnd_iso.diagonal().max():+.3e}  n_neg={int((H_bnd_iso.diagonal()<0).sum())}/{n3}")
print(f"  IsolatedNonbondedForce   : min={H_nb.diagonal().min():+.3e}  max={H_nb.diagonal().max():+.3e}  n_neg={int((H_nb.diagonal()<0).sum())}/{n3}")
print(f"  IsolatedGBSAForce (PAIR) : min={H_gbsa.diagonal().min():+.3e}  max={H_gbsa.diagonal().max():+.3e}  n_neg={int((H_gbsa.diagonal()<0).sum())}/{n3}")
print(f"  plugin total             : min={H_plugin.diagonal().min():+.3e}  max={H_plugin.diagonal().max():+.3e}  n_neg={int((H_plugin.diagonal()<0).sum())}/{n3}")

# Symmetry check
asym = np.max(np.abs(H_plugin - H_plugin.T))
print(f"\nSymmetry ||H - H^T||_inf = {asym:.3e}")

# === JAX autodiff Hessian ===
print("\nBuilding JAX params...")
params = build_params_from_prmtop(LIGAND_PRMTOP, N)
pos_flat = jnp.asarray(x_min.reshape(-1), dtype=jnp.float64)

E_jax = float(gas_total_energy(pos_flat, params))
print(f"JAX energy: {E_jax:+.4f}   plugin energy: {E_min:+.4f}   diff = {E_jax-E_min:+.3e}")

print("Computing JAX Hessian (autodiff)...")
H_jax = np.array(jax.hessian(gas_total_energy)(pos_flat, params))

print(f"\nJAX H diagonal: min={H_jax.diagonal().min():+.3e}  max={H_jax.diagonal().max():+.3e}  n_neg={int((H_jax.diagonal()<0).sum())}/{n3}")

# === Comparison ===
diff = H_plugin - H_jax
absdiff = np.abs(diff)
scale = np.maximum(np.abs(H_jax), 1.0)
rel = absdiff / scale

print("\n=== Plugin vs JAX Hessian ===")
print(f"  ||H_plugin||_F = {np.linalg.norm(H_plugin):.4e}")
print(f"  ||H_jax||_F    = {np.linalg.norm(H_jax):.4e}")
print(f"  ||diff||_F     = {np.linalg.norm(diff):.4e}")
print(f"  max abs diff   = {absdiff.max():.4e}")
print(f"  max rel diff   = {rel.max():.4e}")

# Find worst-offending entries
worst_flat = np.argsort(absdiff.ravel())[-10:][::-1]
print("\nTop 10 largest absolute-error entries:")
for k in worst_flat:
    i, j = np.unravel_index(k, absdiff.shape)
    print(f"  H[{i:3d},{j:3d}]: plugin={H_plugin[i,j]:+.4e}  jax={H_jax[i,j]:+.4e}  diff={diff[i,j]:+.4e}")

# Diagonal comparison
diag_diff = H_plugin.diagonal() - H_jax.diagonal()
print(f"\nDiagonal diff: max abs = {np.abs(diag_diff).max():.4e}")
neg_plugin = np.where(H_plugin.diagonal() < 0)[0]
neg_jax    = np.where(H_jax.diagonal() < 0)[0]
print(f"Plugin diagonals < 0 at indices: {neg_plugin.tolist()}")
print(f"JAX    diagonals < 0 at indices: {neg_jax.tolist()}")

# Full eigenvalue comparison
eig_plugin = np.linalg.eigvalsh(0.5*(H_plugin + H_plugin.T))
eig_jax    = np.linalg.eigvalsh(0.5*(H_jax + H_jax.T))
print(f"\nEigenvalues (symmetrized):")
print(f"  plugin:  min={eig_plugin.min():+.3e}  max={eig_plugin.max():+.3e}  n_neg={int((eig_plugin<0).sum())}/{n3}")
print(f"  jax   :  min={eig_jax.min():+.3e}  max={eig_jax.max():+.3e}  n_neg={int((eig_jax<0).sum())}/{n3}")
