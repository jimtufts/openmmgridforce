"""
Validate that double grid-storage also generates the analytical derivatives in
double. Generates a charge grid in float and in double, then compares the energy
channel (derivative index 0 = sum of k*q_i/r_i over receptor atoms) against an
independent f64 numpy reference. Double generation should track the reference
far more closely than float generation.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import openmm as mm
from openmm import Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer
import gridforceplugin as gfp
from benchmark_utils import get_system_paths, validate_system_paths

COULOMB_CONST = 138.935456  # kJ*nm/(mol*e^2)
R2_MIN = 0.0004             # (0.02 nm)^2, matches the generation clamp


def generate(prm, crd, pos_list, rec_atoms, origin, counts, sp, double):
    ox, oy, oz = origin
    nx, ny, nz = counts
    gen = gfp.GridForce()
    gen.setGridOrigin(ox, oy, oz)
    gen.addGridCounts(nx, ny, nz)
    gen.addGridSpacing(sp, sp, sp)
    gen.setAutoGenerateGrid(True)
    gen.setGridType('charge')
    gen.setComputeDerivatives(True)
    if double:
        gen.setUseDoubleStorage(True)
    gen.setGridCap(1e30)
    gen.setReceptorAtoms(rec_atoms)
    gen.setReceptorPositionsFromLists(pos_list)
    gen.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
    gsys = prm.createSystem(nonbondedMethod=NoCutoff)
    gsys.addForce(gen)
    plat = mm.Platform.getPlatformByName('CUDA')
    ctx = Context(gsys, VerletIntegrator(0.001), plat)
    ctx.setPositions(crd.positions)
    ctx.getState(getEnergy=True)  # trigger generation
    d = np.array(gen.getDerivatives(), dtype=np.float64)
    del ctx
    return d


def f64_energy_reference(charges, pos_list, origin, counts, sp):
    # The kernel reads float32 receptor positions and a float32 origin/spacing, so
    # round inputs to float32 here: that isolates the accumulation precision (what
    # double generation improves) from the float32 input-position error (which it
    # cannot fix). Arithmetic below stays f64 = the exact accumulation reference.
    ox, oy, oz = (np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]))
    sp = np.float32(sp)
    nx, ny, nz = counts
    atoms = np.array(pos_list, dtype=np.float32).astype(np.float64)
    q = np.array(charges, dtype=np.float64)
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    gx = (ox + ii.ravel().astype(np.float32) * sp).astype(np.float64)
    gy = (oy + jj.ravel().astype(np.float32) * sp).astype(np.float64)
    gz = (oz + kk.ravel().astype(np.float32) * sp).astype(np.float64)
    grid = np.stack([gx, gy, gz], axis=1)            # (N,3)
    e = np.empty(grid.shape[0], dtype=np.float64)
    for p in range(grid.shape[0]):
        dr = grid[p] - atoms                          # (A,3)
        r2 = np.sum(dr * dr, axis=1)
        r2 = np.maximum(r2, R2_MIN)
        e[p] = np.sum(COULOMB_CONST * q / np.sqrt(r2))
    return e


def main():
    paths = get_system_paths("1g9v")
    missing = validate_system_paths({k: paths[k] for k in
                                     ('receptor_prmtop', 'receptor_inpcrd')})
    if missing:
        print("Missing data:", missing)
        return 1

    prm = AmberPrmtopFile(paths['receptor_prmtop'])
    crd = AmberInpcrdFile(paths['receptor_inpcrd'])
    pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                 p[2].value_in_unit(nanometer)) for p in crd.positions]
    rec_atoms = list(range(prm.topology.getNumAtoms()))

    charge_sys = prm.createSystem(nonbondedMethod=NoCutoff)
    nb = [f for f in charge_sys.getForces() if isinstance(f, mm.NonbondedForce)][0]
    charges = [nb.getParticleParameters(i)[0].value_in_unit_system(mm.unit.md_unit_system)
               for i in rec_atoms]

    centroid = np.mean(np.array(pos_list), axis=0)
    nx = ny = nz = 16
    sp = 0.10
    origin = tuple(centroid - np.array([nx, ny, nz]) * sp / 2.0)
    counts = (nx, ny, nz)
    N = nx * ny * nz

    print(f"charge grid {nx}x{ny}x{nz}, sp={sp} nm, {len(rec_atoms)} receptor atoms")
    d_float = generate(prm, crd, pos_list, rec_atoms, origin, counts, sp, double=False)
    d_double = generate(prm, crd, pos_list, rec_atoms, origin, counts, sp, double=True)
    e_ref = f64_energy_reference(charges, pos_list, origin, counts, sp)

    e_float = d_float[:N]   # deriv-major: index-0 channel is the first N entries
    e_double = d_double[:N]
    scale = np.maximum(1.0, np.abs(e_ref))
    rel_float = np.abs(e_float - e_ref) / scale
    rel_double = np.abs(e_double - e_ref) / scale

    print("\nenergy channel vs f64 numpy reference (relative error):")
    print(f"  float  generation: max={rel_float.max():.3e}  mean={rel_float.mean():.3e}")
    print(f"  double generation: max={rel_double.max():.3e}  mean={rel_double.mean():.3e}")
    print(f"  mean-error improvement: {rel_float.mean() / rel_double.mean():.1f}x")

    # Oracle-free check: float generation stores fp32 values, so every output is
    # exactly fp32-representable; double generation carries sub-fp32 bits.
    frac_f32_float = np.mean(d_float.astype(np.float32).astype(np.float64) == d_float)
    frac_f32_double = np.mean(d_double.astype(np.float32).astype(np.float64) == d_double)
    print("\nfraction of generated derivatives that are exactly fp32-representable:")
    print(f"  float  generation: {frac_f32_float:.4f}  (expect ~1.0)")
    print(f"  double generation: {frac_f32_double:.4f}  (expect <<1.0: genuine f64)")

    ok = (rel_float.mean() / rel_double.mean() > 3.0) and frac_f32_float > 0.99 and frac_f32_double < 0.1
    print(f"\n  => double generation {'PRODUCES GENUINE f64' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
