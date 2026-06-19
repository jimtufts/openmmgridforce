#!/usr/bin/env python
"""
Full-system fake-PD reproduction: minimize a real ligand on a triquintic-Hermite
grid, then eigendecompose the FULL Cartesian Hessian (bonded + isolated nonbonded
+ grid) and count spurious NEGATIVE ("imaginary") modes.

At a true minimum in an external grid field the full Hessian must be positive
definite (the grid pins all 3N DOF). Spurious negative eigenvalues = "fake" non-PD,
the production symptom. Run on the f32 build (expect fake non-PD) vs the f64-
assembly build (expect PD) to validate the fix.
"""
import os, sys, tempfile
import numpy as np
import openmm as mm
from openmm import Platform, Context, VerletIntegrator, LocalEnergyMinimizer, NonbondedForce
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, elementary_charge, kilojoules_per_mole

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridforceplugin as gfp
from gridforceplugin import BondedHessian
from benchmark_utils import get_system_paths, create_evaluation_system

SP = 0.10
NGRID = 30


def lig_params(prmtop):
    s = prmtop.createSystem()
    nb = next(f for f in (s.getForce(i) for i in range(s.getNumForces())) if isinstance(f, NonbondedForce))
    out = []
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        out.append((q.value_in_unit(elementary_charge), sig.value_in_unit(nanometer),
                    eps.value_in_unit(kilojoules_per_mole)))
    return out


def gen_grid(rp, rc, gtype, center, tmp):
    pl = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
           p[2].value_in_unit(nanometer)) for p in rc.positions]
    origin = center - np.array([NGRID, NGRID, NGRID]) * SP / 2.0
    sysm = rp.createSystem(nonbondedMethod=NoCutoff)
    g = gfp.GridForce()
    g.setGridOrigin(*map(float, origin)); g.addGridCounts(NGRID, NGRID, NGRID)
    g.addGridSpacing(SP, SP, SP); g.setAutoGenerateGrid(True); g.setGridType(gtype)
    g.setComputeDerivatives(True); g.setGridCap(1e30)
    g.setReceptorAtoms(list(range(rp.topology.getNumAtoms()))); g.setReceptorPositionsFromLists(pl)
    g.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
    sysm.addForce(g)
    c = Context(sysm, VerletIntegrator(0.001), Platform.getPlatformByName('CUDA'))
    c.setPositions(rc.positions); c.getState(getEnergy=True)
    path = os.path.join(tmp, f"{gtype}.grid"); g.saveToFile(path); del c
    return path


def full_hessian(system, ctx, grid_forces, isolated_nb, n):
    n3 = 3 * n
    H = np.zeros((n3, n3))
    bh = BondedHessian(); bh.initialize(system, ctx)
    H += np.array(bh.computeHessian(ctx)).reshape(n3, n3)
    if isolated_nb is not None:
        H += np.array(isolated_nb.computeHessian(ctx)).reshape(n3, n3)
    for gtype, gforce in grid_forces.items():
        gforce.computeHessian(ctx)
        blocks = np.array(gforce.getHessianMatrices(ctx))  # (n,3,3) diagonal
        for i in range(n):
            H[3*i:3*i+3, 3*i:3*i+3] += blocks[i]
    return 0.5 * (H + H.T)


def main():
    paths = get_system_paths("1g9v")
    lp = AmberPrmtopFile(paths['ligand_prmtop'])
    lc = AmberInpcrdFile(paths['ligand_inpcrd'])
    rp = AmberPrmtopFile(paths['receptor_prmtop'])
    rc = AmberInpcrdFile(paths['receptor_inpcrd'])
    lig_pos = np.array([[p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                         p[2].value_in_unit(nanometer)] for p in lc.positions])
    center = lig_pos.mean(0)
    n = lp.topology.getNumAtoms()
    lparams = lig_params(lp)
    plat = Platform.getPlatformByName('CUDA')

    with tempfile.TemporaryDirectory() as tmp:
        grid_files = {gt: gen_grid(rp, rc, gt, center, tmp) for gt in ('charge', 'ljr', 'lja')}
        system, grid_forces, isolated_nb = create_evaluation_system(lp, lparams, grid_files, method=3)
        ctx = Context(system, VerletIntegrator(0.001), plat)
        ctx.setPositions(lc.positions)
        e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        LocalEnergyMinimizer.minimize(ctx, 1.0, 5000)
        st = ctx.getState(getEnergy=True, getForces=True)
        e1 = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        fmax = np.abs(np.array(st.getForces(asNumpy=True).value_in_unit(
            kilojoules_per_mole / nanometer))).max()

        H = full_hessian(system, ctx, grid_forces, isolated_nb, n)
        ev = np.linalg.eigvalsh(H)
        tol = 1.0  # kJ/mol/nm^2; modes below -tol are spurious imaginary
        n_neg = int((ev < -tol).sum())
        n_zeroish = int((np.abs(ev) <= tol).sum())

        print(f"=== FULL-SYSTEM fake-PD repro (ligand 1g9v, {n} atoms, triquintic grids) ===")
        print(f"  energy {e0:.1f} -> {e1:.1f} kJ/mol   max|force| after min = {fmax:.3e}")
        print(f"  Hessian eigenvalues: min={ev.min():.3e}  max={ev.max():.3e}")
        print(f"  negative modes (< -{tol}) : {n_neg}   near-zero (|.|<={tol}): {n_zeroish}")
        print(f"  most-negative few: {np.sort(ev)[:6]}")
        print(f"  => {'POSITIVE DEFINITE (no fake modes)' if n_neg == 0 else f'NON-PD: {n_neg} imaginary modes (fake-PD present)'}")
        del ctx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
