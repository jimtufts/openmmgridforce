"""Diagnose: K-group IsolatedGBSAForce receptor desolvation bug.

For a K-replica system where every particle group holds the SAME ligand
positions, every group's `getGroupReceptorDesolvation(g)` must equal the
K=1 baseline. Anything else = the reported bug.

Also runs the exact same K each time and checks whether values change
between successive execute() calls or context reconstructions
(nondeterminism check).
"""
import os
import sys

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile

import gridforceplugin as gf

PRMTOP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'prmtopcrd')


def _read_params(prmtop_path, inpcrd_path):
    prm = AmberPrmtopFile(prmtop_path)
    inpcrd = AmberInpcrdFile(inpcrd_path)
    system = prm.createSystem(implicitSolvent=None, constraints=None,
                              nonbondedMethod=mm.app.NoCutoff)
    nb = None; gb = None
    prm2 = AmberPrmtopFile(prmtop_path)
    system_gb = prm2.createSystem(implicitSolvent=mm.app.OBC2,
                                    constraints=None,
                                    nonbondedMethod=mm.app.NoCutoff)
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            nb = f
    for f in system_gb.getForces():
        if isinstance(f, mm.GBSAOBCForce):
            gb = f
    n = system.getNumParticles()
    positions = np.array(inpcrd.positions.value_in_unit(unit.nanometer))
    charges = np.array([nb.getParticleParameters(i)[0].value_in_unit(
        unit.elementary_charge) for i in range(n)])
    radii = np.array([gb.getParticleParameters(i)[1].value_in_unit(
        unit.nanometer) for i in range(n)])
    scales = np.array([gb.getParticleParameters(i)[2] for i in range(n)])
    return positions, charges, radii, scales


def build_k_group_system(k, lig_pos, lig_charges, lig_radii, lig_scales,
                          rec_pos, rec_charges, rec_radii, rec_scales,
                          jitter_seed=None):
    """K replicas of the SAME ligand pose (unless jitter_seed is set).

    Returns (system, positions_flat, force, per_group_pos).
    """
    n_lig = len(lig_pos)
    n_rec = len(rec_pos)

    system = mm.System()
    for _ in range(k * n_lig):
        system.addParticle(12.0)

    force = gf.IsolatedGBSAForce()
    force.setGBMethod(gf.IsolatedGBSAForce.OBC_II)
    force.setSoluteDielectric(1.0)
    force.setSolventDielectric(78.5)
    force.setIncludeSurfaceArea(False)

    force.setNumAtoms(n_lig)
    for i in range(n_lig):
        force.setAtomParameters(i, float(lig_charges[i]),
                                float(lig_radii[i]), float(lig_scales[i]))

    force.setReceptorMode(gf.IsolatedGBSAForce.PAIRWISE)
    force.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        force.setReceptorAtomParameters(i, float(rec_charges[i]),
                                        float(rec_radii[i]), float(rec_scales[i]))
    force.setReceptorPositions(rec_pos.flatten().tolist())

    # K identical particle groups
    per_group_pos = []
    for g in range(k):
        atoms = list(range(g * n_lig, (g + 1) * n_lig))
        force.addParticleGroup(f'lig_{g}', atoms)
        if jitter_seed is not None:
            rng = np.random.default_rng(jitter_seed + g)
            per_group_pos.append(lig_pos + rng.normal(0, 0.001, lig_pos.shape))
        else:
            per_group_pos.append(lig_pos)

    force.setDownloadBornRadii(True)
    system.addForce(force)
    positions_flat = np.vstack(per_group_pos)
    return system, positions_flat, force, per_group_pos


def run_and_report(k, jitter_seed=None, tag=''):
    lig_pos, lig_q, lig_r, lig_s = _read_params(
        os.path.join(PRMTOP_DIR, 'ligand.prmtop'),
        os.path.join(PRMTOP_DIR, 'ligand.trans.inpcrd'))
    rec_pos, rec_q, rec_r, rec_s = _read_params(
        os.path.join(PRMTOP_DIR, 'receptor.prmtop'),
        os.path.join(PRMTOP_DIR, 'receptor.trans.inpcrd'))
    system, pos, force, _ = build_k_group_system(
        k, lig_pos, lig_q, lig_r, lig_s,
        rec_pos, rec_q, rec_r, rec_s, jitter_seed=jitter_seed)
    plat = mm.Platform.getPlatformByName('CUDA')
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, plat)
    ctx.setPositions(pos * unit.nanometer)
    ctx.getState(getEnergy=True)
    dsv = [force.getGroupReceptorDesolvation(g) for g in range(k)]
    xte = [force.getGroupCrossTermEnergy(g) for g in range(k)]
    lse = [force.getGroupLigandSelfEnergy(g) for g in range(k)]
    born_per_g = [np.array(force.getReceptorBornRadii(g)) for g in range(k)]
    del ctx, integ
    label = f'K={k}{tag}'
    print(f'--- {label} ---')
    print(f'  receptor desolv: min={min(dsv):+.4f}  max={max(dsv):+.4f}  '
          f'spread={max(dsv)-min(dsv):.3e}  vals={["%+.3f"%v for v in dsv[:8]]}')
    print(f'  cross term    : min={min(xte):+.4f}  max={max(xte):+.4f}  '
          f'spread={max(xte)-min(xte):.3e}')
    print(f'  ligand self   : min={min(lse):+.4f}  max={max(lse):+.4f}  '
          f'spread={max(lse)-min(lse):.3e}')
    if k > 1:
        b0 = born_per_g[0]
        for g in range(k):
            bg = born_per_g[g]
            diff = np.abs(bg - b0)
            n_changed = int(np.sum(diff > 1e-6))
            print(f'  born_radii g={g}: mean={bg.mean():.4f} max_delta_vs_g0={diff.max():.3e} '
                  f'n_differ={n_changed}/{len(bg)}')
    return dsv, xte, lse


def main():
    print('=== K-group identical-replica test ===')
    base, _, _ = run_and_report(1, tag=' (baseline)')
    baseline_desolv = base[0]

    for k in (2, 3, 4, 5, 6, 7, 8):
        dsv, _, _ = run_and_report(k)
        errs = np.array(dsv) - baseline_desolv
        print(f'  vs K=1 baseline: max|err|={np.max(np.abs(errs)):.4e}  '
              f'first-vs-last={dsv[0]-dsv[-1]:+.4e}')

    print('\n=== Nondeterminism repro (K=4 identical, 5 runs) ===')
    all_runs = []
    for r in range(5):
        d, _, _ = run_and_report(4, tag=f' run{r}')
        all_runs.append(d)
    arr = np.array(all_runs)
    print(f'\n  per-group std across runs: {arr.std(axis=0)}')
    print(f'  cross-run drift per group : {arr.max(axis=0) - arr.min(axis=0)}')

    print('\n=== K=4 with tiny per-group jitter (0.001 nm) ===')
    run_and_report(4, jitter_seed=17, tag=' (jittered)')


if __name__ == '__main__':
    main()
