#!/usr/bin/env python
"""
Branch coverage for the triquintic f64-assembly consolidation: exercise every
inv_power mode (NONE / RUNTIME / STORED) for BOTH force and Hessian, and emit a
numeric signature per branch. Run on two builds (consolidated vs prototype) and
diff the signatures to prove the refactor is behavior-preserving in every branch.

Usage: python branch_coverage_triquintic.py <tag>   # writes /tmp/branch_<tag>.json
"""
import os, sys, json, tempfile
import numpy as np
import openmm as mm
from openmm import Platform, Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, kilojoules_per_mole

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridforceplugin as gfp
from benchmark_utils import get_system_paths

SP, NPART = 0.025, 200  # 0.25 Angstrom = production resolution
PLAT = Platform.getPlatformByName('CUDA')
INVP = {'charge': 0.0, 'ljr': -6.0, 'lja': -2.0}


def gen_grid(rp, rc, gtype, origin, counts, stored, tmp):
    pl = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
           p[2].value_in_unit(nanometer)) for p in rc.positions]
    s = rp.createSystem(nonbondedMethod=NoCutoff)
    g = gfp.GridForce()
    g.setGridOrigin(*map(float, origin)); g.addGridCounts(int(counts[0]), int(counts[1]), int(counts[2]))
    g.addGridSpacing(SP, SP, SP); g.setAutoGenerateGrid(True); g.setGridType(gtype)
    g.setComputeDerivatives(True); g.setGridCap(1e30)  # triquintic is uncapped; inv_power softens
    g.setReceptorAtoms(list(range(rp.topology.getNumAtoms()))); g.setReceptorPositionsFromLists(pl)
    if stored and INVP[gtype] != 0.0:
        g.setInvPowerMode(gfp.InvPowerMode_STORED, INVP[gtype])
    else:
        g.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
    s.addForce(g)
    c = Context(s, VerletIntegrator(0.001), PLAT); c.setPositions(rc.positions); c.getState(getEnergy=True)
    path = os.path.join(tmp, f"{gtype}_{'stored' if stored else 'raw'}.grid"); g.saveToFile(path); del c
    return path


def sig(name, grid_file, eval_mode, inv_power, pos):
    n = len(pos)
    g = gfp.GridForce(); g.loadFromFile(grid_file); g.setInterpolationMethod(3)
    g.setInvPowerMode(eval_mode, inv_power)
    s = mm.System()
    for _ in range(n):
        s.addParticle(1.0)
    g.addParticleGroup('grp', list(range(n)), [1.0] * n)
    s.addForce(g)
    ctx = Context(s, VerletIntegrator(0.001), PLAT); ctx.setPositions(pos * nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    E = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    F = np.array(st.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer))
    g.computeHessian(ctx)
    H = np.array(g.getHessianMatrices(ctx))
    del ctx
    return {
        'energy': E, 'force_absmax': float(np.abs(F).max()), 'force_l1': float(np.abs(F).sum()),
        'hess_absmax': float(np.abs(H).max()), 'hess_l1': float(np.abs(H).sum()),
        'force_nan': bool(np.isnan(F).any()), 'hess_nan': bool(np.isnan(H).any()),
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'x'
    paths = get_system_paths("1g9v")
    rp = AmberPrmtopFile(paths['receptor_prmtop']); rc = AmberInpcrdFile(paths['receptor_inpcrd'])
    lc = AmberInpcrdFile(paths['ligand_inpcrd'])
    lig = np.array([[p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                     p[2].value_in_unit(nanometer)] for p in lc.positions])
    # Grid sized to the ligand bounding box + margin, at 0.25 A (production res).
    margin = 0.4
    origin = lig.min(0) - margin
    counts = np.ceil((lig.max(0) + margin - origin) / SP).astype(int)
    pos = lig  # evaluate at the real ligand pose (valid region)
    out = {}
    print(f"grid: counts={tuple(int(c) for c in counts)} spacing={SP} nm "
          f"(~{int(np.prod(counts))} pts), {len(pos)} eval particles")
    with tempfile.TemporaryDirectory() as tmp:
        ch_raw = gen_grid(rp, rc, 'charge', origin, counts, False, tmp)
        out['charge_NONE'] = sig('charge_NONE', ch_raw, gfp.InvPowerMode_NONE, 0.0, pos)

        ljr_raw = gen_grid(rp, rc, 'ljr', origin, counts, False, tmp)
        out['ljr_RUNTIME'] = sig('ljr_RUNTIME', ljr_raw, gfp.InvPowerMode_RUNTIME, -6.0, pos)

        ljr_st = gen_grid(rp, rc, 'ljr', origin, counts, True, tmp)
        out['ljr_STORED'] = sig('ljr_STORED', ljr_st, gfp.InvPowerMode_STORED, -6.0, pos)

    json.dump(out, open(f"/tmp/branch_{tag}.json", 'w'), indent=2)
    print(f"=== branch coverage [{tag}] ===")
    for k, v in out.items():
        flag = ' !!NaN!!' if (v['force_nan'] or v['hess_nan']) else ''
        print(f"  {k:14s} E={v['energy']:+.4e}  |F|max={v['force_absmax']:.3e}  "
              f"|H|max={v['hess_absmax']:.3e}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
