#!/usr/bin/env python
"""
Confirm GBSA, bonded, and isolated-nonbonded Hessians compile (NVRTC) and produce
correct results in single vs double precision contexts (posq is float4 in single
and double4 in double).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import openmm as mm
from openmm import Platform, Context, VerletIntegrator, NonbondedForce
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, elementary_charge, kilojoules_per_mole
import gridforceplugin as gfp
from gridforceplugin import BondedHessian, GBSAGridForce
from benchmark_utils import get_system_paths, create_evaluation_system
import test_gbsa_grid_force as TG

PLAT = Platform.getPlatformByName('CUDA')


def lig_params(prmtop):
    s = prmtop.createSystem()
    nb = next(f for f in (s.getForce(i) for i in range(s.getNumForces())) if isinstance(f, NonbondedForce))
    return [(nb.getParticleParameters(i)[0].value_in_unit(elementary_charge),
             nb.getParticleParameters(i)[1].value_in_unit(nanometer),
             nb.getParticleParameters(i)[2].value_in_unit(kilojoules_per_mole))
            for i in range(nb.getNumParticles())]


def bonded_isolated_hess(precision):
    paths = get_system_paths("1g9v")
    lp = AmberPrmtopFile(paths['ligand_prmtop']); lc = AmberInpcrdFile(paths['ligand_inpcrd'])
    n = lp.topology.getNumAtoms()
    system, _, iso_nb = create_evaluation_system(lp, lig_params(lp), {}, method=3)
    ctx = Context(system, VerletIntegrator(0.001), PLAT, {'Precision': precision})
    ctx.setPositions(lc.positions); ctx.getState(getEnergy=True)
    bh = BondedHessian(); bh.initialize(system, ctx)
    Hb = np.array(bh.computeHessian(ctx)).reshape(3*n, 3*n)
    Hn = np.array(iso_nb.computeHessian(ctx)).reshape(3*n, 3*n)
    del ctx
    return Hb, Hn


def gbsa_hess(precision, method=3):
    d = TG.create_test_ligand_receptor()
    rec = np.asarray(d['rec_positions'], float); lig = np.asarray(d['lig_positions'], float)
    n_rec, n_lig = len(rec), len(lig); allp = np.vstack([rec, lig]); sp = 0.05
    lo = allp.min(0) - 0.4
    counts = tuple(int(np.ceil((allp.max(0) + 0.4 - lo)[k] / sp)) for k in range(3))
    system = mm.System()
    for _ in range(n_rec + n_lig):
        system.addParticle(12.0)
    f = GBSAGridForce(); f.setNumAtoms(n_rec + n_lig); f.setIncludeSurfaceArea(False)
    qs = np.concatenate([d['rec_charges'], d['lig_charges']]); rs = np.concatenate([d['rec_radii'], d['lig_radii']])
    ss = np.concatenate([d['rec_scales'], d['lig_scales']])
    for i in range(n_rec + n_lig):
        f.setAtomParameters(i, float(qs[i]), float(rs[i]), float(ss[i]))
    f.setParticles(list(range(n_rec, n_rec + n_lig)))
    f.setAutoGenerateGrid(True); f.setComputeGridDerivatives(True)
    f.setReceptorAtoms(list(range(n_rec)))
    f.setReceptorPositions([float(x) for x in rec.flatten()])
    f.setReceptorRadii([float(x) for x in d['rec_radii']])
    f.setReceptorScaleFactors([float(x) for x in d['rec_scales']])
    f.setGridOrigin(float(lo[0]), float(lo[1]), float(lo[2])); f.setGridCounts(*counts); f.setGridSpacing(sp)
    f.setInterpolationMethod(method); system.addForce(f)
    ctx = Context(system, VerletIntegrator(0.001), PLAT, {'Precision': precision})
    ctx.setPositions(allp * nanometer); ctx.getState(getEnergy=True)
    f.computeHessian(ctx); H = np.array(f.getHessianMatrix(ctx))
    del ctx
    return H


def cmp(name, single, other, prec):
    rel = np.abs(single - other).max() / max(1.0, np.abs(single).max())
    fin = np.isfinite(other).all()
    print(f"  {name:22s} {prec:7s}: finite={fin} rel_vs_single={rel:.2e}  "
          f"{'OK' if fin and rel < 1e-3 else 'CHECK'}")


def main():
    print("=== bonded + isolated-NB Hessian ===")
    Hb_s, Hn_s = bonded_isolated_hess('single')
    for prec in ('mixed', 'double'):
        Hb, Hn = bonded_isolated_hess(prec)
        cmp("BondedHessian", Hb_s, Hb, prec)
        cmp("IsolatedNB Hessian", Hn_s, Hn, prec)
    print("=== GBSA grid Hessian (method 3) ===")
    Hg_s = gbsa_hess('single')
    for prec in ('mixed', 'double'):
        cmp("GBSA Hessian", Hg_s, gbsa_hess(prec), prec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
