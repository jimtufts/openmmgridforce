#!/usr/bin/env python
"""GBSA Hessian in GRID mode with auto-generated derivative grids (method 2/3).
This is the path that should route through interpolateGBSAGridsWithHessian's
polynomial solves. Reuses test_gbsa_grid_force.py's tiny rec/lig system.
Writes /tmp/gbsa_gm_H_<tag>_<method>.npy for f32-vs-f64 A/B."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import openmm as mm
from openmm import unit
import test_gbsa_grid_force as T
from gridforceplugin import GBSAGridForce

tag = sys.argv[1] if len(sys.argv) > 1 else 'x'
d = T.create_test_ligand_receptor()
rec = np.asarray(d['rec_positions'], float); lig = np.asarray(d['lig_positions'], float)
n_rec, n_lig = len(rec), len(lig)
allp = np.vstack([rec, lig])
sp = 0.05
lo = allp.min(0) - 0.4
counts = tuple(int(np.ceil((allp.max(0) + 0.4 - lo)[k] / sp)) for k in range(3))

for method, name in [(2, 'tricubic'), (3, 'triquintic')]:
    try:
        system = mm.System()
        for _ in range(n_rec + n_lig):
            system.addParticle(12.0)
        f = GBSAGridForce()
        f.setNumAtoms(n_rec + n_lig); f.setIncludeSurfaceArea(False)
        qs = np.concatenate([d['rec_charges'], d['lig_charges']])
        rs = np.concatenate([d['rec_radii'], d['lig_radii']])
        ss = np.concatenate([d['rec_scales'], d['lig_scales']])
        for i in range(n_rec + n_lig):
            f.setAtomParameters(i, float(qs[i]), float(rs[i]), float(ss[i]))
        f.setParticles(list(range(n_rec, n_rec + n_lig)))
        # GRID mode with derivatives
        f.setAutoGenerateGrid(True)
        f.setComputeGridDerivatives(True)
        f.setReceptorAtoms(list(range(n_rec)))
        f.setReceptorPositions([float(x) for x in rec.flatten()])
        f.setReceptorRadii([float(x) for x in d['rec_radii']])
        f.setReceptorScaleFactors([float(x) for x in d['rec_scales']])
        f.setGridOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
        f.setGridCounts(*counts)
        f.setGridSpacing(sp)
        f.setInterpolationMethod(method)
        system.addForce(f)
        ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName('CUDA'))
        ctx.setPositions(allp * unit.nanometer)
        ctx.getState(getEnergy=True)
        f.computeHessian(ctx)
        H = np.array(f.getHessianMatrix(ctx))
        np.save(f"/tmp/gbsa_gm_H_{tag}_{method}.npy", H)
        print(f"  GRID-mode GBSA Hessian [{name}={method}]: shape={H.shape} "
              f"finite={np.isfinite(H).all()} sum={H.sum():.10e} |H|max={np.abs(H).max():.10e}")
    except Exception as e:
        print(f"  [{name}={method}] {type(e).__name__}: {e}")
