#!/usr/bin/env python
"""Smoke test: compute a GBSAGridForce Hessian (exercises the f64-routed
interpolateGBSAGridsWithHessian path) and confirm it is finite. Reuses the
setup helpers from test_gbsa_grid_force.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import openmm as mm
from openmm import unit
import test_gbsa_grid_force as T
from desolvation_grid_generator import generate_desolvation_grid
from gridforceplugin import GBSAGridForce


def build(method):
    data = T.create_test_ligand_receptor()
    n_rec = len(data['rec_positions']); n_lig = len(data['lig_positions'])
    py_grid = generate_desolvation_grid(
        data['rec_positions'], data['rec_radii'], data['rec_scales'],
        np.array([-0.2, -0.2, -0.2]), 0.05, (20, 20, 20),
        probe_radius=0.14, r_thresholds=(0.12, 0.16), verbose=False)
    cpp_grid = T.convert_grid_to_cpp(py_grid)

    system = mm.System()
    for _ in range(n_rec + n_lig):
        system.addParticle(12.0)
    f = GBSAGridForce()
    f.setNumAtoms(n_rec + n_lig); f.setIncludeSurfaceArea(False)
    qs = np.concatenate([data['rec_charges'], data['lig_charges']])
    rs = np.concatenate([data['rec_radii'], data['lig_radii']])
    ss = np.concatenate([data['rec_scales'], data['lig_scales']])
    for i in range(n_rec + n_lig):
        f.setAtomParameters(i, float(qs[i]), float(rs[i]), float(ss[i]))
    f.setDesolvationGrid(cpp_grid)
    f.setParticles(list(range(n_rec, n_rec + n_lig)))
    if hasattr(f, 'setInterpolationMethod'):
        f.setInterpolationMethod(method)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'))
    ctx.setPositions(np.vstack([data['rec_positions'], data['lig_positions']]) * unit.nanometer)
    ctx.getState(getEnergy=True)
    f.computeHessian(ctx)
    H = np.array(f.getHessianMatrix(ctx))
    return H


for method, name in [(2, 'tricubic'), (3, 'triquintic')]:
    try:
        H = build(method)
        print(f"  GBSA Hessian [{name} method={method}]: shape={H.shape}  "
              f"finite={np.isfinite(H).all()}  |H|max={np.abs(H).max():.3e}  "
              f"min_eig={np.linalg.eigvalsh((H+H.T)/2).min():.3e}")
    except Exception as e:
        print(f"  [{name}] {type(e).__name__}: {e}")
