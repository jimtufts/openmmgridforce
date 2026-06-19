#!/usr/bin/env python
"""
Tier 1 validation: does the grid force compute correctly when the OpenMM context
uses 'double' precision? The kernels read posq as float4, but cu.getPosq() is a
double4 buffer in double precision -- so a double context reads garbage positions.
This compares grid energy/force across single / mixed / double precision contexts;
they should agree to ~fp32 (single/mixed) and be CLOSE (double). Large disagreement
in double => the latent posq element-size bug.
"""
import os, sys, tempfile
import numpy as np
import openmm as mm
from openmm import Platform, Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, kilojoules_per_mole

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridforceplugin as gfp
from benchmark_utils import get_system_paths


def eval_precision(grid_file, particle_pos, precision):
    plat = Platform.getPlatformByName('CUDA')
    s = mm.System(); s.addParticle(1.0)
    g = gfp.GridForce(); g.loadFromFile(grid_file); g.setInterpolationMethod(3)
    g.addParticleGroup('g', [0], [1.0]); s.addForce(g)
    ctx = Context(s, VerletIntegrator(0.001), plat, {'Precision': precision})
    ctx.setPositions([mm.Vec3(*particle_pos)] * nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    E = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    F = np.array(st.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer))[0]
    del ctx
    return E, F


def main():
    paths = get_system_paths("1g9v")
    rp = AmberPrmtopFile(paths['receptor_prmtop']); rc = AmberInpcrdFile(paths['receptor_inpcrd'])
    lc = AmberInpcrdFile(paths['ligand_inpcrd'])
    pl = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
           p[2].value_in_unit(nanometer)) for p in rc.positions]
    cen = np.array([[p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                     p[2].value_in_unit(nanometer)] for p in lc.positions]).mean(0)
    sp, n = 0.05, 32
    origin = cen - np.array([n, n, n]) * sp / 2.0
    plat = Platform.getPlatformByName('CUDA')
    with tempfile.TemporaryDirectory() as tmp:
        gf = os.path.join(tmp, "charge.grid")
        gsys = rp.createSystem(nonbondedMethod=NoCutoff)
        gen = gfp.GridForce()
        gen.setGridOrigin(*map(float, origin)); gen.addGridCounts(n, n, n); gen.addGridSpacing(sp, sp, sp)
        gen.setAutoGenerateGrid(True); gen.setGridType('charge'); gen.setComputeDerivatives(True)
        gen.setGridCap(1e30); gen.setReceptorAtoms(list(range(rp.topology.getNumAtoms())))
        gen.setReceptorPositionsFromLists(pl); gen.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gsys.addForce(gen)
        gc = Context(gsys, VerletIntegrator(0.001), plat); gc.setPositions(rc.positions)
        gc.getState(getEnergy=True); gen.saveToFile(gf); del gc

        probe = [float(cen[0]), float(cen[1]), float(cen[2])]
        ref_E, ref_F = eval_precision(gf, probe, 'single')
        print(f"  single : E={ref_E:.6e}  F={ref_F}")
        for prec in ('mixed', 'double'):
            E, F = eval_precision(gf, probe, prec)
            dE = abs(E - ref_E) / max(1.0, abs(ref_E))
            dF = np.linalg.norm(F - ref_F) / max(1.0, np.linalg.norm(ref_F))
            ok = dE < 1e-3 and dF < 1e-3
            print(f"  {prec:7s}: E={E:.6e}  F={F}  relE={dE:.2e} relF={dF:.2e}  "
                  f"{'OK' if ok else 'BROKEN vs single'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
