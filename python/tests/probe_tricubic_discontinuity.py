#!/usr/bin/env python
"""
Measure the plugin's cross-cell FORCE discontinuity for tricubic (method 2),
to confirm the fp32 64-coefficient solve has the same bug as triquintic, and to
validate the f64 fix. Straddle a real cell face and isolate the jump from the
smooth trend by linear extrapolation of the left-side slope.
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

SP, N = 0.025, 48  # production resolution


def main():
    paths = get_system_paths("1g9v")
    rp = AmberPrmtopFile(paths['receptor_prmtop']); rc = AmberInpcrdFile(paths['receptor_inpcrd'])
    lc = AmberInpcrdFile(paths['ligand_inpcrd'])
    pl = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
           p[2].value_in_unit(nanometer)) for p in rc.positions]
    center = np.array([[p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                        p[2].value_in_unit(nanometer)] for p in lc.positions]).mean(0)
    origin = center - np.array([N, N, N]) * SP / 2.0
    ox, oy, oz = map(float, origin)
    plat = Platform.getPlatformByName('CUDA')

    with tempfile.TemporaryDirectory() as tmp:
        gf = os.path.join(tmp, "charge.grid")
        gsys = rp.createSystem(nonbondedMethod=NoCutoff)
        gen = gfp.GridForce()
        gen.setGridOrigin(ox, oy, oz); gen.addGridCounts(N, N, N); gen.addGridSpacing(SP, SP, SP)
        gen.setAutoGenerateGrid(True); gen.setGridType('charge'); gen.setComputeDerivatives(True)
        gen.setGridCap(1e30); gen.setReceptorAtoms(list(range(rp.topology.getNumAtoms())))
        gen.setReceptorPositionsFromLists(pl); gen.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gsys.addForce(gen)
        gc = Context(gsys, VerletIntegrator(0.001), plat); gc.setPositions(rc.positions)
        gc.getState(getEnergy=True); gen.saveToFile(gf); del gc

        g = gfp.GridForce(); g.loadFromFile(gf); g.setInterpolationMethod(2)  # TRICUBIC
        g.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        s = mm.System(); s.addParticle(1.0); g.addParticleGroup('g', [0], [1.0]); s.addForce(g)
        ctx = Context(s, VerletIntegrator(0.001), plat)

        bx, by, bz = N // 2, N // 2, N // 2
        xb = ox + (bx + 1) * SP; yb = oy + (by + 0.6) * SP; zb = oz + (bz + 0.5) * SP
        dlt = 1e-5

        def fx(x):
            ctx.setPositions([mm.Vec3(x, yb, zb)] * nanometer)
            return np.array(ctx.getState(getForces=True).getForces(asNumpy=True)
                            .value_in_unit(kilojoules_per_mole / nanometer))[0]

        F1, F2, F3 = fx(xb - 2*dlt), fx(xb - dlt), fx(xb + dlt)
        slope = (F2 - F1) / dlt
        disc = F3 - (F2 + slope * 2 * dlt)
        print(f"=== TRICUBIC (method 2) cross-cell force discontinuity ===")
        print(f"  |disc|max = {np.abs(disc).max():.3e} kJ/mol/nm   "
              f"(smooth floor ~{np.abs(slope).max()*2*dlt:.1e}, L-BFGS needs ~1e-6)")
        del ctx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
