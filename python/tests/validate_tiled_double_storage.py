"""
Validate double grid-derivative storage through the TILED path (generation ->
on-disk tiled file -> streaming -> tiled kernel). Generates the same charge grid
as a tiled file in float and in double, evaluates the triquintic force at a probe
via tiled input, and compares against a non-tiled double-storage reference.

The tiled-double force must (a) match the non-tiled double reference closely and
(b) differ from the tiled-float force -- i.e. the f64 carried by the double file
survives the whole tiled pipeline, which float storage truncates away.
"""
import os, sys, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import openmm as mm
from openmm import Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, kilojoules_per_mole
import gridforceplugin as gfp
from benchmark_utils import get_system_paths, validate_system_paths

NX = NY = NZ = 16
SP = 0.10
TILE = 8

plat = mm.Platform.getPlatformByName('CUDA')


def make_gen(prm, pos_list, rec_atoms, origin, double, tiled_path=None):
    g = gfp.GridForce()
    g.setGridOrigin(*origin)
    g.addGridCounts(NX, NY, NZ)
    g.addGridSpacing(SP, SP, SP)
    g.setAutoGenerateGrid(True)
    g.setGridType('charge')
    g.setComputeDerivatives(True)
    g.setGridCap(1e30)
    g.setReceptorAtoms(rec_atoms)
    g.setReceptorPositionsFromLists(pos_list)
    g.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
    if double:
        g.setUseDoubleStorage(True)
    if tiled_path:
        g.setTiledOutputFile(tiled_path, TILE)
    return g


def generate(prm, crd, pos_list, rec_atoms, origin, double, tiled_path=None, save_path=None):
    g = make_gen(prm, pos_list, rec_atoms, origin, double, tiled_path)
    gsys = prm.createSystem(nonbondedMethod=NoCutoff)
    gsys.addForce(g)
    ctx = Context(gsys, VerletIntegrator(0.001), plat)
    ctx.setPositions(crd.positions)
    ctx.getState(getEnergy=True)  # trigger generation
    if save_path:
        g.saveToFile(save_path)
    del ctx


def eval_force(probe, origin=None, tiled_input=None, grid_file=None, double=False):
    gf = gfp.GridForce()
    if tiled_input:
        gf.setGridOrigin(*origin)
        gf.addGridCounts(NX, NY, NZ)
        gf.addGridSpacing(SP, SP, SP)
        gf.setTiledInputFile(tiled_input)
        gf.setTiledMode(True, TILE, 512)
    else:
        gf.loadFromFile(grid_file)
        if double:
            gf.setUseDoubleStorage(True)
    gf.setInterpolationMethod(3)
    s = mm.System(); s.addParticle(1.0)
    gf.addParticleGroup('charge', [0], [1.0])
    s.addForce(gf)
    # Single-precision context: storage precision (double file/GPU derivatives) is
    # independent of the context compute precision, which keeps this off the
    # separate tiled-double-context posq path.
    ctx = Context(s, VerletIntegrator(0.001), plat, {'Precision': 'single'})
    ctx.setPositions([mm.Vec3(*probe)] * 1 * nanometer)
    f = np.array(ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        kilojoules_per_mole / nanometer))[0]
    del ctx
    return f


def main():
    paths = get_system_paths("1g9v")
    if validate_system_paths({k: paths[k] for k in ('receptor_prmtop', 'receptor_inpcrd')}):
        print("Missing data"); return 1
    prm = AmberPrmtopFile(paths['receptor_prmtop'])
    crd = AmberInpcrdFile(paths['receptor_inpcrd'])
    pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                 p[2].value_in_unit(nanometer)) for p in crd.positions]
    rec_atoms = list(range(prm.topology.getNumAtoms()))
    centroid = np.mean(np.array(pos_list), axis=0)
    origin = tuple(centroid - np.array([NX, NY, NZ]) * SP / 2.0)
    probe = (origin[0] + (NX // 2 + 0.37) * SP,
             origin[1] + (NY // 2 + 0.61) * SP,
             origin[2] + (NZ // 2 + 0.52) * SP)

    with tempfile.TemporaryDirectory() as tmp:
        nontiled = os.path.join(tmp, "nd.grid")
        tiled_d = os.path.join(tmp, "td.tiled")
        tiled_f = os.path.join(tmp, "tf.tiled")
        generate(prm, crd, pos_list, rec_atoms, origin, double=True, save_path=nontiled)
        generate(prm, crd, pos_list, rec_atoms, origin, double=True, tiled_path=tiled_d)
        generate(prm, crd, pos_list, rec_atoms, origin, double=False, tiled_path=tiled_f)

        F_ref = eval_force(probe, grid_file=nontiled, double=True)
        F_td = eval_force(probe, origin=origin, tiled_input=tiled_d)
        F_tf = eval_force(probe, origin=origin, tiled_input=tiled_f)

    def rel(a, b):
        return np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b))

    print(f"  F non-tiled double (reference) = {F_ref}")
    print(f"  F tiled double                 = {F_td}")
    print(f"  F tiled float                  = {F_tf}")
    print(f"  tiled-double vs reference : rel = {rel(F_td, F_ref):.3e}  (expect small)")
    print(f"  tiled-float  vs reference : rel = {rel(F_tf, F_ref):.3e}")
    print(f"  tiled-double vs tiled-float: rel = {rel(F_td, F_tf):.3e}  (expect > 0: f64 carried)")

    ok = rel(F_td, F_ref) < 1e-4 and rel(F_td, F_tf) > 1e-7
    print(f"\n  => tiled double storage {'CARRIES f64 (PASS)' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
