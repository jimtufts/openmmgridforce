"""
Precision test suite for the GridForce CUDA kernels.

Covers the compute-precision (Tier 2) and grid-storage (Tier 3) axes across
interpolation methods and precision modes. The invariant: single and mixed are
bit-identical, and double agrees with single to a loose tolerance (double computes
in higher precision but the same algorithm). Double storage must carry sub-fp32
information that float storage drops.

Runs under pytest, or directly as a script (prints a summary).
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pytest
import openmm as mm
from openmm import Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, kilojoules_per_mole
import gridforceplugin as gfp
from benchmark_utils import get_system_paths, validate_system_paths


def cuda_available():
    try:
        mm.Platform.getPlatformByName('CUDA'); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA not available")

NX = NY = NZ = 16
SP = 0.10
METHODS = {0: "trilinear", 1: "bspline", 2: "tricubic", 3: "triquintic"}


def _system():
    paths = get_system_paths("1g9v")
    if validate_system_paths({k: paths[k] for k in ('receptor_prmtop', 'receptor_inpcrd')}):
        pytest.skip("missing 1g9v data")
    prm = AmberPrmtopFile(paths['receptor_prmtop'])
    crd = AmberInpcrdFile(paths['receptor_inpcrd'])
    pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                 p[2].value_in_unit(nanometer)) for p in crd.positions]
    rec_atoms = list(range(prm.topology.getNumAtoms()))
    centroid = np.mean(np.array(pos_list), axis=0)
    origin = tuple(centroid - np.array([NX, NY, NZ]) * SP / 2.0)
    return prm, crd, pos_list, rec_atoms, origin


def _gen_grid(prm, crd, pos_list, rec_atoms, origin, grid_file, gtype='charge', double=False):
    g = gfp.GridForce()
    g.setGridOrigin(*origin); g.addGridCounts(NX, NY, NZ); g.addGridSpacing(SP, SP, SP)
    g.setAutoGenerateGrid(True); g.setGridType(gtype); g.setComputeDerivatives(True)
    g.setGridCap(1e30); g.setReceptorAtoms(rec_atoms)
    g.setReceptorPositionsFromLists(pos_list); g.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
    if double:
        g.setUseDoubleStorage(True)
    gsys = prm.createSystem(nonbondedMethod=NoCutoff); gsys.addForce(g)
    ctx = Context(gsys, VerletIntegrator(0.001), mm.Platform.getPlatformByName('CUDA'))
    ctx.setPositions(crd.positions); ctx.getState(getEnergy=True)
    g.saveToFile(grid_file); del ctx


def _eval(grid_file, method, probe, precision, double_storage=False):
    gf = gfp.GridForce(); gf.loadFromFile(grid_file); gf.setInterpolationMethod(method)
    if double_storage:
        gf.setUseDoubleStorage(True)
    s = mm.System(); s.addParticle(1.0); gf.addParticleGroup('charge', [0], [1.0]); s.addForce(gf)
    ctx = Context(s, VerletIntegrator(0.001), mm.Platform.getPlatformByName('CUDA'),
                  {'Precision': precision})
    ctx.setPositions([mm.Vec3(*probe)] * 1 * nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    f = np.array(st.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer))[0]
    del ctx
    return e, f


import tempfile, functools


@functools.lru_cache(maxsize=2)
def _grid(double):
    prm, crd, pos_list, rec_atoms, origin = _system()
    tmp = tempfile.mkdtemp()
    gf = os.path.join(tmp, f"charge_{'d' if double else 'f'}.grid")
    _gen_grid(prm, crd, pos_list, rec_atoms, origin, gf, double=double)
    probe = (origin[0] + (NX // 2 + 0.37) * SP, origin[1] + (NY // 2 + 0.61) * SP,
             origin[2] + (NZ // 2 + 0.52) * SP)
    return gf, probe


@pytest.mark.parametrize("method", list(METHODS))
def test_single_mixed_identical(method):
    """single and mixed must be bit-identical (real==float in both)."""
    gf, probe = _grid(False)
    es, fs = _eval(gf, method, probe, 'single')
    em, fm = _eval(gf, method, probe, 'mixed')
    assert es == em, f"{METHODS[method]}: single E {es} != mixed E {em}"
    assert np.array_equal(fs, fm), f"{METHODS[method]}: single/mixed force differ"


@pytest.mark.parametrize("method", list(METHODS))
def test_double_agrees(method):
    """double agrees with single to a loose tolerance (same algorithm, higher precision)."""
    gf, probe = _grid(False)
    es, fs = _eval(gf, method, probe, 'single')
    ed, fd = _eval(gf, method, probe, 'double')
    relf = np.linalg.norm(fd - fs) / max(1.0, np.linalg.norm(fs))
    assert relf < 1e-3, f"{METHODS[method]}: double force rel {relf:.2e} too large"


@pytest.mark.parametrize("method", list(METHODS))
def test_energy_single_mixed_identical(method):
    """Group energy: single and mixed are bit-identical."""
    gf, probe = _grid(False)
    es, _ = _eval(gf, method, probe, 'single')
    em, _ = _eval(gf, method, probe, 'mixed')
    assert es == em, f"{METHODS[method]}: single E {es} != mixed E {em}"


@pytest.mark.parametrize("method", list(METHODS))
def test_energy_double_finite_close(method):
    """Group energy: double is finite and close to single (its buffer is now mixed,
    so double carries higher precision rather than the old fp32-masked value)."""
    gf, probe = _grid(False)
    es, _ = _eval(gf, method, probe, 'single')
    ed, _ = _eval(gf, method, probe, 'double')
    assert np.isfinite(ed)
    assert abs(ed - es) / max(1.0, abs(es)) < 1e-4


def test_double_generation_carries_f64():
    """Double generation produces genuine f64 derivatives (oracle-free check)."""
    import validate_double_generation as vdg
    assert vdg.main() == 0


def test_tiled_double_storage_carries_f64():
    """Double storage survives the full tiled generate->disk->stream->kernel pipeline."""
    import validate_tiled_double_storage as vtds
    assert vtds.main() == 0


def _gen_invpower(prm, crd, pos_list, rec_atoms, origin, grid_file, mode, n):
    g = gfp.GridForce()
    g.setGridOrigin(*origin); g.addGridCounts(NX, NY, NZ); g.addGridSpacing(SP, SP, SP)
    g.setAutoGenerateGrid(True); g.setGridType('ljr'); g.setComputeDerivatives(True)
    g.setGridCap(1e30); g.setReceptorAtoms(rec_atoms)
    g.setReceptorPositionsFromLists(pos_list); g.setInvPowerMode(mode, n)
    gsys = prm.createSystem(nonbondedMethod=NoCutoff); gsys.addForce(g)
    ctx = Context(gsys, VerletIntegrator(0.001), mm.Platform.getPlatformByName('CUDA'))
    ctx.setPositions(crd.positions); ctx.getState(getEnergy=True)
    g.saveToFile(grid_file); del ctx


def _eval_invpower(grid_file, method, probe, precision, mode, n):
    gf = gfp.GridForce(); gf.loadFromFile(grid_file); gf.setInterpolationMethod(method)
    gf.setInvPowerMode(mode, n)
    s = mm.System(); s.addParticle(1.0); gf.addParticleGroup('ljr', [0], [1.0]); s.addForce(gf)
    ctx = Context(s, VerletIntegrator(0.001), mm.Platform.getPlatformByName('CUDA'),
                  {'Precision': precision})
    ctx.setPositions([mm.Vec3(*probe)] * 1 * nanometer)
    f = np.array(ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        kilojoules_per_mole / nanometer))[0]
    del ctx
    return f


@functools.lru_cache(maxsize=1)
def _invpower_grid():
    prm, crd, pos_list, rec_atoms, origin = _system()
    tmp = tempfile.mkdtemp()
    gfile = os.path.join(tmp, "ljr_rt.grid")
    _gen_invpower(prm, crd, pos_list, rec_atoms, origin, gfile, gfp.InvPowerMode_RUNTIME, 6.0)
    probe = (origin[0] + (NX // 2 + 0.37) * SP, origin[1] + (NY // 2 + 0.61) * SP,
             origin[2] + (NZ // 2 + 0.52) * SP)
    return gfile, probe


@pytest.mark.parametrize("method", [0, 3])
def test_invpower_runtime_precision(method):
    """inv_power RUNTIME path: single==mixed, double agrees."""
    gfile, probe = _invpower_grid()
    fs = _eval_invpower(gfile, method, probe, 'single', gfp.InvPowerMode_RUNTIME, 6.0)
    fm = _eval_invpower(gfile, method, probe, 'mixed', gfp.InvPowerMode_RUNTIME, 6.0)
    fd = _eval_invpower(gfile, method, probe, 'double', gfp.InvPowerMode_RUNTIME, 6.0)
    assert np.array_equal(fs, fm), f"invpower {METHODS[method]}: single/mixed differ"
    relf = np.linalg.norm(fd - fs) / max(1.0, np.linalg.norm(fs))
    assert relf < 1e-3, f"invpower {METHODS[method]}: double rel {relf:.2e}"


import time


def _bench(grid_file, method, precision, origin, natoms=2000, niter=100):
    """Median wall-time per force evaluation for `natoms` probe atoms."""
    rng = np.linspace(0.2, 0.8, int(round(natoms ** (1 / 3.0))) + 1)[1:]
    pts = [(origin[0] + a * NX * SP, origin[1] + b * NY * SP, origin[2] + c * NZ * SP)
           for a in rng for b in rng for c in rng][:natoms]
    n = len(pts)
    gf = gfp.GridForce(); gf.loadFromFile(grid_file); gf.setInterpolationMethod(method)
    s = mm.System()
    for _ in range(n):
        s.addParticle(1.0)
    gf.addParticleGroup('charge', list(range(n)), [1.0] * n)
    s.addForce(gf)
    ctx = Context(s, VerletIntegrator(0.001), mm.Platform.getPlatformByName('CUDA'),
                  {'Precision': precision})
    ctx.setPositions([mm.Vec3(*p) for p in pts] * nanometer)
    ctx.getState(getForces=True)  # warm up (JIT compile)
    ts = []
    for _ in range(niter):
        t0 = time.perf_counter()
        ctx.setPositions([mm.Vec3(*p) for p in pts] * nanometer)
        ctx.getState(getForces=True)
        ts.append(time.perf_counter() - t0)
    del ctx
    return float(np.median(ts)) * 1e3, n  # ms/call


def run_benchmark():
    prm, crd, pos_list, rec_atoms, origin = _system()
    tmp = tempfile.mkdtemp()
    gfile = os.path.join(tmp, "charge_bench.grid")
    _gen_grid(prm, crd, pos_list, rec_atoms, origin, gfile)
    print("\nforce eval wall-time (ms/call), median over 100 iters:")
    hdr = f"  {'method':<12}" + "".join(f"{p:>12}" for p in ('single', 'mixed', 'double'))
    print(hdr)
    for m in (0, 2, 3):
        row = f"  {METHODS[m]:<12}"
        for prec in ('single', 'mixed', 'double'):
            ms, n = _bench(gfile, m, prec, origin)
            row += f"{ms:>12.3f}"
        print(row + f"   ({n} atoms)")


if __name__ == "__main__":
    gf, probe = _grid(False)
    print(f"{'method':<12}{'single==mixed':<16}{'double rel(force)':<18}")
    for m, name in METHODS.items():
        es, fs = _eval(gf, m, probe, 'single')
        em, fm = _eval(gf, m, probe, 'mixed')
        ed, fd = _eval(gf, m, probe, 'double')
        ident = (es == em) and np.array_equal(fs, fm)
        relf = np.linalg.norm(fd - fs) / max(1.0, np.linalg.norm(fs))
        print(f"{name:<12}{str(ident):<16}{relf:<18.3e}")
    print("\ninv_power RUNTIME (ljr):")
    gfile, probe = _invpower_grid()
    for m in (0, 3):
        fs = _eval_invpower(gfile, m, probe, 'single', gfp.InvPowerMode_RUNTIME, 6.0)
        fm = _eval_invpower(gfile, m, probe, 'mixed', gfp.InvPowerMode_RUNTIME, 6.0)
        fd = _eval_invpower(gfile, m, probe, 'double', gfp.InvPowerMode_RUNTIME, 6.0)
        ident = np.array_equal(fs, fm)
        relf = np.linalg.norm(fd - fs) / max(1.0, np.linalg.norm(fs))
        print(f"{METHODS[m]:<12}{str(ident):<16}{relf:<18.3e}")
    run_benchmark()
