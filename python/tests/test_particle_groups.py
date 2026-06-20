"""
Regression for the k-particle-groups (multi-ligand) feature, with emphasis on the
per-group / per-atom / total energy buffers (now mixed precision).

Builds K groups of M atoms each inside a charge grid and checks, across
single/mixed/double:
  - getParticleGroupEnergies returns K values that sum to the total potential energy
  - getParticleAtomEnergies sums to the total
  - getParticleGroupAtomRawEnergies returns K*M finite values (double getter)
  - alchemical per-group scaling (setParticleGroupScalingFactor + update) zeroes a
    group's contribution and drops the total accordingly
  - single/mixed forces are bit-identical; energies agree to float-summation
    tolerance (mixed/double accumulate the per-group/total energy in double)

Run under pytest, or directly as a script.
"""
import os, sys, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pytest
import openmm as mm
from openmm import Context, VerletIntegrator
from openmm.unit import nanometer, kilojoules_per_mole
import gridforceplugin as gfp
from test_precision_suite import _system, _gen_grid, NX, NY, NZ, SP, cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA not available")

K = 3      # particle groups (ligands/poses)
M = 8      # atoms per group


def _positions(origin):
    """K*M distinct positions in the grid interior [0.2, 0.8] of each axis."""
    rng = np.random.default_rng(0)
    fr = 0.2 + 0.6 * rng.random((K * M, 3))
    return [(origin[0] + f[0] * NX * SP, origin[1] + f[1] * NY * SP,
             origin[2] + f[2] * NZ * SP) for f in fr]


import functools


@functools.lru_cache(maxsize=1)
def _grid():
    prm, crd, pos_list, rec_atoms, origin = _system()
    tmp = tempfile.mkdtemp()
    gf = os.path.join(tmp, "charge.grid")
    _gen_grid(prm, crd, pos_list, rec_atoms, origin, gf)
    return gf, origin


def _build(grid_file, precision, group_scales=None):
    gf = gfp.GridForce()
    gf.loadFromFile(grid_file)
    gf.setInterpolationMethod(3)
    s = mm.System()
    for _ in range(K * M):
        s.addParticle(1.0)
    for g in range(K):
        idx = list(range(g * M, (g + 1) * M))
        # per-atom scaling factors (non-trivial, varies per group)
        scal = [1.0 + 0.1 * g] * M
        gf.addParticleGroup(f"lig{g}", idx, scal)
    if group_scales:
        for g, f in enumerate(group_scales):
            gf.setParticleGroupScalingFactor(g, f)
    s.addForce(gf)
    ctx = Context(s, VerletIntegrator(0.001), mm.Platform.getPlatformByName('CUDA'),
                  {'Precision': precision})
    return gf, ctx


def _readout(gf, ctx, origin):
    ctx.setPositions([mm.Vec3(*p) for p in _positions(origin)] * nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    total = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    frc = np.array(st.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer))
    grp = np.array(gf.getParticleGroupEnergies(ctx))
    atom = np.array(gf.getParticleAtomEnergies(ctx))
    raw = np.array(gf.getParticleGroupAtomRawEnergies(ctx))
    return total, grp, atom, raw, frc


@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
def test_group_energy_bookkeeping(precision):
    grid_file, origin = _grid()
    gf, ctx = _build(grid_file, precision)
    total, grp, atom, raw, _ = _readout(gf, ctx, origin)
    assert gf.getNumParticleGroups() == K
    assert grp.shape == (K,)
    assert atom.shape == (K * M,)
    assert raw.shape == (K * M,)
    assert np.all(np.isfinite(grp)) and np.all(np.isfinite(atom)) and np.all(np.isfinite(raw))
    # per-group energies sum to the total potential energy
    assert abs(grp.sum() - total) / max(1.0, abs(total)) < 1e-6
    # per-atom energies also sum to the total
    assert abs(atom.sum() - total) / max(1.0, abs(total)) < 1e-5
    del ctx


def test_single_mixed_force_identical_energy_close():
    """Forces are bit-identical in single vs mixed (fixed-point). Energies are NOT:
    the per-group/total energy now accumulates in the double mixed buffer, so mixed
    is more precise than single -- they agree only to float-summation tolerance."""
    grid_file, origin = _grid()
    gfs, cs = _build(grid_file, 'single')
    ts, gs, _, _, fs = _readout(gfs, cs, origin)
    del cs
    gfm, cm = _build(grid_file, 'mixed')
    tm, gm, _, _, fm = _readout(gfm, cm, origin)
    del cm
    assert np.array_equal(fs, fm), "single/mixed forces must be bit-identical"
    assert abs(ts - tm) / max(1.0, abs(ts)) < 1e-6


def test_modes_agree():
    """single/mixed/double per-group energies agree to float-summation tolerance,
    and each mode's groups sum to its own total."""
    grid_file, origin = _grid()
    res = {}
    for prec in ("single", "mixed", "double"):
        gf, ctx = _build(grid_file, prec)
        total, grp, _, _, _ = _readout(gf, ctx, origin)
        res[prec] = (total, grp)
        assert abs(grp.sum() - total) / max(1.0, abs(total)) < 1e-6
        del ctx
    ts, _ = res['single']; td, gd = res['double']
    assert np.all(np.isfinite(gd))
    assert abs(ts - td) / max(1.0, abs(td)) < 1e-5


@pytest.mark.parametrize("precision", ["single", "double"])
def test_alchemical_group_scaling(precision):
    """Zeroing a group's scaling removes exactly its contribution from the total."""
    grid_file, origin = _grid()
    gf, ctx = _build(grid_file, precision)
    total0, grp0, _, _, _ = _readout(gf, ctx, origin)
    # turn off group 0 alchemically
    gf.setParticleGroupScalingFactor(0, 0.0)
    gf.updateParametersInContext(ctx)
    total1, grp1, _, _, _ = _readout(gf, ctx, origin)
    assert abs(grp1[0]) < 1e-6 * max(1.0, abs(grp0[0]))
    # total drops by exactly group 0's original contribution
    assert abs((total0 - total1) - grp0[0]) / max(1.0, abs(grp0[0])) < 1e-5
    # the other groups are unchanged
    assert np.allclose(grp1[1:], grp0[1:], rtol=1e-5, atol=1e-6)
    del ctx


if __name__ == "__main__":
    grid_file, origin = _grid()
    print(f"{K} groups x {M} atoms, charge grid, triquintic")
    print(f"  {'precision':<10}{'total':>14}{'sum(group)':>14}{'sum(atom)':>14}{'#raw':>7}")
    for prec in ("single", "mixed", "double"):
        gf, ctx = _build(grid_file, prec)
        total, grp, atom, raw, _ = _readout(gf, ctx, origin)
        print(f"  {prec:<10}{total:>14.6f}{grp.sum():>14.6f}{atom.sum():>14.6f}{len(raw):>7}")
        del ctx
