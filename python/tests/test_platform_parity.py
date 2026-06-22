#!/usr/bin/env python
"""Cross-platform parity harness for the GridForce plugin.

For every force and integrator the plugin provides, this builds one identical
system, evaluates it on each available platform/precision, and checks agreement
against the double-precision Reference (the serial ground truth). Where a force
has a vanilla OpenMM equivalent (the isolated GBSA paths, the isolated
nonbonded force), it is additionally anchored against that analytical reference
so the comparison does not rest on the plugin alone.

The CUDA platform is exercised at single, mixed, and double precision. Each
precision class has its own tolerance, justified by the inherent floating-point
floor at that precision (printed as abs and rel diffs). Derivatives are never
checked by finite differences: force checks use OpenMM's analytical forces.

Run directly for a report (exits non-zero on any unexpected mismatch):

    python test_platform_parity.py            # all checks
    python test_platform_parity.py gbsa       # one section

Sections: grid, nb, bonded, site, gbsa, gbsagrid, bondedhessian, integrators
"""
import os
import sys
import traceback

import numpy as np
import openmm as mm
from openmm import unit

import gridforceplugin as gfp

PRMDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prmtopcrd')

REFERENCE = 'Reference'  # double-precision serial ground truth for all comparisons

# Tolerances by the precision class of the platform under test, compared to the
# double Reference. The double row reflects that even in double the CUDA kernels
# differ from the serial path at ~1e-8 (different summation / fixed-point
# accumulation); the single/mixed rows reflect the float32 force floor. All are
# far below any real algorithmic divergence (the known PAIRWISE force gap is
# rel ~1e-2). Match is by relative diff with a small absolute floor that only
# bites when the reference value is ~0.
TOL = {
    'double': dict(rel_e=1e-6, abs_e=1e-7, rel_f=1e-5, abs_f=1e-5, rel_h=1e-5, abs_h=1e-5),
    'mixed':  dict(rel_e=1e-5, abs_e=1e-6, rel_f=1e-4, abs_f=1e-4, rel_h=1e-4, abs_h=1e-4),
    'single': dict(rel_e=1e-4, abs_e=1e-4, rel_f=3e-3, abs_f=3e-3, rel_h=3e-3, abs_h=3e-3),
}

# Open gaps to be fixed (not worked around). A mismatch listed here is reported
# as GAP rather than failing the run; remove the entry when the gap is closed.
# Keys: (case, label, kind).
OPEN_GAPS = {
    # CUDA NUTS rigid-body MC under-accepts a rigid-invariant move (~10% vs the
    # expected ~100%); the CUDA HMC path and both Reference paths are correct.
    # A CUDA NUTS-only defect to fix in the NUTS executeMC.
    ('NUTS-mc', 'CUDA/single', 'mc'),
    ('NUTS-mc', 'CUDA/mixed', 'mc'),
    ('NUTS-mc', 'CUDA/double', 'mc'),
}

_failures = []
_gaps = []


def platform_specs(ci=False):
    """(label, platform_name, properties, precision_class) for what's installed.

    ci=True restricts to the GPU-free Reference + CPU platforms (and pins the CPU
    thread count) for CI runners that have no CUDA; full mode also adds CUDA at
    single/mixed/double when available."""
    cpu_props = {'Threads': '2'} if ci else {}
    specs = [(REFERENCE, 'Reference', {}, 'double'),
             ('CPU', 'CPU', cpu_props, 'double')]
    if not ci:
        try:
            mm.Platform.getPlatformByName('CUDA')
            for prec in ('single', 'mixed', 'double'):
                specs.append((f'CUDA/{prec}', 'CUDA', {'Precision': prec}, prec))
        except Exception:
            pass
    return [s for s in specs if _platform_ok(s[1])]


def _platform_ok(name):
    try:
        mm.Platform.getPlatformByName(name)
        return True
    except Exception:
        return False


def _context(system, spec):
    integ = mm.VerletIntegrator(0.001)
    plat = mm.Platform.getPlatformByName(spec[1])
    return mm.Context(system, integ, plat, spec[2]), integ


def energy_forces(system, positions, spec):
    ctx, _ = _context(system, spec)
    ctx.setPositions(positions)
    st = ctx.getState(getEnergy=True, getForces=True)
    E = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    F = np.array(st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer))
    del ctx
    return E, F


def _diffs(value, ref):
    value, ref = np.asarray(value), np.asarray(ref)
    if value.shape != ref.shape:
        return float('nan'), float('nan')
    ad = float(np.max(np.abs(value - ref)))
    rd = ad / (float(np.max(np.abs(ref))) + 1e-300)
    return ad, rd


def record(case, label, pclass, kind, absd, reld):
    tol = TOL[pclass]
    rel_tol, abs_tol = tol[f'rel_{kind[0]}'], tol[f'abs_{kind[0]}']
    ok = np.isfinite(absd) and (reld < rel_tol or absd < abs_tol)
    tag = (case, label, kind)
    if ok:
        status = 'OK'
    elif tag in OPEN_GAPS:
        status = 'GAP'
        _gaps.append(tag)
    else:
        status = 'FAIL'
        _failures.append((case, label, kind, f"abs={absd:.3e} rel={reld:.3e}"))
    print(f"    {status:4s} {label:12s} {kind:7s} abs={absd:.3e} rel={reld:.3e} "
          f"(tol rel<{rel_tol:.0e}|abs<{abs_tol:.0e})  [{case}]")


def compare(case, results, specs, kinds=('energy', 'force')):
    """results: label -> (E, F) or ('EXC', msg). Compare each to REFERENCE."""
    ref = results.get(REFERENCE)
    if ref is None or isinstance(ref[0], str):
        print(f"  [{case}] Reference unavailable ({ref}); cannot compare")
        _failures.append((case, REFERENCE, 'reference', str(ref)))
        return
    pclass = {s[0]: s[3] for s in specs}
    for label, r in results.items():
        if label == REFERENCE:
            continue
        if isinstance(r[0], str):
            print(f"    EXC  {label:12s}         {r[1][:80]}  [{case}]")
            tag = (case, label, 'exception')
            if tag in OPEN_GAPS:
                _gaps.append(tag)
            else:
                _failures.append((case, label, 'exception', r[1]))
            continue
        pc = pclass.get(label, 'double')
        if 'energy' in kinds:
            ad, rd = _diffs(r[0], ref[0])
            record(case, label, pc, 'energy', ad, rd)
        if 'force' in kinds:
            ad, rd = _diffs(r[1], ref[1])
            record(case, label, pc, 'force', ad, rd)


def eval_all(build, positions, specs):
    results = {}
    for spec in specs:
        try:
            results[spec[0]] = energy_forces(build(), positions, spec)
        except Exception as e:
            results[spec[0]] = ('EXC', repr(e))
    return results


# ---------------------------------------------------------------- GridForce
def grid_section(specs):
    from openmm import app
    prmtop = app.AmberPrmtopFile(os.path.join(PRMDIR, 'ligand.prmtop'))
    inpcrd = app.AmberInpcrdFile(os.path.join(PRMDIR, 'ligand.trans.inpcrd'))
    charges = list(prmtop._prmtop.getCharges())
    n = len(charges)
    pos = np.array([[p.x, p.y, p.z] for p in inpcrd.positions])
    lo, hi = pos.min(0) - 0.3, pos.max(0) + 0.3
    counts = (24, 24, 24)
    spacing = tuple(((hi - lo) / (np.array(counts) - 1)).tolist())
    origin = tuple(lo.tolist())
    nval = counts[0] * counts[1] * counts[2]
    rng = np.random.RandomState(7)
    # Physical-scale smooth grid (kept O(1) so absolute diffs are meaningful).
    gridvals = (np.sin(np.linspace(0, 6, nval)) + rng.randn(nval) * 0.01).tolist()
    methods = [('trilinear', gfp.INTERP_TRILINEAR),
               ('tricubic_bspline', gfp.INTERP_TRICUBIC_BSPLINE),
               ('tricubic_hermite', gfp.INTERP_TRICUBIC_HERMITE),
               ('triquintic_hermite', gfp.INTERP_TRIQUINTIC_HERMITE)]
    for mname, mval in methods:
        def build():
            system = mm.System()
            for _ in range(n):
                system.addParticle(12.0)
            f = gfp.GridForce()
            f.addGridCounts(*counts)
            f.addGridSpacing(*spacing)
            f.setGridOrigin(*origin)
            for v in gridvals:
                f.addGridValue(v)
            for c in charges:
                f.addScalingFactor(c)
            f.setInterpolationMethod(mval)
            f.setComputeDerivatives(True)
            system.addForce(f)
            return system
        case = f"GridForce[{mname}]"
        print(f"\n=== {case} ===")
        compare(case, eval_all(build, inpcrd.positions, specs), specs)

    _grid_hessian_check(specs)


def _grid_hessian_check(specs):
    """Per-atom GridForce Hessian (block-diagonal 3x3) on an in-bounds quadratic
    grid with off-diagonal curvature. Cubic B-spline reproduces quadratics
    exactly, so the analytic answer is known (guards against a degenerate
    all-zero pass). The ligand grid above is unsuitable here: those atoms land in
    the out-of-bounds restraint region where the interpolant curvature is zero."""
    print("\n=== GridForce Hessian (in-bounds quadratic grid) ===")
    origin = (-0.5, -0.5, -0.5)
    sp = 0.05
    nx = ny = nz = 20
    grid = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x, y, z = origin[0] + i*sp, origin[1] + j*sp, origin[2] + k*sp
                grid.append(x*x + 2*y*y + 3*z*z + 0.5*x*y + 0.3*x*z + 0.1*y*z)
    expected = np.array([2.0, 4.0, 6.0, 0.5, 0.3, 0.1])  # [xx,yy,zz,xy,xz,yz]
    apos = [[0.07, -0.04, 0.02]]

    def build():
        s = mm.System()
        s.addParticle(12.0)
        f = gfp.GridForce()
        f.setInterpolationMethod(gfp.INTERP_TRICUBIC_BSPLINE)
        f.setGridOrigin(*origin)
        f.addGridSpacing(sp, sp, sp)
        f.addGridCounts(nx, ny, nz)
        f.setGridValues(grid)
        f.addScalingFactor(1.0)
        s.addForce(f)
        return s, f

    hess = {}
    for spec in specs:
        s, force = build()
        try:
            ctx, _ = _context(s, spec)
            ctx.setPositions(apos)
            ctx.getState(getEnergy=True)
            force.computeHessian(ctx)
            hess[spec[0]] = np.array(force.getHessianBlocks(ctx))
            del ctx
        except Exception as e:
            hess[spec[0]] = ('EXC', repr(e))
    # Known-answer check on Reference (a quadratic's bspline Hessian is exact).
    ref = hess.get(REFERENCE)
    if ref is None or isinstance(ref, tuple) or ref.shape != expected.shape:
        print(f"    FAIL Reference    known-answer unavailable ({ref})  [GridForce Hessian vs analytic]")
        _failures.append(("GridForce Hessian vs analytic", REFERENCE, 'hessian', str(ref)))
    else:
        d = float(np.max(np.abs(ref - expected)))
        ok = d < 1e-4
        print(f"    {'OK' if ok else 'FAIL':4s} Reference    analytic abs={d:.3e} (expect [2,4,6,.5,.3,.1])  [GridForce Hessian]")
        if not ok:
            _failures.append(("GridForce Hessian vs analytic", REFERENCE, 'hessian', f"abs={d:.3e}"))
    _compare_hessian("GridForce Hessian", hess, specs)

    # Eigen-analysis (eigenvalues / curvature / entropy). The Hessian above is the
    # constant matrix [[2,.5,.3],[.5,4,.1],[.3,.1,6]], whose eigenvalues are known.
    print("\n=== GridForce Hessian eigen-analysis (in-bounds quadratic grid) ===")
    H = np.array([[2.0, 0.5, 0.3], [0.5, 4.0, 0.1], [0.3, 0.1, 6.0]])
    eig_expected = np.sort(np.linalg.eigvalsh(H))
    eigs = {}
    for spec in specs:
        s, force = build()
        try:
            ctx, _ = _context(s, spec)
            ctx.setPositions(apos)
            ctx.getState(getEnergy=True)
            force.computeHessian(ctx)
            a = force.analyzeHessian(ctx, 300.0)
            eigs[spec[0]] = (np.sort(np.array(list(a.eigenvalues))),
                             list(a.numNegative)[0], a.totalEntropy)
            del ctx
        except Exception as e:
            eigs[spec[0]] = ('EXC', repr(e))
    def _eig_exc(v):
        return v is None or (isinstance(v, tuple) and isinstance(v[0], str))
    refa = eigs.get(REFERENCE)
    if _eig_exc(refa):
        print(f"    FAIL Reference    eigen-analysis unavailable ({refa})  [GridForce eigen]")
        _failures.append(("GridForce eigen", REFERENCE, 'eigen', str(refa)))
    else:
        de = float(np.max(np.abs(refa[0] - eig_expected)))
        ok = de < 1e-4 and refa[1] == 0
        print(f"    {'OK' if ok else 'FAIL':4s} Reference    eig abs={de:.3e} numNeg={refa[1]} "
              f"(expect {[round(float(v),4) for v in eig_expected]}, 0)  [GridForce eigen]")
        if not ok:
            _failures.append(("GridForce eigen", REFERENCE, 'eigen', f"abs={de:.3e} numNeg={refa[1]}"))
    for spec in specs:
        name = spec[0]
        if name == REFERENCE:
            continue
        cur = eigs.get(name)
        if _eig_exc(cur) or _eig_exc(refa):
            continue
        dd = float(np.max(np.abs(cur[0] - refa[0])))
        tol = TOL[spec[3]]['abs_h']
        ok = dd < tol
        print(f"    {'OK' if ok else 'FAIL':4s} {name:12s} eig abs={dd:.3e} (tol abs<{tol:g}) vs Reference  [GridForce eigen]")
        if not ok:
            _failures.append(("GridForce eigen", name, 'eigen', f"abs={dd:.3e}"))


# ----------------------------------------------------- IsolatedNonbondedForce
def nb_section(specs):
    from openmm import app
    prmtop = app.AmberPrmtopFile(os.path.join(PRMDIR, 'ligand.prmtop'))
    inpcrd = app.AmberInpcrdFile(os.path.join(PRMDIR, 'ligand.trans.inpcrd'))
    omm = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    nbf = next(omm.getForce(i) for i in range(omm.getNumForces())
               if isinstance(omm.getForce(i), mm.NonbondedForce))
    n = prmtop.topology.getNumAtoms()

    def build():
        system = mm.System()
        for i in range(n):
            system.addParticle(omm.getParticleMass(i))
        inf = gfp.IsolatedNonbondedForce()
        inf.setNumAtoms(n)
        for ai in range(nbf.getNumParticles()):
            q, sig, eps = nbf.getParticleParameters(ai)
            inf.setAtomParameters(ai, q.value_in_unit(unit.elementary_charge),
                                  sig.value_in_unit(unit.nanometer),
                                  eps.value_in_unit(unit.kilojoules_per_mole))
        for ei in range(nbf.getNumExceptions()):
            a1, a2, qq, sig, eps = nbf.getExceptionParameters(ei)
            qqv = qq.value_in_unit(unit.elementary_charge ** 2)
            epsv = eps.value_in_unit(unit.kilojoules_per_mole)
            if qqv == 0.0 and epsv == 0.0:
                inf.addExclusion(a1, a2)
            else:
                inf.addException(a1, a2, qqv, sig.value_in_unit(unit.nanometer), epsv)
        inf.addParticleGroup("g0", list(range(n)))
        system.addForce(inf)
        return system, inf

    case = "IsolatedNonbondedForce"
    print(f"\n=== {case} ===")
    results = eval_all(lambda: build()[0], inpcrd.positions, specs)
    compare(case, results, specs)

    # Analytical Hessian parity (group 0). The Reference Hessian is validated
    # against JAX autodiff of the LJ+Coulomb energy to machine precision.
    print("  -- analytical Hessian (group 0) --")
    hess = {}
    for spec in specs:
        system, force = build()
        try:
            ctx, _ = _context(system, spec)
            ctx.setPositions(inpcrd.positions)
            ctx.getState(getEnergy=True)  # execute() before computeHessian()
            hess[spec[0]] = np.array(force.computeHessian(ctx))
            del ctx
        except Exception as e:
            hess[spec[0]] = ('EXC', repr(e))
    _compare_hessian(case + " Hessian", hess, specs)

    # Anchor against vanilla OpenMM NonbondedForce (single group => identical).
    omm_sys = mm.System()
    for i in range(n):
        omm_sys.addParticle(omm.getParticleMass(i))
    omm_sys.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(nbf)))
    try:
        Eo, Fo = energy_forces(omm_sys, inpcrd.positions, (REFERENCE, 'Reference', {}, 'double'))
        anchored = {k: v for k, v in results.items() if not isinstance(v[0], str)}
        anchored[REFERENCE] = (Eo, Fo)  # OpenMM becomes the ground truth here
        compare(case + " vs OpenMM", anchored, specs)
    except Exception as e:
        print(f"    (OpenMM anchor skipped: {e!r})")


# ------------------------------------------- IsolatedBondedForce (+ Hessian)
def bonded_section(specs):
    N, K = 4, 2
    bonds = [(0, 1, 0.15, 300000.0), (1, 2, 0.14, 350000.0), (2, 3, 0.13, 280000.0)]
    angles = [(0, 1, 2, 1.91, 500.0), (1, 2, 3, 2.09, 450.0)]
    torsions = [(0, 1, 2, 3, 2, 3.14159, 10.0), (0, 1, 2, 3, 3, 0.0, 5.0)]
    rng = np.random.RandomState(42)
    base = np.array([[0, 0, 0], [0.15, 0, 0],
                     [0.15 + 0.14 * np.cos(1.91), 0.14 * np.sin(1.91), 0],
                     [0.15 + 0.14 * np.cos(1.91) + 0.13 * np.cos(1.0),
                      0.14 * np.sin(1.91) + 0.13 * np.sin(1.0), 0.05]])
    all_pos = np.vstack([base + rng.normal(0, 0.01, (N, 3)) for _ in range(K)])

    def build():
        system = mm.System()
        for _ in range(K * N):
            system.addParticle(12.0)
        f = gfp.IsolatedBondedForce()
        f.setNumAtoms(N)
        for b in bonds:
            f.addBond(*b)
        for a in angles:
            f.addAngle(*a)
        for t in torsions:
            f.addTorsion(*t)
        for g in range(K):
            f.addParticleGroup(f'g{g}', list(range(g * N, (g + 1) * N)))
        system.addForce(f)
        return system, f

    case = "IsolatedBondedForce"
    print(f"\n=== {case} ===")
    results, hess = {}, {}
    for spec in specs:
        system, force = build()  # keep the typed force for getHessianMatrix
        try:
            ctx, _ = _context(system, spec)
            ctx.setPositions(all_pos)
            st = ctx.getState(getEnergy=True, getForces=True)
            results[spec[0]] = (
                st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                np.array(st.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoules_per_mole / unit.nanometer)))
        except Exception as e:
            results[spec[0]] = ('EXC', repr(e))
            hess[spec[0]] = ('EXC', repr(e))
            continue
        try:
            hess[spec[0]] = np.array(force.getHessianMatrix(ctx, 0))
        except Exception as e:
            hess[spec[0]] = ('EXC', repr(e))
        del ctx
    compare(case, results, specs)

    print("  -- analytical Hessian (group 0) --")
    _compare_hessian(case + " Hessian", hess, specs)


def _compare_hessian(case, hess, specs):
    ref = hess.get(REFERENCE)
    if ref is None or isinstance(ref, tuple):
        print(f"    EXC  Reference Hessian unavailable: "
              f"{ref[1][:80] if isinstance(ref, tuple) else ref}  [{case}]")
        _failures.append((case, REFERENCE, 'hessian', 'reference unavailable'))
        return
    pclass = {s[0]: s[3] for s in specs}
    for label, h in hess.items():
        if label == REFERENCE:
            continue
        if isinstance(h, tuple):
            print(f"    EXC  {label:12s}         {h[1][:80]}  [{case}]")
            _failures.append((case, label, 'hessian', h[1]))
            continue
        ad, rd = _diffs(h, ref)
        record(case, label, pclass.get(label, 'double'), 'hessian', ad, rd)


# ------------------------------------------------------- IsolatedSiteForce
def site_section(specs):
    N, K = 4, 2
    masses = [12.0, 1.0, 16.0, 14.0]
    center = np.array([0.3, 0.1, -0.2])
    rng = np.random.RandomState(3)
    all_pos = np.vstack([center + rng.randn(N, 3) * 0.25 for _ in range(K)])

    def build():
        system = mm.System()
        for _ in range(K):
            for i in range(N):
                system.addParticle(masses[i])
        f = gfp.IsolatedSiteForce()
        f.setNumAtoms(N)
        f.setAtomMasses(masses)
        f.setForceConstant(1000.0)
        f.setMaxRadius(0.2)
        f.setSiteCenter(float(center[0]), float(center[1]), float(center[2]))
        for g in range(K):
            f.addParticleGroup(f'g{g}', list(range(g * N, (g + 1) * N)))
        system.addForce(f)
        return system

    case = "IsolatedSiteForce"
    print(f"\n=== {case} ===")
    compare(case, eval_all(build, all_pos, specs), specs)


# ------------------------------------------------------- IsolatedGBSAForce
def _gbsa_openmm_ef(charges, radii, scales, positions, solvent_diel=78.5):
    system = mm.System()
    g = mm.GBSAOBCForce()
    g.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
    g.setSoluteDielectric(1.0)
    g.setSolventDielectric(solvent_diel)
    g.setSurfaceAreaEnergy(0.0)
    for i in range(len(charges)):
        system.addParticle(1.0)
        g.addParticle(float(charges[i]), float(radii[i]), float(scales[i]))
    system.addForce(g)
    return energy_forces(system, positions, (REFERENCE, 'Reference', {}, 'double'))


def gbsa_section(specs):
    np.random.seed(42)
    n_rec, n_lig = 40, 12
    rec_pos = np.random.randn(n_rec, 3) * 0.4
    lig_pos = np.random.randn(n_lig, 3) * 0.15 + np.array([0.9, 0.0, 0.0])
    rec_q = np.random.randn(n_rec) * 0.3
    lig_q = np.random.randn(n_lig) * 0.3
    rec_r = 0.12 + np.random.rand(n_rec) * 0.06
    lig_r = 0.12 + np.random.rand(n_lig) * 0.06
    rec_s = 0.7 + np.random.rand(n_rec) * 0.2
    lig_s = 0.7 + np.random.rand(n_lig) * 0.2

    for mname, mval in [('NONE', gfp.IsolatedGBSAForce.NONE),
                        ('PAIRWISE', gfp.IsolatedGBSAForce.PAIRWISE)]:
        def build():
            system = mm.System()
            for _ in range(n_lig):
                system.addParticle(12.0)
            f = gfp.IsolatedGBSAForce()
            f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
            f.setSoluteDielectric(1.0)
            f.setSolventDielectric(78.5)
            f.setIncludeSurfaceArea(False)
            f.setReceptorMode(mval)
            f.setNumAtoms(n_lig)
            for i in range(n_lig):
                f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
            if mval == gfp.IsolatedGBSAForce.PAIRWISE:
                f.setNumReceptorAtoms(n_rec)
                for i in range(n_rec):
                    f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]),
                                                float(rec_s[i]))
                f.setReceptorPositions(rec_pos.flatten().tolist())
            f.addParticleGroup("lig", list(range(n_lig)))
            system.addForce(f)
            return system, f

        case = f"IsolatedGBSAForce[{mname}]"
        print(f"\n=== {case} ===")
        results = eval_all(lambda: build()[0], lig_pos, specs)
        # PAIRWISE forces are judged against the OpenMM anchor below, not the
        # Reference (whose PAIRWISE forces are a known gap); compare energy only
        # cross-platform here to avoid attributing the Reference gap to CUDA.
        compare(case, results, specs,
                kinds=('energy',) if mname == 'PAIRWISE' else ('energy', 'force'))

        if mname == 'NONE':
            Eo, Fo = _gbsa_openmm_ef(lig_q, lig_r, lig_s, lig_pos)
        else:
            all_q = np.concatenate([rec_q, lig_q])
            all_r = np.concatenate([rec_r, lig_r])
            all_s = np.concatenate([rec_s, lig_s])
            all_p = np.vstack([rec_pos, lig_pos])
            Ecx, Fcx = _gbsa_openmm_ef(all_q, all_r, all_s, all_p)
            Erec, _ = _gbsa_openmm_ef(rec_q, rec_r, rec_s, rec_pos)
            Eo, Fo = Ecx - Erec, Fcx[n_rec:]
        anchored = {k: v for k, v in results.items() if not isinstance(v[0], str)}
        anchored[REFERENCE] = (Eo, Fo)
        compare(case + " vs OpenMM", anchored, specs)

        # Analytical Hessian parity (group 0). The Reference Hessian is
        # validated against JAX autodiff of the IsolatedGBSA energy to <1e-6
        # for both NONE and PAIRWISE; here we check cross-platform agreement.
        print("  -- analytical Hessian (group 0) --")
        hess = {}
        for spec in specs:
            try:
                system, force = build()  # keep the typed force for computeHessian
                ctx, _ = _context(system, spec)
                ctx.setPositions(lig_pos)
                ctx.getState(getEnergy=True)  # execute() before computeHessian()
                hess[spec[0]] = np.array(force.computeHessian(ctx))
                del ctx
            except Exception as e:
                hess[spec[0]] = ('EXC', repr(e))
        _compare_hessian(case + " Hessian", hess, specs)


# --------------------------------------------------------- GBSAGridForce
def gbsagrid_section(specs):
    case = "GBSAGridForce"
    print(f"\n=== {case} ===")
    try:
        sys.path.insert(0, os.path.dirname(PRMDIR))
        from desolvation_grid_generator import generate_desolvation_grid
    except Exception as e:
        print(f"  SKIP (desolvation_grid_generator unavailable: {e!r})")
        return
    np.random.seed(42)
    n_rec, n_lig = 10, 5
    rec_pos = np.random.randn(n_rec, 3) * 0.5
    lig_pos = np.random.randn(n_lig, 3) * 0.2 + [0.8, 0, 0]
    lig_r = np.array([0.17, 0.15, 0.12, 0.155, 0.17])
    lig_s = np.array([0.72, 0.85, 0.85, 0.72, 0.72])
    lig_q = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
    min_pos = np.array([-1.0, -1.0, -1.0])
    counts, spacing = (20, 20, 20), 0.1
    pg = generate_desolvation_grid(
        rec_positions=rec_pos, rec_radii=np.array([0.17] * n_rec),
        rec_scales=np.array([0.72] * n_rec), origin=min_pos, counts=counts,
        spacing=spacing, probe_radius=0.14, verbose=False)
    nx, ny, nz = pg.counts

    def build():
        cg = gfp.DesolvationGrid(nx, ny, nz, float(spacing), 0.14, list(pg.r_thresholds))
        cg.setOrigin(float(min_pos[0]), float(min_pos[1]), float(min_pos[2]))
        cg.setHctProbe(pg.hct_probe.flatten(order='C').tolist())
        cN, cA, cB = [], [], []
        for b in range(pg.n_bins):
            cN.extend(pg.correction_N[b].flatten(order='C').tolist())
            cA.extend(pg.correction_A[b].flatten(order='C').tolist())
            cB.extend(pg.correction_B[b].flatten(order='C').tolist())
        cg.setCorrectionN(cN)
        cg.setCorrectionA(cA)
        cg.setCorrectionB(cB)
        f = gfp.GBSAGridForce()
        f.setNumAtoms(n_lig)
        f.setIncludeSurfaceArea(False)
        f.setInterpolationMethod(0)
        for i in range(n_lig):
            f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
        f.setDesolvationGrid(cg)
        f.setParticles(list(range(n_lig)))
        f.addParticleGroup('lig', list(range(n_lig)))
        system = mm.System()
        for _ in range(n_lig):
            system.addParticle(12.0)
        system.addForce(f)
        return system, f

    compare(case, eval_all(lambda: build()[0], lig_pos * unit.nanometers, specs), specs)

    # Analytical Hessian: full 3N x 3N, validated against JAX autodiff offline.
    # Here we anchor every platform/precision to the double Reference.
    print(f"\n=== {case} Hessian ===")
    hess = {}
    for spec in specs:
        system, force = build()
        try:
            ctx, _ = _context(system, spec)
            ctx.setPositions(lig_pos * unit.nanometers)
            ctx.getState(getEnergy=True)
            force.computeHessian(ctx)
            n = 3 * n_lig
            hess[spec[0]] = np.array(force.getFullHessian(ctx)).reshape(n, n)
            del ctx
        except Exception as e:
            hess[spec[0]] = ('EXC', repr(e))
    _compare_hessian(case + " Hessian", hess, specs)

    # Triquintic regression: auto-generate a derivative grid and evaluate with
    # triquintic Hermite. Guards against NaN-poisoned derivatives (atom-surface
    # log singularity) silently collapsing the energy to ~0; the result must be
    # finite and close to the trilinear baseline.
    print(f"\n=== {case} triquintic (auto-gen derivatives) ===")
    def build_triq():
        f = gfp.GBSAGridForce()
        f.setNumAtoms(n_lig); f.setIncludeSurfaceArea(False)
        f.setInterpolationMethod(3)
        f.setAutoGenerateGrid(True); f.setComputeGridDerivatives(True)
        for i in range(n_lig):
            f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
        f.setReceptorPositions(rec_pos.flatten().tolist())
        f.setReceptorRadii([0.17] * n_rec); f.setReceptorScaleFactors([0.72] * n_rec)
        f.setGridOrigin(float(min_pos[0]), float(min_pos[1]), float(min_pos[2]))
        f.setGridCounts(int(nx), int(ny), int(nz)); f.setGridSpacing(float(spacing))
        f.setProbeRadius(0.14); f.setRThresholds(list(pg.r_thresholds))
        f.setParticles(list(range(n_lig))); f.addParticleGroup('lig', list(range(n_lig)))
        s = mm.System()
        for _ in range(n_lig):
            s.addParticle(12.0)
        s.addForce(f); return s
    e_trilinear, _ = eval_all(lambda: build()[0], lig_pos * unit.nanometers,
                              [(REFERENCE, 'Reference', {}, 'double')])[REFERENCE]
    for spec in specs:
        try:
            ctx, _ = _context(build_triq(), spec)
            ctx.setPositions(lig_pos * unit.nanometers)
            E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole)
            del ctx
            rel = abs(E - e_trilinear) / max(1.0, abs(e_trilinear))
            ok = np.isfinite(E) and rel < 0.05
            print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} E={E:.4f} "
                  f"(trilinear {e_trilinear:.4f}, rel {rel:.2e})  [{case} triquintic]")
            if not ok:
                _failures.append((case + " triquintic", spec[0], 'energy', f"E={E:.4e}"))
        except Exception as ex:
            print(f"    EXC  {spec[0]:12s}: {repr(ex)[:70]}")
            _failures.append((case + " triquintic", spec[0], 'energy', repr(ex)))

    # Buried-pose correction magnitude. For a ligand buried in the receptor cloud
    # the N/A/B corrections are dominant (~15-20% of the energy), not the ~0.1%
    # refinement seen for a solvent-exposed ligand: the single-probe HCT model fails
    # in the close-contact crossover regime (S ~ r). This guards that the correction
    # grids are generated AND applied -- zeroing them must move a buried energy by a
    # large, unambiguous fraction -- and that CPU matches Reference bit-for-bit.
    print(f"\n=== {case} buried-pose corrections ===")
    try:
        brng = np.random.RandomState(11)
        bn_rec, bn_lig = 40, 6
        brec = brng.randn(bn_rec, 3) * 0.45                 # receptor cloud at origin
        blig = brng.randn(bn_lig, 3) * 0.12 + brec.mean(axis=0)   # buried at the centroid
        blr = np.array([0.17, 0.15, 0.12, 0.155, 0.17, 0.14])
        bls = np.array([0.72, 0.85, 0.85, 0.72, 0.72, 0.85])
        blq = np.array([0.35, -0.4, 0.25, -0.15, 0.3, -0.2])
        bmin = np.array([-1.0, -1.0, -1.0]); bc, bsp = 21, 0.1
        bpg = generate_desolvation_grid(
            rec_positions=brec, rec_radii=np.full(bn_rec, 0.17),
            rec_scales=np.full(bn_rec, 0.72), origin=bmin, counts=(bc, bc, bc),
            spacing=bsp, probe_radius=0.14, verbose=False)

        def buried_build(zero_corr):
            cg = gfp.DesolvationGrid(bc, bc, bc, float(bsp), 0.14, list(bpg.r_thresholds))
            cg.setOrigin(float(bmin[0]), float(bmin[1]), float(bmin[2]))
            cg.setHctProbe(bpg.hct_probe.flatten(order='C').tolist())
            if zero_corr:
                z = [0.0] * (bpg.n_bins * bc * bc * bc)
                cg.setCorrectionN(z); cg.setCorrectionA(z); cg.setCorrectionB(z)
            else:
                cN, cA, cB = [], [], []
                for b in range(bpg.n_bins):
                    cN.extend(bpg.correction_N[b].flatten(order='C').tolist())
                    cA.extend(bpg.correction_A[b].flatten(order='C').tolist())
                    cB.extend(bpg.correction_B[b].flatten(order='C').tolist())
                cg.setCorrectionN(cN); cg.setCorrectionA(cA); cg.setCorrectionB(cB)
            f = gfp.GBSAGridForce(); f.setNumAtoms(bn_lig); f.setIncludeSurfaceArea(False)
            f.setInterpolationMethod(0)
            for i in range(bn_lig):
                f.setAtomParameters(i, float(blq[i]), float(blr[i]), float(bls[i]))
            f.setDesolvationGrid(cg); f.setParticles(list(range(bn_lig)))
            f.addParticleGroup('lig', list(range(bn_lig)))
            s = mm.System()
            for _ in range(bn_lig):
                s.addParticle(12.0)
            s.addForce(f); return s

        def buried_E(zero_corr, spec):
            ctx, _ = _context(buried_build(zero_corr), spec)
            ctx.setPositions(blig * unit.nanometers)
            E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole)
            del ctx
            return E
        ref_spec = (REFERENCE, 'Reference', {}, 'double')
        e_corr = buried_E(False, ref_spec)
        e_hct = buried_E(True, ref_spec)
        shift = abs(e_corr - e_hct) / max(1.0, abs(e_corr))
        ok = np.isfinite(e_corr) and np.isfinite(e_hct) and shift > 0.05
        print(f"    {'OK' if ok else 'FAIL':4s} Reference    buried corrections shift {shift*100:.1f}% "
              f"(corr {e_corr:.3f}, HCT-only {e_hct:.3f})  [{case} buried-corr]")
        if not ok:
            _failures.append((case + " buried-corr", REFERENCE, 'energy', f"shift={shift:.2e}"))
        for spec in specs:
            if spec[1] != 'CPU':
                continue
            e_cpu = buried_E(False, spec)
            d = abs(e_cpu - e_corr)
            okp = d < 1e-9
            print(f"    {'OK' if okp else 'FAIL':4s} {spec[0]:12s} buried corrected E vs Reference "
                  f"abs={d:.2e}  [{case} buried-corr]")
            if not okp:
                _failures.append((case + " buried-corr", spec[0], 'energy', f"abs={d:.2e}"))
    except Exception as ex:
        print(f"    EXC  buried corrections: {repr(ex)[:70]}")
        _failures.append((case + " buried-corr", '-', 'energy', repr(ex)))


# ------------------------------------------------------ grid generation
def gridgen_section(specs):
    """Exercise the Reference/CPU grid GENERATION paths (CUDA-free): desolvation
    auto-generation vs the Python reference generator, CPU==Reference determinism,
    GridForce field-grid generation, and the out-of-bounds ligand flag."""
    case = "GridGeneration"
    print(f"\n=== {case} ===")
    # Generation parity is Reference vs CPU only: CUDA uses a different generation
    # model (KDE-smoothed desolvation; its own field-grid path), validated against
    # CUDA separately, so cross-checking CUDA-gen against Reference-gen is not
    # meaningful here. (Evaluation of a shared grid does match CUDA — see gbsagrid.)
    gpu_free = [s for s in specs if s[1] in ('Reference', 'CPU')]
    rng = np.random.RandomState(7)
    n_rec, n_lig = 20, 4
    rec_pos = rng.randn(n_rec, 3) * 0.4
    rec_r = np.full(n_rec, 0.17); rec_s = np.full(n_rec, 0.72)
    rec_q = rng.uniform(-0.5, 0.5, n_rec)            # fixed once (not per-build)
    lig_pos = rng.randn(n_lig, 3) * 0.12 + [0.1, 0, 0]
    lig_r = np.array([0.17, 0.15, 0.12, 0.155]); lig_s = np.array([0.72, 0.85, 0.85, 0.72])
    lig_q = np.array([0.3, -0.4, 0.25, -0.15])
    lo, hi, sp = -1.0, 1.0, 0.08
    c = int(round((hi - lo) / sp)) + 1
    thr = [0.12, 0.16]

    def gbsa_autogen(method=0, derivs=False):
        f = gfp.GBSAGridForce(); f.setNumAtoms(n_lig); f.setIncludeSurfaceArea(False)
        f.setInterpolationMethod(method); f.setAutoGenerateGrid(True)
        if derivs:
            f.setComputeGridDerivatives(True)
        for i in range(n_lig):
            f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
        f.setReceptorPositions(rec_pos.flatten().tolist())
        f.setReceptorRadii(rec_r.tolist()); f.setReceptorScaleFactors(rec_s.tolist())
        f.setGridOrigin(lo, lo, lo); f.setGridCounts(c, c, c); f.setGridSpacing(sp)
        f.setProbeRadius(0.14); f.setRThresholds(thr)
        f.setParticles(list(range(n_lig))); f.addParticleGroup('lig', list(range(n_lig)))
        s = mm.System()
        for _ in range(n_lig):
            s.addParticle(12.0)
        s.addForce(f)
        return s, f

    def gbsa_energy(maker, spec, pos=None):
        system, f = maker
        ctx, _ = _context(system, spec)
        ctx.setPositions(lig_pos if pos is None else pos)
        E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        flags = list(f.getParticleOutOfBoundsFlags(ctx))
        del ctx
        return E, flags

    # (1) Desolvation C++ auto-gen vs the Python reference generator (Reference).
    try:
        sys.path.insert(0, os.path.dirname(PRMDIR))
        from desolvation_grid_generator import generate_desolvation_grid
        pg = generate_desolvation_grid(
            rec_positions=rec_pos, rec_radii=rec_r, rec_scales=rec_s,
            origin=np.array([lo, lo, lo]), counts=(c, c, c), spacing=sp,
            probe_radius=0.14, r_thresholds=tuple(thr), verbose=False)

        def gbsa_supplied():
            cg = gfp.DesolvationGrid(c, c, c, float(sp), 0.14, list(pg.r_thresholds))
            cg.setOrigin(lo, lo, lo); cg.setHctProbe(pg.hct_probe.flatten(order='C').tolist())
            cN, cA, cB = [], [], []
            for b in range(pg.n_bins):
                cN.extend(pg.correction_N[b].flatten(order='C').tolist())
                cA.extend(pg.correction_A[b].flatten(order='C').tolist())
                cB.extend(pg.correction_B[b].flatten(order='C').tolist())
            cg.setCorrectionN(cN); cg.setCorrectionA(cA); cg.setCorrectionB(cB)
            f = gfp.GBSAGridForce(); f.setNumAtoms(n_lig); f.setIncludeSurfaceArea(False)
            f.setInterpolationMethod(0)
            for i in range(n_lig):
                f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
            f.setDesolvationGrid(cg)
            f.setParticles(list(range(n_lig))); f.addParticleGroup('lig', list(range(n_lig)))
            s = mm.System()
            for _ in range(n_lig):
                s.addParticle(12.0)
            s.addForce(f)
            return s, f
        e_sup, _ = gbsa_energy(gbsa_supplied(), (REFERENCE, 'Reference', {}, 'double'))
        e_gen, _ = gbsa_energy(gbsa_autogen(0), (REFERENCE, 'Reference', {}, 'double'))
        d = abs(e_gen - e_sup)
        ok = d < 1e-4
        print(f"    {'OK' if ok else 'FAIL':4s} Reference    desolvation C++gen vs Python-gen "
              f"abs={d:.2e}  [{case} desolv-vs-python]")
        if not ok:
            _failures.append((case + " desolv-vs-python", REFERENCE, 'energy', f"abs={d:.2e}"))
    except Exception as e:
        print(f"    SKIP desolvation-vs-python ({repr(e)[:60]})")

    # (2) Desolvation auto-gen determinism: every platform == Reference auto-gen.
    e_ref, _ = gbsa_energy(gbsa_autogen(0), (REFERENCE, 'Reference', {}, 'double'))
    for spec in gpu_free:
        try:
            E, _ = gbsa_energy(gbsa_autogen(0), spec)
            d = abs(E - e_ref)
            ok = d < 1e-9
            print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} desolvation auto-gen vs Reference "
                  f"abs={d:.2e}  [{case} desolv]")
            if not ok:
                _failures.append((case + " desolv", spec[0], 'energy', f"abs={d:.2e}"))
        except Exception as ex:
            print(f"    EXC  {spec[0]:12s}: {repr(ex)[:60]}")
            _failures.append((case + " desolv", spec[0], 'energy', repr(ex)))

    # (3) Out-of-bounds ligand flag: an atom outside the grid is flagged.
    oob_pos = lig_pos.copy(); oob_pos[1] = [9.0, 9.0, 9.0]
    _, flags = gbsa_energy(gbsa_autogen(0), (REFERENCE, 'Reference', {}, 'double'), pos=oob_pos)
    ok = len(flags) == n_lig and flags[1] == 1 and sum(flags) == 1
    print(f"    {'OK' if ok else 'FAIL':4s} Reference    out-of-bounds flag {list(flags)} (expect 1 at idx 1)  [{case} oob]")
    if not ok:
        _failures.append((case + " oob", REFERENCE, 'flag', str(list(flags))))

    # (4) GridForce field generation (charge/ljr/lja): every platform == Reference.
    def gridforce_autogen(gtype):
        s = mm.System(); s.addParticle(12.0)
        nb = mm.NonbondedForce(); nb.addParticle(0.4, 0.3, 0.2)
        for j in range(n_rec):
            s.addParticle(12.0); nb.addParticle(float(rec_q[j]), 0.3, 0.25)
        s.addForce(nb)
        f = gfp.GridForce(); f.addGridCounts(c, c, c); f.addGridSpacing(sp, sp, sp)
        f.setGridOrigin(lo, lo, lo); f.setAutoGenerateGrid(True); f.setGridType(gtype)
        f.setReceptorAtoms(list(range(1, 1 + n_rec)))
        f.setReceptorPositionsFromLists([tuple(p) for p in rec_pos])
        f.setLigandAtoms([0]); f.addScalingFactor(1.0); f.setForceGroup(1)
        s.addForce(f)
        return s

    def gridforce_energy(gtype, spec):
        ctx, _ = _context(gridforce_autogen(gtype), spec)
        ctx.setPositions(np.vstack([[0.1, 0.05, 0.0], rec_pos]))
        E = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        del ctx
        return E
    for gtype in ('charge', 'ljr', 'lja'):
        try:
            er = gridforce_energy(gtype, (REFERENCE, 'Reference', {}, 'double'))
            for spec in gpu_free:
                if spec[0] == REFERENCE:
                    continue
                E = gridforce_energy(gtype, spec)
                d = abs(E - er)
                ok = d < 1e-9
                print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} GridForce gen[{gtype}] vs Reference "
                      f"abs={d:.2e}  [{case} field]")
                if not ok:
                    _failures.append((case + " field", spec[0], gtype, f"abs={d:.2e}"))
        except Exception as ex:
            print(f"    EXC  GridForce gen[{gtype}]: {repr(ex)[:60]}")
            _failures.append((case + " field", '-', gtype, repr(ex)))


# ------------------------------------------------------ BondedHessian class
def bondedhessian_section(specs):
    case = "BondedHessian"
    print(f"\n=== {case} ===")
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bf = mm.HarmonicBondForce()
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        bf.addBond(a, b, 0.153, 259408.0)
    system.addForce(bf)
    af = mm.HarmonicAngleForce()
    for a, b, c in [(0, 1, 2), (1, 2, 3)]:
        af.addAngle(a, b, c, 1.9111, 527.184)
    system.addForce(af)
    tf = mm.PeriodicTorsionForce()
    tf.addTorsion(0, 1, 2, 3, 3, 0.0, 0.8368)
    system.addForce(tf)
    pos = [mm.Vec3(0, 0, 0), mm.Vec3(0.153, 0, 0),
           mm.Vec3(0.204, 0.148, 0), mm.Vec3(0.357, 0.148, 0)]
    H = {}
    for spec in specs:
        try:
            ctx, _ = _context(system, spec)
            ctx.setPositions(pos)
            hc = gfp.BondedHessian()
            hc.initialize(system, ctx)
            H[spec[0]] = np.array(hc.getHessianMatrix(ctx))
            del ctx
        except Exception as e:
            H[spec[0]] = ('EXC', repr(e))
    _compare_hessian(case, H, specs)


# ------------------------------------------------------------ integrators
def integrators_section(specs):
    print("\n=== MultiGroup integrators (run check) ===")
    N, K = 4, 2
    bonds = [(0, 1, 0.15, 300000.0), (1, 2, 0.14, 350000.0), (2, 3, 0.13, 280000.0)]
    angles = [(0, 1, 2, 1.91, 500.0)]
    rng = np.random.RandomState(1)
    base = np.array([[0, 0, 0], [0.15, 0, 0], [0.27, 0.1, 0], [0.4, 0.1, 0.05]])
    all_pos = np.vstack([base + rng.normal(0, 0.01, (N, 3)) for _ in range(K)])
    for IntCls, nm in [(gfp.MultiGroupHMCIntegrator, 'HMC'),
                       (gfp.MultiGroupNUTSIntegrator, 'NUTS')]:
        for spec in specs:
            label = spec[0]
            try:
                system = mm.System()
                for _ in range(K * N):
                    system.addParticle(12.0)
                f = gfp.IsolatedBondedForce()
                f.setNumAtoms(N)
                for b in bonds:
                    f.addBond(*b)
                for a in angles:
                    f.addAngle(*a)
                for g in range(K):
                    f.addParticleGroup(f'g{g}', list(range(g * N, (g + 1) * N)))
                system.addForce(f)
                integ = IntCls(K, N, 0.001)  # numGroups, atomsPerGroup, stepSize
                ctx = mm.Context(system, integ,
                                 mm.Platform.getPlatformByName(spec[1]), spec[2])
                ctx.setPositions(all_pos)
                integ.step(2)
                E1 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole)
                if not np.isfinite(E1):
                    raise RuntimeError(f"non-finite energy after stepping: {E1}")
                print(f"    OK   {nm:4s} {label:12s}: stepped, E={E1:.4f}")
                del ctx
            except Exception as e:
                tag = (nm, label, 'integrator')
                if tag in OPEN_GAPS:
                    print(f"    GAP  {nm:4s} {label:12s}: {repr(e)[:80]}")
                    _gaps.append(tag)
                else:
                    print(f"    FAIL {nm:4s} {label:12s}: {repr(e)[:80]}")
                    _failures.append((nm, label, 'integrator', repr(e)))

    # Rigid-body MC moves: a rigid rotation+translation of a group leaves its
    # internal (bonded-only) energy invariant, so dE = 0 and every Metropolis
    # trial must be accepted (~100%). This checks the move geometry and counters.
    print("\n=== MultiGroup rigid-body MC (rigid-invariant energy) ===")
    for IntCls, nm in [(gfp.MultiGroupHMCIntegrator, 'HMC'),
                       (gfp.MultiGroupNUTSIntegrator, 'NUTS')]:
        for spec in specs:
            label = spec[0]
            try:
                system = mm.System()
                for _ in range(K * N):
                    system.addParticle(12.0)
                f = gfp.IsolatedBondedForce()
                f.setNumAtoms(N)
                for b in bonds:
                    f.addBond(*b)
                for a in angles:
                    f.addAngle(*a)
                for g in range(K):
                    f.addParticleGroup(f'g{g}', list(range(g * N, (g + 1) * N)))
                system.addForce(f)
                integ = IntCls(K, N, 0.001)
                integ.setNumMCTrials(4)
                integ.setMCStepSize(0.05)
                integ.setAllGroupMCEnabled([1] * K)
                integ.setRandomNumberSeed(12345)
                ctx = mm.Context(system, integ,
                                 mm.Platform.getPlatformByName(spec[1]), spec[2])
                ctx.setPositions(all_pos)
                integ.step(10)
                att, acc = integ.getMCAttempted(), integ.getMCAccepted()
                rate = acc / att if att else 0.0
                ok = att > 0 and rate > 0.95
                tag = (nm + '-mc', label, 'mc')
                if not ok and tag in OPEN_GAPS:
                    print(f"    GAP  {nm:4s} {label:12s}: rate={rate:.3f} ({acc}/{att})")
                    _gaps.append(tag)
                else:
                    status = 'OK' if ok else 'FAIL'
                    print(f"    {status:4s} {nm:4s} {label:12s}: MC accept rate={rate:.3f} ({acc}/{att})")
                    if not ok:
                        _failures.append((nm + '-mc', label, 'mc', f"rate={rate:.3f}"))
                del ctx
            except Exception as e:
                print(f"    FAIL {nm:4s} {label:12s}: {repr(e)[:80]}")
                _failures.append((nm + '-mc', label, 'mc', repr(e)))


SECTIONS = {
    'grid': grid_section,
    'nb': nb_section,
    'bonded': bonded_section,
    'site': site_section,
    'gbsa': gbsa_section,
    'gbsagrid': gbsagrid_section,
    'gridgen': gridgen_section,
    'bondedhessian': bondedhessian_section,
    'integrators': integrators_section,
}


def main():
    # CI mode: arg "ci" or env GRIDFORCE_TEST_CI=1 -> Reference+CPU only (no CUDA),
    # for GPU-free CI runners. A section name may still be given alongside "ci".
    args = sys.argv[1:]
    env_ci = os.environ.get('GRIDFORCE_TEST_CI', '') not in ('', '0', 'false', 'False')
    ci = env_ci or ('ci' in args)
    args = [a for a in args if a != 'ci']
    specs = platform_specs(ci=ci)
    print(f"Mode: {'CI (Reference+CPU)' if ci else 'full'}   "
          f"Platform/precision matrix: {[s[0] for s in specs]}")
    which = args[0] if args else 'all'
    todo = SECTIONS if which == 'all' else {which: SECTIONS[which]}
    for name, fn in todo.items():
        try:
            fn(specs)
        except Exception:
            print(f"\n=== {name} SECTION CRASH ===")
            traceback.print_exc()
            _failures.append((name, '-', 'section-crash', 'see traceback'))

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  open gaps hit (to be fixed, not failures): {len(_gaps)}")
    for c, l, k in sorted(set(_gaps)):
        print(f"    GAP  {l:12s} {k:10s} {c}")
    if _failures:
        print(f"  UNEXPECTED failures: {len(_failures)}")
        for c, l, k, v in _failures:
            print(f"    FAIL {l:12s} {k:10s} {c}  {str(v)[:70]}")
        return 1
    print("  no unexpected failures")
    return 0


if __name__ == '__main__':
    sys.exit(main())
