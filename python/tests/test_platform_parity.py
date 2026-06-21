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
    # Reference/CPU isolated-GBSA PAIRWISE omits one cross-term chain-rule
    # contribution (dependence of ligand-screened receptor Born radii on ligand
    # positions). Energy is exact; only the ligand forces drift. CUDA is correct,
    # so the gap shows up against the OpenMM force anchor (CPU mirrors Reference).
    ('IsolatedGBSAForce[PAIRWISE] vs OpenMM', 'CPU', 'force'),
}

_failures = []
_gaps = []


def platform_specs():
    """(label, platform_name, properties, precision_class) for what's installed."""
    specs = [(REFERENCE, 'Reference', {}, 'double'),
             ('CPU', 'CPU', {}, 'double')]
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
        return system

    case = "IsolatedNonbondedForce"
    print(f"\n=== {case} ===")
    results = eval_all(build, inpcrd.positions, specs)
    compare(case, results, specs)

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
            return system

        case = f"IsolatedGBSAForce[{mname}]"
        print(f"\n=== {case} ===")
        results = eval_all(build, lig_pos, specs)
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
        return system

    compare(case, eval_all(build, lig_pos * unit.nanometers, specs), specs)


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


SECTIONS = {
    'grid': grid_section,
    'nb': nb_section,
    'bonded': bonded_section,
    'site': site_section,
    'gbsa': gbsa_section,
    'gbsagrid': gbsagrid_section,
    'bondedhessian': bondedhessian_section,
    'integrators': integrators_section,
}


def main():
    specs = platform_specs()
    print(f"Platform/precision matrix: {[s[0] for s in specs]}")
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
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
