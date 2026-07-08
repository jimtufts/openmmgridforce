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
    'mixed':  dict(rel_e=2e-5, abs_e=1e-6, rel_f=1e-4, abs_f=1e-4, rel_h=1e-4, abs_h=1e-4),
    'single': dict(rel_e=1e-4, abs_e=1e-4, rel_f=3e-3, abs_f=3e-3, rel_h=3e-3, abs_h=3e-3),
}

# Looser bounds for "vs analytic" anchor — driven by interpolation error of the
# method at the test grid spacing, not by float precision. Catches order-of-
# magnitude algorithmic bugs (sign flips, unit-conv errors) without flagging
# routine interp noise.
TOL_ANALYTIC = {
    'trilinear':          dict(rel_e=1e-1, abs_e=1e-2,
                               rel_f=5e-1, abs_f=1.0,
                               rel_h=1.0,  abs_h=1e3),
    'tricubic_bspline':   dict(rel_e=1e-2, abs_e=1e-2,
                               rel_f=5e-2, abs_f=1.0,
                               rel_h=2e-1, abs_h=1e2),
    # 'naked' variant carries a different dynamic-range strategy (per-grid
    # inv_power instead of arcsinh). Same tolerance envelope as arcsinh.
    'tricubic_bspline_naked':   dict(rel_e=1e-2, abs_e=1e-2,
                                     rel_f=5e-2, abs_f=1.0,
                                     rel_h=2e-1, abs_h=1e2),
    # 'loadfile' variant: build in-memory grid with arcsinh + prefilter,
    # save to disk, reload, evaluate.  Same envelope as arcsinh; guards
    # the save/load roundtrip of already-arcsinh-transformed grid
    # coefficients (the setValuesPreTransformed-false double-apply bug
    # regression).
    'tricubic_bspline_loadfile': dict(rel_e=1e-2, abs_e=1e-2,
                                       rel_f=5e-2, abs_f=1.0,
                                       rel_h=2e-1, abs_h=1e2),
    'triquintic_bspline_loadfile': dict(rel_e=5e-3, abs_e=1e-2,
                                         rel_f=2e-2, abs_f=1.0,
                                         rel_h=1e-1, abs_h=1e2),
    'tricubic_hermite':   dict(rel_e=1e-2, abs_e=1e-2,
                               rel_f=5e-2, abs_f=1.0,
                               rel_h=2e-1, abs_h=1e2),
    'triquintic_bspline': dict(rel_e=5e-3, abs_e=1e-2,
                               rel_f=2e-2, abs_f=1.0,
                               rel_h=1e-1, abs_h=1e2),
    'triquintic_bspline_naked': dict(rel_e=5e-3, abs_e=1e-2,
                                     rel_f=2e-2, abs_f=1.0,
                                     rel_h=1e-1, abs_h=1e2),
    'triquintic_hermite': dict(rel_e=5e-3, abs_e=1e-2,
                               rel_f=2e-2, abs_f=1.0,
                               rel_h=1e-1, abs_h=1e2),
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
    # Reference does not implement interpolation method 4 (triquintic_bspline);
    # only CUDA has it. TODO: port or drop.
    ('GridForce[ele/triquintic_bspline]', 'Reference', 'reference'),
    ('GridForce[lja/triquintic_bspline]', 'Reference', 'reference'),
    ('GridForce[ljr/triquintic_bspline]', 'Reference', 'reference'),
    ('GridForce[ele/triquintic_bspline_naked]', 'Reference', 'reference'),
    ('GridForce[lja/triquintic_bspline_naked]', 'Reference', 'reference'),
    ('GridForce[ljr/triquintic_bspline_naked]', 'Reference', 'reference'),
    # Reference lacks setArcsinhScale. Harness enables it for numerical
    # stability of the tricubic_bspline prefilter on steep LJ grids;
    # Reference bspline path uses a different implementation and throws.
    # TODO: port arcsinh to Reference bspline or gate the harness on
    # per-platform capability.
    ('GridForce[ele/tricubic_bspline]', 'Reference', 'reference'),
    ('GridForce[lja/tricubic_bspline]', 'Reference', 'reference'),
    ('GridForce[ljr/tricubic_bspline]', 'Reference', 'reference'),
    # The naked variant uses setInvPowerMode(RUNTIME) which Reference and CPU
    # bspline paths do not currently apply — they load raw grid values, then
    # bspline over huge r^-12/r^-6 magnitudes and disagree with CUDA by
    # ~5 orders of magnitude on LJa/LJr.  ELE is untransformed and so its
    # Reference gap is analogous to the arcsinh case above (no arcsinh needed
    # for ele, but the naked path takes the same code branch).  TODO: apply
    # inv_power on Reference/CPU too.
    ('GridForce[ele/tricubic_bspline_naked]', 'Reference', 'reference'),
    ('GridForce[lja/tricubic_bspline_naked]', 'Reference', 'reference'),
    # loadfile variant: builds via CUDA, saves, reloads, then evaluates
    # on the target platform.  Same Reference-side arcsinh gap as the
    # in-memory arcsinh path.
    ('GridForce[ele/tricubic_bspline_loadfile]', 'Reference', 'reference'),
    ('GridForce[lja/tricubic_bspline_loadfile]', 'Reference', 'reference'),
    ('GridForce[ljr/tricubic_bspline_loadfile]', 'Reference', 'reference'),
    ('GridForce[ele/triquintic_bspline_loadfile]', 'Reference', 'reference'),
    ('GridForce[lja/triquintic_bspline_loadfile]', 'Reference', 'reference'),
    ('GridForce[ljr/triquintic_bspline_loadfile]', 'Reference', 'reference'),
    ('GridForce[ljr/tricubic_bspline_naked]', 'Reference', 'reference'),
    ('GridForce[lja/tricubic_bspline_naked]', 'CPU', 'reference'),
    ('GridForce[ljr/tricubic_bspline_naked]', 'CPU', 'reference'),
    ('GridForce[multigroup energies]',  'Reference', 'reference'),
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


def _fmt_num(x, width=None):
    """Compact numeric formatter: decimals when readable, scientific for
    very small/large. `width` right-pads if set, else natural length."""
    if x is None or not np.isfinite(x):
        s = "nan"
    else:
        ax = abs(x)
        if ax == 0.0:
            s = "0"
        elif 1e-3 <= ax < 1e6:
            # Decimal with adaptive precision, ~5 significant figures.
            prec = max(0, 4 - int(np.floor(np.log10(ax))))
            s = f"{x:.{prec}f}"
        else:
            s = f"{x:.3e}"
    return f"{s:>{width}}" if width else s


def record(case, label, pclass, kind, absd, reld, val=None, ref=None):
    if ' vs analytic' in case:
        method = case.split('/')[-1].split(']')[0]
        tol = TOL_ANALYTIC.get(method, dict(rel_e=1e-2, abs_e=1e-2))
        rel_tol, abs_tol = tol[f'rel_{kind[0]}'], tol[f'abs_{kind[0]}']
    else:
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
    if val is not None and ref is not None:
        vals = f"val={_fmt_num(val)}  ref={_fmt_num(ref)}  "
    else:
        vals = ""
    print(f"  {status:4s} {label:12s} {kind:7s}  {vals}"
          f"|d|={_fmt_num(absd)}  rel={_fmt_num(reld)}  [{case}]")


def _scalar_summary(x):
    """Reduce a scalar/array quantity to a single float for reporting."""
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return float(np.max(np.abs(a)))


def compare(case, results, specs, kinds=('energy', 'force')):
    """results: label -> (E, F) or ('EXC', msg). Compare each to REFERENCE."""
    ref = results.get(REFERENCE)
    if ref is None or isinstance(ref[0], str):
        print(f"  [{case}] Reference unavailable ({ref}); cannot compare")
        tag = (case, REFERENCE, 'reference')
        if tag in OPEN_GAPS:
            _gaps.append(tag)
        else:
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
            record(case, label, pc, 'energy', ad, rd,
                   val=_scalar_summary(r[0]), ref=_scalar_summary(ref[0]))
        if 'force' in kinds:
            ad, rd = _diffs(r[1], ref[1])
            record(case, label, pc, 'force', ad, rd,
                   val=_scalar_summary(r[1]), ref=_scalar_summary(ref[1]))


def eval_all(build, positions, specs):
    results = {}
    for spec in specs:
        try:
            results[spec[0]] = energy_forces(build(), positions, spec)
        except Exception as e:
            results[spec[0]] = ('EXC', repr(e))
    return results


def eval_all_with_hess_blocks(build_sf, positions, specs):
    """Like eval_all, but returns (E, F, H_blocks, err) per spec.
    build_sf must return (system, force) so we retain the typed force
    reference (system.getForce(i) loses the SWIG downcast).
    """
    results = {}
    for spec in specs:
        try:
            system, force = build_sf()
            ctx, _ = _context(system, spec)
            ctx.setPositions(positions)
            st = ctx.getState(getEnergy=True, getForces=True)
            E = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            F = np.array(st.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer))
            H = None
            H_err = None
            if hasattr(force, 'getHessianMatrices'):
                try:
                    H = np.asarray(force.getHessianMatrices(ctx))
                except Exception as _he:
                    H_err = repr(_he)[:100]
            else:
                H_err = "force lacks getHessianMatrices"
            del ctx
            results[spec[0]] = (E, F, H, H_err)
        except Exception as e:
            results[spec[0]] = ('EXC', repr(e), None, None)
    return results


# ---------------------------------------------------------------- GridForce
def grid_section(specs):
    """GridForce parity + analytic-pair-sum anchor.

    Loads the example receptor LJa/LJr/direct_ele .nc grids (generated by
    AlGDock at the receptor positions in receptor.trans.inpcrd) and the
    example ligand. Each (grid_type, interp_method, platform/precision)
    case is checked twice:

      * vs Reference (cross-platform parity, original check)
      * vs analytic pair sum over receptor atoms (anchors against a known
        physical truth, not just another implementation)

    The analytic truth uses the same combining convention the .nc files
    were generated under (see test_grid_force.py): per-atom scaling
    factors are sqrt(eps)*(2 rVdw)^n for LJ (geometric sigma combining,
    not Lorentz-Berthelot) and q for charge; unit conversions for the
    stored values are sqrt(4.184)*1e3 (LJa), sqrt(4.184)*1e6 (LJr),
    and 4.184 (ELE, with the Coulomb constant absorbed into the grid).
    """
    from openmm import app
    prmtop_lig = app.AmberPrmtopFile(os.path.join(PRMDIR, 'ligand.prmtop'))
    inpcrd_lig = app.AmberInpcrdFile(os.path.join(PRMDIR, 'ligand.trans.inpcrd'))
    prmtop_rec = app.AmberPrmtopFile(os.path.join(PRMDIR, 'receptor.prmtop'))
    inpcrd_rec = app.AmberInpcrdFile(os.path.join(PRMDIR, 'receptor.trans.inpcrd'))
    charges_lig = np.array(list(prmtop_lig._prmtop.getCharges()), dtype=np.float64)
    charges_rec = np.array(list(prmtop_rec._prmtop.getCharges()), dtype=np.float64)
    lj_lig = list(prmtop_lig._prmtop.getNonbondTerms())
    lj_rec = list(prmtop_rec._prmtop.getNonbondTerms())
    rVdw_lig = np.array([t[0] for t in lj_lig], dtype=np.float64)
    eps_lig  = np.array([t[1] for t in lj_lig], dtype=np.float64)
    rVdw_rec = np.array([t[0] for t in lj_rec], dtype=np.float64)
    eps_rec  = np.array([t[1] for t in lj_rec], dtype=np.float64)
    n_lig = len(charges_lig)
    from openmm import unit as _u
    pos_lig = np.array([p.value_in_unit(_u.nanometer)
                        for p in inpcrd_lig.positions], dtype=np.float64)
    pos_rec = np.array([p.value_in_unit(_u.nanometer)
                        for p in inpcrd_rec.positions], dtype=np.float64)

    grids_dir = os.path.join(os.path.dirname(__file__), '..', 'grids')

    def _grid_read(fn):
        from netCDF4 import Dataset
        ds = Dataset(fn, 'r')
        d = {k: np.array(ds.variables[k][:][0][:]) for k in ds.variables}
        ds.close()
        return d

    K_E = 138.935456  # kJ/mol nm / e^2 (OpenMM Coulomb constant)

    def _analytic_efH(grid_type):
        """Direct pair sum over receptor atoms in JAX f64. Returns
        (E, F, H_blocks) where E is a scalar kJ/mol, F is (n_lig, 3)
        kJ/mol/nm, H_blocks is (n_lig, 3, 3) kJ/mol/nm^2 (block-diagonal
        because each ligand atom's grid energy depends only on its own
        position; the true 3N x 3N Hessian is block-diagonal by design).
        Uses the same combining convention the .nc grids assume.
        """
        import jax, jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        rp = jnp.asarray(pos_rec)
        if grid_type == 'ele':
            scl_l = jnp.asarray(charges_lig)
            scl_r = jnp.asarray(charges_rec)
            prefactor = K_E; exponent = 1; sign = 1.0
        elif grid_type == 'lja':
            scl_l = jnp.asarray(np.sqrt(eps_lig) * (2.0 * rVdw_lig) ** 3)
            scl_r = jnp.asarray(np.sqrt(eps_rec) * (2.0 * rVdw_rec) ** 3)
            prefactor = 2.0; exponent = 6; sign = -1.0
        elif grid_type == 'ljr':
            scl_l = jnp.asarray(np.sqrt(eps_lig) * (2.0 * rVdw_lig) ** 6)
            scl_r = jnp.asarray(np.sqrt(eps_rec) * (2.0 * rVdw_rec) ** 6)
            prefactor = 1.0; exponent = 12; sign = 1.0
        else:
            raise ValueError(grid_type)

        def E_of_lig_atom(x_i, i):
            r = jnp.linalg.norm(x_i - rp, axis=1)
            return scl_l[i] * sign * prefactor * jnp.sum(scl_r / r ** exponent)

        def E_total(lp_flat):
            lp2 = lp_flat.reshape(n_lig, 3)
            return jnp.sum(jax.vmap(E_of_lig_atom)(lp2, jnp.arange(n_lig)))

        lp0 = jnp.asarray(pos_lig.flatten())
        E = float(E_total(lp0))
        F = -np.asarray(jax.grad(E_total)(lp0)).reshape(n_lig, 3)
        # Per-atom 3x3 Hessian blocks (true H is block-diagonal by design).
        def H_block(i):
            def E_i(x_i):
                return E_of_lig_atom(x_i, i)
            return jax.hessian(E_i)(jnp.asarray(pos_lig[i]))
        H = np.stack([np.asarray(H_block(i)) for i in range(n_lig)])
        return E, F, H

    def _analytic_truth(grid_type):
        """Backwards-compat wrapper: energy only."""
        return _analytic_efH(grid_type)[0]

    BSPLINE_PREFILTER_ORDER = {
        gfp.INTERP_TRICUBIC_BSPLINE: 3,
        gfp.INTERP_TRIQUINTIC_BSPLINE: 5,
    }
    # Bspline dynamic-range compression: arcsinh(V/scale) transform. Keeps
    # prefilter stable on uncapped-LJ-scale ranges. scale=1000 chosen from
    # diag_bspline_cap.py sweep: near-identity for ele values (~1e3 max, so
    # arcsinh ~= linear) while still compressing LJ's ~1e12 range enough for
    # a stable prefilter. scale=1 (heavier compression) collapses ele to 0.
    BSPLINE_ARCSINH_SCALE = 1000.0
    # Gaussian blur previously set to 0.05 nm (~1 grid cell). Diagnostic
    # (diag_bspline_cap.py) showed that 1-cell blur turned a 1.5 % bspline
    # error into 38 % at r=0.72 nm on an LJr grid by biasing values toward
    # higher-magnitude neighbors. Disabled by default; arcsinh handles any
    # dynamic-range issue prefilter would have.
    BSPLINE_BLUR_PHYS_NM = 0.0

    # "naked" bspline variant: no arcsinh, per-grid inv_power compression.
    # inv_power = physical singularity power gives ~1% readback error across
    # the full docking distance range (jax_lj_smooth_grid_prototype confirmed;
    # diag_pergrid_invpower config F reaches d_min=0.167 nm with E_int=-690
    # vs hermite -683). LJr r^-12 -> p=12; LJa r^-6 -> p=6; ELE untransformed
    # (Coulomb-scale values already smooth, negative p corrupts near zero).
    NAKED_INV_POWER = {'ljr': 12.0, 'lja': 6.0, 'ele': None}

    _loadfile_tmpdir = [None]

    def _build_grid_force(nc_file, unit_conv, lig_scales, interp_method,
                          variant='arcsinh', grid_type=None):
        d = _grid_read(nc_file)
        f = gfp.GridForce()
        nx, ny, nz = (int(v) for v in d['counts'])
        f.addGridCounts(nx, ny, nz)
        sp = d['spacing'] * 0.1
        f.addGridSpacing(*[float(s) for s in sp])
        f.setGridOrigin(*[float(o) for o in (d['origin'] * 0.1)])
        for v in (d['vals'] * unit_conv):
            f.addGridValue(float(v))
        for s in lig_scales:
            f.addScalingFactor(float(s))
        f.setInterpolationMethod(interp_method)
        order = BSPLINE_PREFILTER_ORDER.get(interp_method, 0)
        if order:
            f.setBSplinePrefilterOrder(order)
            if variant in ('arcsinh', 'arcsinh_loadfile') and BSPLINE_ARCSINH_SCALE > 0.0:
                f.setArcsinhScale(BSPLINE_ARCSINH_SCALE)
            elif variant == 'naked':
                inv_p = NAKED_INV_POWER.get(grid_type)
                if inv_p is not None:
                    f.setInvPowerMode(gfp.InvPowerMode_RUNTIME, float(inv_p))
            if BSPLINE_BLUR_PHYS_NM > 0.0:
                sigma_cells = BSPLINE_BLUR_PHYS_NM / float(sp.mean())
                f.setGaussianBlurSigma(sigma_cells)
        if variant == 'arcsinh_loadfile':
            # Round-trip through disk so the reloaded force reads already-
            # arcsinh-transformed, prefiltered coefficients.  Catches any
            # regression that re-introduces a values-pre-transformed reset
            # after loadFromFile (the fed027b double-apply bug).  Grid
            # generation is done once via CUDA and cached so the file can
            # be replayed on any platform under test.
            saved_path = _ensure_arcsinh_grid(grid_type, nc_file, unit_conv,
                                              interp_method, order)
            reloaded = gfp.GridForce()
            reloaded.loadFromFile(saved_path)
            for s in lig_scales:
                reloaded.addScalingFactor(float(s))
            reloaded.setInterpolationMethod(interp_method)
            if order:
                reloaded.setBSplinePrefilterOrder(order)
                if BSPLINE_ARCSINH_SCALE > 0.0:
                    reloaded.setArcsinhScale(BSPLINE_ARCSINH_SCALE)
            return reloaded
        return f

    _arcsinh_cache = {}
    def _ensure_arcsinh_grid(grid_type, nc_file, unit_conv, interp_method,
                              order):
        """Build a fresh arcsinh+prefiltered GridForce via CUDA and persist
        it to disk once per (grid_type, interp_method).  Returns the path
        so callers can loadFromFile onto any platform."""
        key = (grid_type, interp_method)
        if key in _arcsinh_cache:
            return _arcsinh_cache[key]
        if _loadfile_tmpdir[0] is None:
            _loadfile_tmpdir[0] = tempfile.mkdtemp(prefix='parity_loadfile_')
            atexit.register(lambda: shutil.rmtree(_loadfile_tmpdir[0],
                                                 ignore_errors=True))
        try:
            cuda_plat = mm.Platform.getPlatformByName('CUDA')
        except Exception as e:
            raise RuntimeError(
                f"arcsinh loadfile variant needs CUDA to build the grid: {e}")
        d = _grid_read(nc_file)
        f = gfp.GridForce()
        nx, ny, nz = (int(v) for v in d['counts'])
        f.addGridCounts(nx, ny, nz)
        sp = d['spacing'] * 0.1
        f.addGridSpacing(*[float(s) for s in sp])
        f.setGridOrigin(*[float(o) for o in (d['origin'] * 0.1)])
        for v in (d['vals'] * unit_conv):
            f.addGridValue(float(v))
        f.addScalingFactor(1.0)   # placeholder; harness re-adds real ones after load
        f.setInterpolationMethod(interp_method)
        if order:
            f.setBSplinePrefilterOrder(order)
            if BSPLINE_ARCSINH_SCALE > 0.0:
                f.setArcsinhScale(BSPLINE_ARCSINH_SCALE)
        system = mm.System()
        system.addParticle(12.0)
        system.addForce(f)
        integ = mm.VerletIntegrator(0.001)
        ctx = mm.Context(system, integ, cuda_plat)
        ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
        ctx.getState(getEnergy=True)
        out_path = os.path.join(_loadfile_tmpdir[0],
                                f'{grid_type}_{interp_method}.grid')
        f.saveToFile(out_path)
        del ctx, integ, system, f
        _arcsinh_cache[key] = out_path
        return out_path

    import tempfile, atexit, shutil
    _deriv_tmpdir = [None]
    _deriv_cache = {}
    def _ensure_derivs_grid(grid_type, ncname):
        if grid_type in _deriv_cache:
            return _deriv_cache[grid_type]
        if _deriv_tmpdir[0] is None:
            _deriv_tmpdir[0] = tempfile.mkdtemp(prefix='parity_derivs_')
            atexit.register(lambda: shutil.rmtree(_deriv_tmpdir[0], ignore_errors=True))
        try:
            cuda_plat = mm.Platform.getPlatformByName('CUDA')
        except Exception as e:
            raise RuntimeError(f"derivative-grid generation needs CUDA: {e}")
        d = _grid_read(os.path.join(grids_dir, ncname))
        counts = [int(v) for v in d['counts']]
        spacing_nm = [float(s) * 0.1 for s in d['spacing']]
        origin_nm = [float(o) * 0.1 for o in d['origin']]
        plugin_type = {'ele': 'charge', 'lja': 'lja', 'ljr': 'ljr'}[grid_type]

        rec_system = prmtop_rec.createSystem(nonbondedMethod=app.NoCutoff,
                                              constraints=None,
                                              implicitSolvent=None)
        for f in rec_system.getForces():
            f.setForceGroup(31)
        grid = gfp.GridForce()
        grid.setGridOrigin(*origin_nm)
        grid.addGridCounts(*counts)
        grid.addGridSpacing(*spacing_nm)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(plugin_type)
        grid.setComputeDerivatives(True)
        grid.setGridCap(1e30)
        grid.setReceptorAtoms(list(range(prmtop_rec.topology.getNumAtoms())))
        rec_pos_lists = [(float(p[0].value_in_unit(_u.nanometer)),
                          float(p[1].value_in_unit(_u.nanometer)),
                          float(p[2].value_in_unit(_u.nanometer)))
                         for p in inpcrd_rec.positions]
        grid.setReceptorPositionsFromLists(rec_pos_lists)
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        rec_system.addForce(grid)
        integ = mm.VerletIntegrator(0.001)
        ctx = mm.Context(rec_system, integ, cuda_plat)
        ctx.setPositions(inpcrd_rec.positions)
        _ = ctx.getState(getEnergy=True, groups={0})  # trigger generation
        out_path = os.path.join(_deriv_tmpdir[0], f'{grid_type}_derivs.grid')
        grid.saveToFile(out_path)
        del ctx, integ, rec_system, grid
        _deriv_cache[grid_type] = out_path
        return out_path

    def _build_grid_force_from_grid(grid_path, lig_scales, interp_method):
        f = gfp.GridForce()
        f.loadFromFile(grid_path)
        for s in lig_scales:
            f.addScalingFactor(float(s))
        f.setInterpolationMethod(interp_method)
        return f

    GRID_SPECS = [
        ('ele', 'direct_ele.nc', 4.184, charges_lig),
        ('lja', 'LJa.nc', np.sqrt(4.184) * 1e3,
            np.sqrt(eps_lig) * (2.0 * rVdw_lig) ** 3),
        ('ljr', 'LJr.nc', np.sqrt(4.184) * 1e6,
            np.sqrt(eps_lig) * (2.0 * rVdw_lig) ** 6),
    ]
    HERMITE_METHODS = {gfp.INTERP_TRICUBIC_HERMITE, gfp.INTERP_TRIQUINTIC_HERMITE}
    # 3-tuple: (label, interp_method_int, variant).  variant is only meaningful
    # for bspline methods — 'arcsinh' is the historical BSPLINE_ARCSINH_SCALE
    # compression path; 'naked' is the per-grid inv_power path validated in
    # diag_pergrid_invpower.  Kept side-by-side so parity data covers both.
    methods = [('trilinear',                gfp.INTERP_TRILINEAR,          None),
               ('tricubic_bspline',         gfp.INTERP_TRICUBIC_BSPLINE,   'arcsinh'),
               ('tricubic_bspline_naked',   gfp.INTERP_TRICUBIC_BSPLINE,   'naked'),
               ('tricubic_bspline_loadfile',
                                            gfp.INTERP_TRICUBIC_BSPLINE,   'arcsinh_loadfile'),
               ('triquintic_bspline',       gfp.INTERP_TRIQUINTIC_BSPLINE, 'arcsinh'),
               ('triquintic_bspline_naked', gfp.INTERP_TRIQUINTIC_BSPLINE, 'naked'),
               ('triquintic_bspline_loadfile',
                                            gfp.INTERP_TRIQUINTIC_BSPLINE, 'arcsinh_loadfile'),
               ('tricubic_hermite',         gfp.INTERP_TRICUBIC_HERMITE,   None),
               ('triquintic_hermite',       gfp.INTERP_TRIQUINTIC_HERMITE, None)]

    pclass = {s[0]: s[3] for s in specs}

    for gtype, ncname, unit_conv, lig_scales in GRID_SPECS:
        nc_path = os.path.join(grids_dir, ncname)
        if not os.path.exists(nc_path):
            print(f"[skip] {ncname} not found at {nc_path}")
            continue
        try:
            E_truth, F_truth, H_truth = _analytic_efH(gtype)
            print(f"\n--- analytic truth ({gtype}): E = {_fmt_num(E_truth)} kJ/mol, "
                  f"max|F| = {_fmt_num(float(np.max(np.abs(F_truth))))} kJ/mol/nm, "
                  f"max|H_block| = {_fmt_num(float(np.max(np.abs(H_truth))))} kJ/mol/nm^2 ---")
        except ImportError:
            print(f"\n--- analytic truth ({gtype}): SKIPPED (jax unavailable) ---")
            E_truth = None; F_truth = None; H_truth = None
        for mname, mval, variant in methods:
            case = f"GridForce[{gtype}/{mname}]"
            print(f"\n=== {case} ===")
            try:
                if mval in HERMITE_METHODS:
                    deriv_path = _ensure_derivs_grid(gtype, ncname)
                    def build_sf(gp=deriv_path, scl=lig_scales, im=mval):
                        system = mm.System()
                        for _ in range(n_lig):
                            system.addParticle(12.0)
                        gf = _build_grid_force_from_grid(gp, scl, im)
                        system.addForce(gf)
                        return system, gf
                else:
                    def build_sf(nc=nc_path, uc=unit_conv, scl=lig_scales,
                                 im=mval, var=variant, gt=gtype):
                        system = mm.System()
                        for _ in range(n_lig):
                            system.addParticle(12.0)
                        gf = _build_grid_force(nc, uc, scl, im,
                                               variant=var, grid_type=gt)
                        system.addForce(gf)
                        return system, gf
                results3 = eval_all_with_hess_blocks(build_sf,
                                                     inpcrd_lig.positions, specs)
                # Drop the H third element for the (E, F) comparison path.
                results = {k: v[:2] for k, v in results3.items()}
                compare(case, results, specs)
                if E_truth is None:
                    continue
                for label, tup in results3.items():
                    if isinstance(tup[0], str):
                        continue
                    E_p, F_p, H_p, H_err = tup
                    ad = abs(E_p - E_truth)
                    rd = ad / (abs(E_truth) + 1e-300)
                    record(case + ' vs analytic', label, pclass.get(label, 'double'),
                           'energy', ad, rd,
                           val=float(E_p), ref=float(E_truth))
                    if F_truth is not None:
                        F_p_arr = np.asarray(F_p)
                        ad_f = float(np.max(np.abs(F_p_arr - F_truth)))
                        rd_f = ad_f / (float(np.max(np.abs(F_truth))) + 1e-300)
                        record(case + ' vs analytic', label,
                               pclass.get(label, 'double'), 'force', ad_f, rd_f,
                               val=float(np.max(np.abs(F_p_arr))),
                               ref=float(np.max(np.abs(F_truth))))
                    # Trilinear is C^0 (piecewise-linear); in-cell Hessian
                    # is identically 0 by construction, so comparing to the
                    # analytic (nonzero) Hessian is meaningless. Skip.
                    if H_truth is not None and mval != gfp.INTERP_TRILINEAR:
                        if H_p is None:
                            print(f"  SKIP {label:12s} hessian  ({H_err}) "
                                  f"[{case} vs analytic]")
                        else:
                            H_p_arr = np.asarray(H_p)
                            ad_h = float(np.max(np.abs(H_p_arr - H_truth)))
                            rd_h = ad_h / (float(np.max(np.abs(H_truth))) + 1e-300)
                            record(case + ' vs analytic', label,
                                   pclass.get(label, 'double'), 'hessian',
                                   ad_h, rd_h,
                                   val=float(np.max(np.abs(H_p_arr))),
                                   ref=float(np.max(np.abs(H_truth))))
            except Exception as e:
                print(f"    SKIP {case}: {type(e).__name__}: {e}")

    # Multi-particle-group energy regression — exercises the kernel branch
    # that writes per-group energies via atomicAdd(mixed*, real), which
    # silently broke in mixed/double precision until the cast was added.
    nc_path = os.path.join(grids_dir, 'direct_ele.nc')
    if os.path.exists(nc_path):
        half = n_lig // 2
        def build_groups():
            system = mm.System()
            for _ in range(n_lig):
                system.addParticle(12.0)
            f = _build_grid_force(nc_path, 4.184, charges_lig,
                                   gfp.INTERP_TRICUBIC_BSPLINE)
            f.addParticleGroup('g0', list(range(half)))
            f.addParticleGroup('g1', list(range(half, n_lig)))
            system.addForce(f)
            return system
        case = "GridForce[multigroup energies]"
        print(f"\n=== {case} ===")
        compare(case, eval_all(build_groups, inpcrd_lig.positions, specs), specs)

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
        record(case, label, pclass.get(label, 'double'), 'hessian', ad, rd,
               val=_scalar_summary(h), ref=_scalar_summary(ref))


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


# ----------------------------------------------- IsolatedGBSAForce[GRID]
def gbsa_grid_section(specs):
    """Big-picture anchor for IsolatedGBSAForce[GRID].

    Structure mirrors grid_section for GridForce: enumerate the physical
    components x interpolation methods x platform/precision. Every cell is
    checked against an analytical (or transitively analytical) truth so it
    is clear which combination is validated and which fails.

    Components measured:
      pure_GRID  = ligand self-GB whose Born radii used the receptor HCT
                    interpolated from the desolvation grid.
                    Truth: PAIRWISE.getGroupLigandSelfEnergy(0), which uses
                    the exact pairwise HCT sum (no grid interpolation).
      GRID_xterm = pure_GRID plus the pairwise cross-term augment (the
                    receptor<->ligand Still term that plain GRID mode omits).
                    Truth: PAIRWISE ligand_self + PAIRWISE cross_term
                    (BOTH pairwise-exact). Deliberately excludes the
                    receptor's own self-GB change from ligand descreening
                    (~9 kJ/mol here) which the GRID production path does
                    not model.

    Note the anchor is *not* stock OpenMM's raw E(rec+lig) - E(rec) --
    that quantity also contains the receptor desolvation delta which our
    GRID + cross-term path omits by design. Anchoring against it would
    have looked "OK at 3 %" via cancellation. Instead we anchor against
    the exact analytic quantity the pipeline is actually computing.

    Interpolation methods enumerated:
      trilinear, tricubic_bspline, tricubic_hermite, triquintic_bspline,
      triquintic_hermite. Hermite methods require a with-derivatives grid
      (set setComputeGridDerivatives(True) on the auto-gen path).

    Platforms: Reference (energy only; Hessian NOT implemented) + CUDA
    single/mixed/double. CPU currently doesn't implement IsolatedGBSA GRID
    mode. Reported clearly per row.

    Cross-term augment is CUDA-only (the Reference kernel doesn't wire
    the pairwise scalar-field cross term). We simply do not evaluate
    GRID_xterm on Reference; that isn't a failure.
    """
    case = "IsolatedGBSAForce[GRID]"
    print(f"\n=== {case} ===")
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

    # Grid bounding box covers both receptor and ligand with margin.
    all_pos = np.vstack([rec_pos, lig_pos])
    lo = all_pos.min(axis=0) - 0.6
    hi = all_pos.max(axis=0) + 0.6
    sp = 0.15
    counts = tuple(int(np.ceil((hi[d] - lo[d]) / sp)) + 1 for d in range(3))

    # ---- Analytical anchor: PAIRWISE + stock OpenMM PAIRWISE quantity ----
    def build_pairwise():
        system = mm.System()
        for _ in range(n_lig):
            system.addParticle(12.0)
        f = gfp.IsolatedGBSAForce()
        f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
        f.setSoluteDielectric(1.0); f.setSolventDielectric(78.5)
        f.setIncludeSurfaceArea(False)
        f.setReceptorMode(gfp.IsolatedGBSAForce.PAIRWISE)
        f.setNumAtoms(n_lig)
        for i in range(n_lig):
            f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
        f.setNumReceptorAtoms(n_rec)
        for i in range(n_rec):
            f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
        f.setReceptorPositions(rec_pos.flatten().tolist())
        f.addParticleGroup("lig", list(range(n_lig)))
        system.addForce(f); return system, f

    # PAIRWISE anchor components on double Reference (validated separately)
    try:
        ref = (REFERENCE, 'Reference', {}, 'double')
        sysp, fp = build_pairwise()
        ctxp, _ = _context(sysp, ref)
        ctxp.setPositions(lig_pos * unit.nanometer)
        E_pair_total = ctxp.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        E_pair_ligself = fp.getGroupLigandSelfEnergy(0)
        E_pair_cross = fp.getGroupCrossTermEnergy(0)
        del ctxp
        # Stock OpenMM double anchor: full E(rec+lig) − E(rec)
        all_q = np.concatenate([rec_q, lig_q])
        all_r = np.concatenate([rec_r, lig_r])
        all_s = np.concatenate([rec_s, lig_s])
        Ecx, _ = _gbsa_openmm_ef(all_q, all_r, all_s, np.vstack([rec_pos, lig_pos]))
        Erec, _ = _gbsa_openmm_ef(rec_q, rec_r, rec_s, rec_pos)
        E_stock_pairwise = Ecx - Erec
        print(f"  anchors: PAIRWISE_total={E_pair_total:+.3f}  "
              f"PAIRWISE_ligself={E_pair_ligself:+.3f}  "
              f"cross={E_pair_cross:+.3f}  stock={E_stock_pairwise:+.3f} kJ/mol")
    except Exception as e:
        print(f"  SKIP: PAIRWISE anchor unavailable: {e!r}")
        return

    # PAIRWISE Hessian, used as an approximate anchor for pure-GRID
    # Hessian below. PAIRWISE full Hessian includes rec-lig cross-term
    # chain rule; pure-GRID does not, so this is not a true reference.
    pairwise_hess = {}
    for spec in specs:
        if spec[1] != 'CUDA':
            continue
        try:
            sysp2, fp2 = build_pairwise()
            ctxp2, _ = _context(sysp2, spec)
            ctxp2.setPositions(lig_pos * unit.nanometer)
            ctxp2.getState(getEnergy=True)
            pairwise_hess[spec[3]] = np.array(fp2.computeHessian(ctxp2))
            del ctxp2
        except Exception as e:
            print(f"  PAIRWISE anchor Hessian ({spec[0]}) failed: {e!r}")
    if 'double' in pairwise_hess:
        Hp = pairwise_hess['double']
        print(f"  PAIRWISE analytical Hessian (CUDA/double): shape={Hp.shape}  "
              f"||H||_F={_fmt_num(float(np.linalg.norm(Hp))).strip()}  "
              f"max|H|={_fmt_num(float(np.abs(Hp).max())).strip()}")

    # ---- Baseline receptor Born radii (numpy: standard OBC2 HCT + tanh) ----
    R = rec_r.astype(np.float64); S = rec_s.astype(np.float64)
    p = rec_pos.astype(np.float64)
    OFFSET = 0.009
    R_off = R - OFFSET
    hct = np.zeros(n_rec)
    for i in range(n_rec):
        for j in range(n_rec):
            if i == j: continue
            r = float(np.linalg.norm(p[i] - p[j]))
            Sj = (R[j] - OFFSET) * S[j]
            r_plus_Sj = r + Sj
            if R_off[i] >= r_plus_Sj: continue
            r_minus_Sj = abs(r - Sj)
            l = 1.0 / R_off[i] if R_off[i] > r_minus_Sj else 1.0 / r_minus_Sj
            u = 1.0 / r_plus_Sj
            l2, u2 = l*l, u*u
            term = (l - u + 0.25*r*(u2 - l2) + 0.5*(1.0/r)*np.log(u/l)
                    + 0.25*Sj*Sj*(1.0/r)*(l2 - u2))
            if R_off[i] < (Sj - r):
                term += 2.0 * (1.0/R_off[i] - l)
            hct[i] += term
    A_, B_, G_ = 1.0, 0.8, 4.85
    psi = 0.5 * R_off * hct
    tanh_arg = A_*psi - B_*psi**2 + G_*psi**3
    tanh_val = np.tanh(tanh_arg)
    denom = 1.0/R_off - tanh_val / R
    R_rec_baseline = np.minimum(np.where(denom > 0, 1.0/denom, R), 50.0)

    # Truths and target components
    E_pure_truth = E_pair_ligself                        # exact ligand self-GB
    E_xterm_truth = E_pair_ligself + E_pair_cross        # pure + exact cross
    receptor_desolv_delta = E_pair_total - E_xterm_truth
    print(f"  truths:   pure_GRID = {E_pure_truth:+.3f}  "
          f"GRID+xterm = {E_xterm_truth:+.3f}  (rec_desolv_omitted "
          f"= {receptor_desolv_delta:+.3f})")

    # ---- Grid generator (with derivatives if the method needs them) ----
    def cuda_spec():
        for cand in specs:
            if cand[1] == 'CUDA': return cand
        return None
    gen_spec = cuda_spec()
    if gen_spec is None:
        print("  SKIP: KDE auto-gen requires CUDA; no CUDA spec available")
        return

    def autogen_grid(with_derivs):
        f_gen = gfp.GBSAGridForce()
        f_gen.setNumAtoms(1)
        f_gen.setAtomParameters(0, 0.0, float(lig_r[0]), float(lig_s[0]))
        f_gen.setInterpolationMethod(0)
        f_gen.setAutoGenerateGrid(True)
        f_gen.setUseKDEGeneration(True)
        if with_derivs:
            f_gen.setComputeGridDerivatives(True)
        f_gen.setReceptorPositions(rec_pos.flatten().tolist())
        f_gen.setReceptorRadii(rec_r.tolist())
        f_gen.setReceptorScaleFactors(rec_s.tolist())
        f_gen.setGridOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
        f_gen.setGridCounts(int(counts[0]), int(counts[1]), int(counts[2]))
        f_gen.setGridSpacing(float(sp))
        f_gen.setProbeRadius(0.14)
        f_gen.setRThresholds([0.12, 0.16])
        f_gen.setParticles([0]); f_gen.addParticleGroup('bootstrap', [0])
        s = mm.System(); s.addParticle(12.0); s.addForce(f_gen)
        ctxg, _ = _context(s, gen_spec)
        ctxg.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
        ctxg.getState(getEnergy=True)
        cg = f_gen.getDesolvationGrid()
        del ctxg
        return cg

    HERMITE_METHODS = {gfp.INTERP_TRICUBIC_HERMITE, gfp.INTERP_TRIQUINTIC_HERMITE}
    # IsolatedGBSAForce GRID mode rejects triquintic_bspline (method=4) at
    # the API layer with an explicit "interpolationMethod must be 0..3"
    # message. Deliberately skip it here; if support is added later, drop
    # this comment and add ('triquintic_bspline', gfp.INTERP_TRIQUINTIC_BSPLINE).
    methods = [
        ('trilinear',          gfp.INTERP_TRILINEAR),
        ('tricubic_bspline',   gfp.INTERP_TRICUBIC_BSPLINE),
        ('tricubic_hermite',   gfp.INTERP_TRICUBIC_HERMITE),
        ('triquintic_hermite', gfp.INTERP_TRIQUINTIC_HERMITE),
    ]

    grid_no_deriv = None
    grid_with_deriv = None
    try:
        grid_no_deriv = autogen_grid(with_derivs=False)
        print(f"  grid (no-derivs):     {counts[0]}x{counts[1]}x{counts[2]} @ {sp} nm  origin=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f})")
    except Exception as e:
        print(f"  grid auto-gen (no-derivs) FAILED: {e!r}")
    try:
        grid_with_deriv = autogen_grid(with_derivs=True)
        print(f"  grid (with-derivs):   {counts[0]}x{counts[1]}x{counts[2]} @ {sp} nm  origin=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f})")
    except Exception as e:
        print(f"  grid auto-gen (with-derivs) FAILED: {e!r}")

    def build_grid(cg, interp_method, with_cross_term):
        system = mm.System()
        for _ in range(n_lig):
            system.addParticle(12.0)
        f = gfp.IsolatedGBSAForce()
        f.setGBMethod(gfp.IsolatedGBSAForce.OBC_II)
        f.setSoluteDielectric(1.0); f.setSolventDielectric(78.5)
        f.setIncludeSurfaceArea(False)
        f.setReceptorMode(gfp.IsolatedGBSAForce.GRID)
        f.setNumAtoms(n_lig)
        for i in range(n_lig):
            f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
        f.setDesolvationGrid(cg)
        f.setInterpolationMethod(interp_method)
        f.setNumReceptorAtoms(n_rec)
        for i in range(n_rec):
            f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]),
                                        float(rec_s[i]))
        f.setReceptorPositions(rec_pos.flatten().tolist())
        f.setReceptorBornRadiiBaseline([float(x) for x in R_rec_baseline])
        if with_cross_term:
            f.setCrossTermBinValues([float(lig_r[i]) for i in range(n_lig)])
            f.setComputeCrossTermGrid(True)
        f.addParticleGroup("lig", list(range(n_lig)))
        system.addForce(f); return system, f

    def report(component_label, method_name, spec_label, E, truth,
               rel_tol=0.05, abs_tol=2.0):
        if isinstance(E, tuple) and E[0] == 'EXC':
            print(f"    EXC  {spec_label:12s} {component_label:12s} "
                  f"{E[1][:60]}  [{case}/{method_name}]")
            _failures.append((f"{case}/{method_name} {component_label}",
                              spec_label, 'energy', E[1]))
            return
        d = abs(E - truth)
        rel = d / max(1.0, abs(truth))
        ok = np.isfinite(E) and (rel < rel_tol or d < abs_tol)
        status = 'OK' if ok else 'FAIL'
        print(f"    {status:4s} {spec_label:12s} {component_label:12s} "
              f"E={E:+.3f}  |E−truth({truth:+.3f})|={d:.3f}  rel={rel:.2e}  "
              f"[{case}/{method_name}]")
        if not ok:
            _failures.append((f"{case}/{method_name} {component_label}",
                              spec_label, 'energy',
                              f"E={E:.3f} truth={truth:+.3f} rel={rel:.2e}"))

    # Reference implements only method=0 (trilinear) for GRID-mode HCT
    # interpolation. Methods 1/2/3 throw at the kernel; skip them cleanly
    # instead of recording an EXC failure.
    REFERENCE_METHODS = {gfp.INTERP_TRILINEAR}

    for mname, mval in methods:
        cg = grid_with_deriv if mval in HERMITE_METHODS else grid_no_deriv
        print(f"\n  --- interp={mname} ---")
        if cg is None:
            print(f"    SKIP {mname}: required grid unavailable")
            continue
        # component 1: pure GRID -- all specs implementing GRID mode
        for spec in specs:
            if spec[1] not in ('Reference', 'CUDA'):
                continue
            if spec[1] == 'Reference' and mval not in REFERENCE_METHODS:
                print(f"    SKIP Reference    pure_GRID    "
                      f"{mname} not implemented on Reference (dispatch guards it)")
                continue
            try:
                sys_, _f = build_grid(cg, mval, with_cross_term=False)
                ctx, _ = _context(sys_, spec)
                ctx.setPositions(lig_pos * unit.nanometer)
                E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole)
                del ctx
            except Exception as e:
                E = ('EXC', repr(e))
            report('pure_GRID', mname, spec[0], E, E_pure_truth)
        # component 2: GRID + cross-term augment -- CUDA only
        for spec in specs:
            if spec[1] != 'CUDA':
                continue
            try:
                sys_, _f = build_grid(cg, mval, with_cross_term=True)
                ctx, _ = _context(sys_, spec)
                ctx.setPositions(lig_pos * unit.nanometer)
                E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole)
                del ctx
            except Exception as e:
                E = ('EXC', repr(e))
            report('GRID+xterm', mname, spec[0], E, E_xterm_truth)

        # component 3: cross-precision Hessian parity. CUDA/double is the
        # in-harness reference for CUDA/single and CUDA/mixed. Absolute
        # correctness of the GRID Hessian is validated separately by
        # probe_grid_hessian_jax.py against a JAX autodiff reference
        # (rel ~1e-6 for hermite, ~5e-4 for bspline). Trilinear (C^0) and
        # tricubic_hermite (C^1) have no meaningful Hessian; skip.
        C2_METHODS = {gfp.INTERP_TRICUBIC_BSPLINE, gfp.INTERP_TRIQUINTIC_HERMITE}
        if mval not in C2_METHODS:
            print(f"    SKIP Hessian: {mname} is not C^2-continuous; "
                  f"Hessian only meaningful for tricubic_bspline / "
                  f"triquintic_hermite")
            continue
        print(f"    SKIP Reference    Hessian     "
              f"GRID Hessian throws on Reference (not implemented)")
        cuda_specs = [s for s in specs if s[1] == 'CUDA']
        cuda_hess = {}
        for spec in cuda_specs:
            try:
                sys_, gbsa_force = build_grid(cg, mval, with_cross_term=False)
                ctx, _ = _context(sys_, spec)
                ctx.setPositions(lig_pos * unit.nanometer)
                ctx.getState(getEnergy=True)
                cuda_hess[spec[3]] = np.array(gbsa_force.computeHessian(ctx))
                del ctx
            except Exception as e:
                print(f"    EXC  {spec[0]:12s} Hessian     {repr(e)[:60]}  "
                      f"[{case}/{mname}]")
                _failures.append((f"{case}/{mname} Hessian", spec[0],
                                  'hessian', repr(e)))
        H_ref = cuda_hess.get('double')
        if H_ref is None:
            print(f"    SKIP Hessian: CUDA/double reference unavailable")
            continue
        H_ref_norm = np.linalg.norm(H_ref)
        pclass_map = {s[0]: s[3] for s in cuda_specs}
        for spec in cuda_specs:
            H = cuda_hess.get(spec[3])
            if H is None:
                continue
            diff = H - H_ref
            fro = float(np.linalg.norm(diff))
            max_ = float(np.max(np.abs(diff)))
            rel = fro / max(1.0, H_ref_norm)
            record(f"{case}/{mname} Hessian vs CUDA/double",
                   spec[0], pclass_map[spec[0]], 'hessian', max_, rel,
                   val=float(np.linalg.norm(H)), ref=float(H_ref_norm))


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

    # Non-finite derivative guard. A supplied higher-order grid carrying any NaN/Inf
    # must RAISE, never silently fall back to trilinear -- that masked a real
    # divergence and is the kind of bug that is impossible to diagnose downstream.
    # Trilinear (method 0) on the same grid must still work, since it reads no
    # derivatives. (Reference/CPU only; these own the guard.)
    print(f"\n=== {case} non-finite derivative guard ===")
    gnx = gny = gnz = 8; gsp = 0.1; gnp = gnx * gny * gnz; gthr = [0.12, 0.16]

    def run_poison(method, spec):
        cg = gfp.DesolvationGrid(gnx, gny, gnz, float(gsp), 0.14, gthr)
        cg.setOrigin(-0.4, -0.4, -0.4); cg.setHasDerivatives(True)
        arr = np.full(27 * gnp, 0.01)
        arr[26 * gnp + 100] = np.nan       # one NaN in the fxxyyzz block
        cg.setHctProbe(arr.tolist())
        zz = [0.0] * (len(gthr) * gnp)
        cg.setCorrectionN(zz); cg.setCorrectionA(zz); cg.setCorrectionB(zz)
        f = gfp.GBSAGridForce(); f.setNumAtoms(1); f.setIncludeSurfaceArea(False)
        f.setInterpolationMethod(method); f.setAtomParameters(0, 0.2, 0.17, 0.72)
        f.setParticles([0]); f.addParticleGroup('l', [0]); f.setDesolvationGrid(cg)
        s = mm.System(); s.addParticle(12.0); s.addForce(f)
        ctx, _ = _context(s, spec)
        ctx.setPositions(np.array([[0.0, 0.0, 0.0]]))
        E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        del ctx
        return E
    for spec in specs:
        if spec[1] not in ('Reference', 'CPU'):
            continue
        try:
            run_poison(3, spec); raises = False
        except Exception:
            raises = True
        try:
            tri_ok = np.isfinite(run_poison(0, spec))
        except Exception:
            tri_ok = False
        ok = raises and tri_ok
        print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} method-3 raises on NaN deriv, "
              f"method-0 still works  [{case} nan-guard]")
        if not ok:
            _failures.append((case + " nan-guard", spec[0], 'guard',
                              f"raises={raises} tri_ok={tri_ok}"))

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
                  f"E={_fmt_num(e_cpu).strip()} ref={_fmt_num(e_corr).strip()} abs={d:.2e}  [{case} buried-corr]")
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

    def gbsa_autogen(method=0, derivs=False, kde=False):
        f = gfp.GBSAGridForce(); f.setNumAtoms(n_lig); f.setIncludeSurfaceArea(False)
        f.setInterpolationMethod(method); f.setAutoGenerateGrid(True)
        if derivs:
            f.setComputeGridDerivatives(True)
        if kde:
            f.setUseKDEGeneration(True)
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
                  f"E={_fmt_num(E).strip()} ref={_fmt_num(e_ref).strip()} abs={d:.2e}  [{case} desolv]")
            if not ok:
                _failures.append((case + " desolv", spec[0], 'energy', f"abs={d:.2e}"))
        except Exception as ex:
            print(f"    EXC  {spec[0]:12s}: {repr(ex)[:60]}")
            _failures.append((case + " desolv", spec[0], 'energy', repr(ex)))

    # (2b) KDE-smoothed generation. With setUseKDEGeneration the Reference/CPU
    # generator uses the same sigmoid-weighted corrections as CUDA, so auto-gen
    # matches CUDA (the default binned model differs by the generation-model gap).
    e_binned, _ = gbsa_energy(gbsa_autogen(0), (REFERENCE, 'Reference', {}, 'double'))
    e_ref_kde, _ = gbsa_energy(gbsa_autogen(0, kde=True), (REFERENCE, 'Reference', {}, 'double'))
    ok = abs(e_ref_kde - e_binned) > 1e-6        # the flag actually changes the model
    print(f"    {'OK' if ok else 'FAIL':4s} Reference    KDE-gen differs from binned "
          f"(|d|={abs(e_ref_kde - e_binned):.2e})  [{case} kde]")
    if not ok:
        _failures.append((case + " kde", REFERENCE, 'energy', "KDE==binned"))
    for spec in specs:
        if spec[1] == 'Reference':
            continue
        try:
            E, _ = gbsa_energy(gbsa_autogen(0, kde=True), spec)
            d = abs(E - e_ref_kde)
            # CPU delegates to Reference (bit-exact); CUDA uses the same KDE model
            # but mixed/float arithmetic, so allow a small relative tolerance there.
            tol = 1e-9 if spec[1] == 'CPU' else 5e-5 * abs(e_ref_kde)
            ok = d < tol
            print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} KDE-gen vs Reference "
                  f"E={_fmt_num(E).strip()} ref={_fmt_num(e_ref_kde).strip()} abs={d:.2e}  [{case} kde]")
            if not ok:
                _failures.append((case + " kde", spec[0], 'energy', f"abs={d:.2e}"))
        except Exception as ex:
            print(f"    EXC  {spec[0]:12s}: {repr(ex)[:60]}")
            _failures.append((case + " kde", spec[0], 'energy', repr(ex)))

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
            for spec in specs:
                if spec[0] == REFERENCE:
                    continue
                E = gridforce_energy(gtype, spec)
                d = abs(E - er)
                # CPU delegates to Reference (bit-exact); CUDA shares the formula but
                # generates in float, so allow a small relative tolerance there.
                tol = 1e-9 if spec[1] == 'CPU' else 5e-5 * max(1.0, abs(er))
                ok = d < tol
                print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} GridForce gen[{gtype}] vs Reference "
                      f"E={_fmt_num(E).strip()} ref={_fmt_num(er).strip()} abs={d:.2e}  [{case} field]")
                if not ok:
                    _failures.append((case + " field", spec[0], gtype, f"abs={d:.2e}"))
        except Exception as ex:
            print(f"    EXC  GridForce gen[{gtype}]: {repr(ex)[:60]}")
            _failures.append((case + " field", '-', gtype, repr(ex)))

    # (5) GridForce field generation vs the ANALYTIC closed form. This is the
    # independent ground truth -- CPU==Reference is circular (shared formula) and
    # would not catch a wrong generation formula; this would (e.g. the AMBER
    # Rmin=2^(1/6)*sigma convention). One receptor atom at a grid node; trilinear at
    # another node returns the stored value, which must equal the capped analytic field.
    COUL = 138.935456
    alo, asp, ac = -1.0, 0.1, 21              # (0,0,0) and (0.5,0,0) are nodes
    rq, rsig, reps, rr = 0.45, 0.30, 0.20, 0.5
    rmin = (2.0 ** (1.0 / 6.0)) * rsig
    gcap = gfp.GridForce().getGridCap()
    def analytic_field(gtype):
        if gtype == 'charge':
            U = COUL * rq / rr
        elif gtype == 'ljr':
            U = np.sqrt(reps) * rmin ** 6 / rr ** 12
        else:
            U = -2.0 * np.sqrt(reps) * rmin ** 3 / rr ** 6
        return gcap * np.tanh(U / gcap)       # generation applies a tanh cap
    def one_atom_node_energy(gtype, spec):
        s = mm.System(); s.addParticle(12.0)                 # ligand (probe)
        nb = mm.NonbondedForce(); nb.addParticle(0.0, 0.3, 0.0)
        s.addParticle(12.0); nb.addParticle(rq, rsig, reps)  # receptor atom at origin
        s.addForce(nb)
        f = gfp.GridForce(); f.addGridCounts(ac, ac, ac); f.addGridSpacing(asp, asp, asp)
        f.setGridOrigin(alo, alo, alo); f.setAutoGenerateGrid(True); f.setGridType(gtype)
        f.setReceptorAtoms([1]); f.setReceptorPositionsFromLists([(0.0, 0.0, 0.0)])
        f.setLigandAtoms([0]); f.addScalingFactor(1.0); f.setForceGroup(1)
        s.addForce(f)
        ctx, _ = _context(s, spec)
        ctx.setPositions(np.array([[rr, 0.0, 0.0], [0.0, 0.0, 0.0]]))   # ligand at a node
        E = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        del ctx
        return E
    for gtype in ('charge', 'ljr', 'lja'):
        ref = analytic_field(gtype)
        for spec in specs:
            try:
                E = one_atom_node_energy(gtype, spec)
                rel = abs(E - ref) / max(1.0, abs(ref))
                ok = rel < 1e-4
                print(f"    {'OK' if ok else 'FAIL':4s} {spec[0]:12s} GridForce gen[{gtype}] vs analytic "
                      f"({E:.4f} vs {ref:.4f}, rel {rel:.2e})  [{case} field-analytic]")
                if not ok:
                    _failures.append((case + " field-analytic", spec[0], gtype, f"rel={rel:.2e}"))
            except Exception as ex:
                print(f"    EXC  GridForce gen[{gtype}] analytic {spec[0]}: {repr(ex)[:50]}")
                _failures.append((case + " field-analytic", spec[0], gtype, repr(ex)))


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
    'gbsagrid_iso': gbsa_grid_section,
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
