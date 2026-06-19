#!/usr/bin/env python
"""
Verify the "fake PD" (spurious non-positive-definite Hessian) issue and validate
the f64-assembly fix for it -- isolated from the minimizer.

gridHessian.cu computes the triquintic Hessian with its OWN inline fp32 216-term
coefficient solve (separate from the force path). The hypothesis: that fp32 solve
corrupts the second derivatives enough to (a) disagree with the true (f64) Hessian
on positive-definiteness, and (b) be discontinuous across cell faces -- which is
what produces "fake" non-PD minima in production.

Method (no minimizer): place N particles at fixed points in a triquintic grid,
read the plugin's per-atom 3x3 grid-Hessian blocks, and compare to the JAX f64
Hessian of the SAME triquintic polynomial (built from the same stored fp32
derivatives via the TRIQUINTIC_COEFFICIENTS matrix). Report:
  * PD-verdict disagreement rate (plugin says non-PD where f64 says PD, or vice
    versa) -- the isolated "fake PD" signal.
  * Hessian entry / eigenvalue error magnitude (f32 vs f64).
  * cross-cell Hessian discontinuity at a shared face.

Run the SAME script before vs after the gridHessian.cu/gridHessianTiled.cu f64
fix to validate.
"""
import os, re, sys, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import openmm as mm
from openmm import Platform, Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridforceplugin as gfp
from benchmark_utils import get_system_paths

COEFF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "platforms/cuda/src/kernels/include/TriquinticCoefficients.cuh")
_POW = np.array([[i % 6, (i // 6) % 6, (i // 36)] for i in range(216)])


def load_matrix():
    txt = open(COEFF).read()
    start = txt.index('{', txt.index('TRIQUINTIC_COEFFICIENTS'))
    end = txt.index('}};', start) + 1  # matrix only; helpers follow it in the header
    body = txt[start:end]
    rows = re.findall(r'\{([^{}]*)\}', body)
    M = np.array([[float(x) for x in r.split(',') if x.strip() != ''] for r in rows])
    assert M.shape == (216, 216)
    return M


def poly_value(a, f):
    px = jnp.stack([f[0] ** p for p in range(6)])
    py = jnp.stack([f[1] ** p for p in range(6)])
    pz = jnp.stack([f[2] ** p for p in range(6)])
    return jnp.sum(a * px[_POW[:, 0]] * py[_POW[:, 1]] * pz[_POW[:, 2]])


def main():
    M = load_matrix(); Mj = jnp.asarray(M)
    hess_fn = jax.hessian(lambda f, a: poly_value(a, f))

    paths = get_system_paths("1g9v")
    rp = AmberPrmtopFile(paths['receptor_prmtop']); rc = AmberInpcrdFile(paths['receptor_inpcrd'])
    pl = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
           p[2].value_in_unit(nanometer)) for p in rc.positions]
    cen = np.mean(np.array(pl), 0); sp = 0.1; n = 24
    origin = cen - np.array([n, n, n]) * sp / 2.0
    ox, oy, oz = map(float, origin); N = n * n * n
    plat = Platform.getPlatformByName('CUDA')

    with tempfile.TemporaryDirectory() as tmp:
        gf = os.path.join(tmp, "charge.grid")
        gsys = rp.createSystem(nonbondedMethod=NoCutoff)
        gen = gfp.GridForce()
        gen.setGridOrigin(ox, oy, oz); gen.addGridCounts(n, n, n); gen.addGridSpacing(sp, sp, sp)
        gen.setAutoGenerateGrid(True); gen.setGridType('charge'); gen.setComputeDerivatives(True)
        gen.setGridCap(1e30); gen.setReceptorAtoms(list(range(rp.topology.getNumAtoms())))
        gen.setReceptorPositionsFromLists(pl); gen.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gsys.addForce(gen)
        gctx = Context(gsys, VerletIntegrator(0.001), plat)
        gctx.setPositions(rc.positions); gctx.getState(getEnergy=True)
        gen.saveToFile(gf); del gctx

        gridf = gfp.GridForce(); gridf.loadFromFile(gf); gridf.setInterpolationMethod(3)
        derivs = np.array(gridf.getDerivatives(), dtype=np.float64)
        assert derivs.size == 27 * N

        def deriv_at(d, ix, iy, iz):
            return derivs[d * N + (ix * n * n + iy * n + iz)]

        def build_a(bx, by, bz):
            X = np.empty(216)
            for d in range(27):
                for c in range(8):
                    cx, cy, cz = c & 1, (c >> 1) & 1, (c >> 2) & 1
                    X[d * 8 + c] = deriv_at(d, bx + cx, by + cy, bz + cz)
            return 0.125 * (Mj @ jnp.asarray(X))

        # N_P particles at fixed interior points.
        N_P = 1200
        rng = np.random.RandomState(1)
        lo = origin + 2 * sp
        hi = origin + (n - 3) * sp
        pos = lo + rng.rand(N_P, 3) * (hi - lo)

        esys = mm.System()
        for _ in range(N_P):
            esys.addParticle(1.0)
        gridf.addParticleGroup('charge', list(range(N_P)), [1.0] * N_P)
        esys.addForce(gridf)
        ctx = Context(esys, VerletIntegrator(0.001), plat)
        ctx.setPositions(pos * nanometer)
        ctx.getState(getEnergy=True)

        # plugin per-atom 3x3 grid-Hessian blocks (fp32 assembly)
        gridf.computeHessian(ctx)
        Hplug = np.array(gridf.getHessianMatrices(ctx))  # (N_P, 3, 3), kJ/mol/nm^2

        # f64 reference Hessian for each particle from its cell
        tol = 1e-6
        flips = 0          # PD-verdict disagreement (the fake-PD signal)
        f64_pd_f32_not = 0
        max_entry_err = 0.0
        eig_errs = []
        sign_pd_pairs = []
        for p in range(N_P):
            rel = (pos[p] - origin) / sp
            base = np.floor(rel).astype(int)
            frac = rel - base
            a = build_a(int(base[0]), int(base[1]), int(base[2]))
            Hf = np.array(hess_fn(jnp.asarray(frac), a)) / (sp * sp)  # physical d2V/dx2
            Hf = 0.5 * (Hf + Hf.T)
            Hp = 0.5 * (Hplug[p] + Hplug[p].T)
            max_entry_err = max(max_entry_err, np.abs(Hp - Hf).max())
            ef = np.linalg.eigvalsh(Hf); ep = np.linalg.eigvalsh(Hp)
            eig_errs.append(np.abs(ep - ef).max())
            pd_f = ef.min() > tol; pd_p = ep.min() > tol
            if pd_f != pd_p:
                flips += 1
                if pd_f and not pd_p:
                    f64_pd_f32_not += 1

        eig_errs = np.array(eig_errs)
        print(f"=== HESSIAN PRECISION ({N_P} points, grid {n}^3) ===")
        print(f"  max |Hessian entry error| (f32 plugin vs f64) : {max_entry_err:.3e} kJ/mol/nm^2")
        print(f"  eigenvalue error: median={np.median(eig_errs):.3e} max={eig_errs.max():.3e}")
        print(f"  PD-verdict DISAGREEMENTS (fake-PD signal)     : {flips}/{N_P} "
              f"({100*flips/N_P:.1f}%)")
        print(f"    of which f64=PD but plugin=non-PD           : {f64_pd_f32_not}")

        # cross-cell Hessian discontinuity (straddle a face, isolate jump from trend)
        bx, by, bz = n // 2, n // 2, n // 2
        xb = ox + (bx + 1) * sp; yb = oy + (by + 0.6) * sp; zb = oz + (bz + 0.5) * sp
        dlt = 1e-5

        def plug_hess_at(x):
            c2 = Context(esys, VerletIntegrator(0.001), plat)
            p2 = pos.copy(); p2[0] = [x, yb, zb]
            c2.setPositions(p2 * nanometer); c2.getState(getEnergy=True)
            gridf.computeHessian(c2)
            H = np.array(gridf.getHessianMatrices(c2))[0]
            del c2
            return 0.5 * (H + H.T)

        H1 = plug_hess_at(xb - 2 * dlt); H2 = plug_hess_at(xb - dlt); H3 = plug_hess_at(xb + dlt)
        slope = (H2 - H1) / dlt
        disc = H3 - (H2 + slope * 2 * dlt)
        print(f"  cross-cell Hessian discontinuity |max|        : {np.abs(disc).max():.3e} kJ/mol/nm^2")
        del ctx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
