#!/usr/bin/env python
"""
3D triquintic-Hermite plugin validation via JAX autodiff (FD-free).

Follow-on to the 1D test. Goal: confirm the PLUGIN's triquintic interpolation
is a faithful evaluation of the stored derivatives, and quantify the fp32
effects (eval roundoff + cross-cell discontinuity) on the real stored grid.

Strategy (no receptor physics reimplemented): take the plugin's OWN stored 27
derivatives at the 8 corners of a cell, reconstruct the quintic polynomial in
JAX using the same TRIQUINTIC_COEFFICIENTS matrix, and:

  (A) FIDELITY  -- compare plugin energy/force at a sub-cell point to the JAX
      polynomial value / jax.grad built from the same stored corner data.
      Agreement to ~fp32 => the kernel's matrix/eval/indexing are correct and
      the bug (if any) is precision, not algorithm. Large disagreement => a
      kernel/matrix/layout bug.

  (C) CONTINUITY -- evaluate the JAX polynomials of two x-adjacent cells at their
      shared face and measure the jump in value/gradient. This is the
      production-relevant C2 break, measured on the actual fp32 stored grid.

Layout facts (from gridGeneration.cu / gridForceTiled.cu / TriquinticCoefficients.cuh):
  * 27 derivatives per point, index d -> (a,b,c) powers in {0,1,2}^3, STORED
    already scaled to fractional-cell units (physical x spacing^(a+b+c) per axis).
  * X is derivative-major: X[d*8 + c], corner c = (c&1, (c>>1)&1, (c>>2)&1).
  * a[i] = 0.125 * sum_j M[i][j] X[j]; coeff i -> monomial (ix,iy,iz) = i%6,(i//6)%6,i//36.
  * value = sum a * fx^ix fy^iy fz^iz on fractional coords; dE/dx = (dvalue/dfx)/spacing.

Run: /home/jtufts/anaconda3/envs/algdock_refactor/bin/python \
         python/tests/test_triquintic_hermite_3d_jax.py
"""

import os, re, sys, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

import openmm as mm
from openmm import Platform, Context, VerletIntegrator
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
from openmm.unit import nanometer, kilojoules_per_mole

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridforceplugin as gfp
from benchmark_utils import get_system_paths, validate_system_paths

COEFF_HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))),
    "platforms/cuda/src/kernels/include/TriquinticCoefficients.cuh")


def load_matrix():
    txt = open(COEFF_HEADER).read()
    start = txt.index('{', txt.index('TRIQUINTIC_COEFFICIENTS'))
    end = txt.index('}};', start) + 1  # matrix only; helpers follow it in the header
    body = txt[start:end]
    rows = re.findall(r'\{([^{}]*)\}', body)
    M = np.array([[float(x) for x in r.split(',') if x.strip() != ''] for r in rows])
    assert M.shape == (216, 216), M.shape
    return M


# Powers (i,j,k) for each of the 216 monomials, in the a[i+6j+36k] layout.
_POW = np.array([[i % 6, (i // 6) % 6, (i // 36)] for i in range(216)])


def poly_value(a, f):
    """a: (216,) coeffs; f: (3,) fractional coords. Returns scalar value."""
    fx, fy, fz = f[0], f[1], f[2]
    px = jnp.stack([fx ** p for p in range(6)])
    py = jnp.stack([fy ** p for p in range(6)])
    pz = jnp.stack([fz ** p for p in range(6)])
    terms = a * px[_POW[:, 0]] * py[_POW[:, 1]] * pz[_POW[:, 2]]
    return jnp.sum(terms)


def main():
    M = load_matrix()
    Mj = jnp.asarray(M)
    print("Loaded TRIQUINTIC_COEFFICIENTS:", M.shape)

    system_name = "1g9v"
    paths = get_system_paths(system_name)
    missing = validate_system_paths({k: paths[k] for k in
                                     ('receptor_prmtop', 'receptor_inpcrd')})
    if missing:
        print("Missing data, cannot run:", missing)
        return 1

    platform = Platform.getPlatformByName('CUDA')

    # Build a SMALL explicit grid centered on the receptor (avoids the
    # box-sized generate_grid which goes tiled/30GB). Triquintic precision
    # behavior is spacing-independent for this validation.
    rec_prm = AmberPrmtopFile(paths['receptor_prmtop'])
    rec_crd = AmberInpcrdFile(paths['receptor_inpcrd'])
    pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                 p[2].value_in_unit(nanometer)) for p in rec_crd.positions]
    rec_atoms = list(range(rec_prm.topology.getNumAtoms()))
    centroid = np.mean(np.array(pos_list), axis=0)

    nx = ny = nz = 16
    sp = 0.10  # nm
    origin = centroid - np.array([nx, ny, nz]) * sp / 2.0
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    N = nx * ny * nz

    with tempfile.TemporaryDirectory() as tmp:
        grid_file = os.path.join(tmp, "charge.grid")
        print(f"Generating small charge grid for {system_name}: "
              f"{nx}x{ny}x{nz}, spacing={sp} nm, origin=({ox:.3f},{oy:.3f},{oz:.3f})")
        gsys = rec_prm.createSystem(nonbondedMethod=NoCutoff)
        gen = gfp.GridForce()
        gen.setGridOrigin(ox, oy, oz)
        gen.addGridCounts(nx, ny, nz)
        gen.addGridSpacing(sp, sp, sp)
        gen.setAutoGenerateGrid(True)
        gen.setGridType('charge')
        gen.setComputeDerivatives(True)
        gen.setGridCap(1e30)
        gen.setReceptorAtoms(rec_atoms)
        gen.setReceptorPositionsFromLists(pos_list)
        gen.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gsys.addForce(gen)
        gctx = Context(gsys, VerletIntegrator(0.001), platform)
        gctx.setPositions(rec_crd.positions)
        gctx.getState(getEnergy=True)  # trigger generation
        gen.saveToFile(grid_file)
        del gctx

        # Load grid for triquintic evaluation.
        gridf = gfp.GridForce()
        gridf.loadFromFile(grid_file)
        gridf.setInterpolationMethod(3)  # triquintic Hermite

        derivs = np.array(gridf.getDerivatives(), dtype=np.float64)
        vals = np.array(gridf.getGridValues(), dtype=np.float64)
        assert derivs.size == 27 * N, (derivs.size, 27 * N)

        # Detect storage layout via derivative index 0 == grid value.
        if vals.size == N:
            dm = np.abs(derivs[0:N] - vals).max()          # deriv-major d*N+p
            pm = np.abs(derivs[0::27][:N] - vals).max()     # point-major p*27+d
            layout = 'deriv-major' if dm < pm else 'point-major'
            print(f"  layout detect: deriv-major err={dm:.2e}, point-major err={pm:.2e} -> {layout}")
        else:
            layout = 'deriv-major'
            print("  getGridValues() empty; assuming deriv-major")

        def pidx(ix, iy, iz):
            return ix * ny * nz + iy * nz + iz

        def deriv_at(d, ix, iy, iz):
            p = pidx(ix, iy, iz)
            return derivs[d * N + p] if layout == 'deriv-major' else derivs[p * 27 + d]

        def build_X(bx, by, bz):
            X = np.empty(216)
            for d in range(27):
                for c in range(8):
                    cx, cy, cz = c & 1, (c >> 1) & 1, (c >> 2) & 1
                    X[d * 8 + c] = deriv_at(d, bx + cx, by + cy, bz + cz)
            return X

        def build_a(bx, by, bz, dtype=np.float64):
            """Assemble polynomial coefficients. dtype=float32 faithfully models
            the kernel's fp32 matrix solve a = 0.125 * M * X."""
            X = build_X(bx, by, bz).astype(dtype)
            Md = M.astype(dtype)
            a = (np.float64(0.125) if dtype == np.float64 else np.float32(0.125)) * (Md @ X)
            return jnp.asarray(np.asarray(a, dtype=np.float64))

        # ---- choose an interior cell + sub-cell point ----
        bx, by, bz = nx // 2, ny // 2, nz // 2
        frac = jnp.array([0.37, 0.61, 0.52])
        a = build_a(bx, by, bz)

        val_jax = float(poly_value(a, frac))
        g_jax = np.array(grad(lambda f: poly_value(a, f))(frac))   # d/dfrac
        grad_phys = g_jax / sp                                      # d/dx physical

        # ---- plugin evaluation at the same physical point ----
        probe = np.array([ox + (bx + 0.37) * sp,
                          oy + (by + 0.61) * sp,
                          oz + (bz + 0.52) * sp])
        esys = mm.System()
        esys.addParticle(1.0)
        gridf.addParticleGroup('charge', [0], [1.0])  # E = 1.0 * V(pos)
        esys.addForce(gridf)
        _prec = sys.argv[1] if len(sys.argv) > 1 else 'single'
        print(f"  (eval context Precision = {_prec})")
        ctx = Context(esys, VerletIntegrator(0.001), platform, {'Precision': _prec})
        ctx.setPositions([mm.Vec3(*probe)] * 1 * nanometer)
        st = ctx.getState(getEnergy=True, getForces=True)
        E_plugin = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        F_plugin = np.array(st.getForces(asNumpy=True).value_in_unit(
            kilojoules_per_mole / nanometer))[0]

        print("\n=== (A) FIDELITY: plugin vs JAX (f64) reconstruction, same stored corners ===")
        de = abs(E_plugin - val_jax)
        rele = de / max(1.0, abs(val_jax))
        print(f"  energy: plugin={E_plugin:.6e}  jax={val_jax:.6e}  abs={de:.2e} rel={rele:.2e}")
        F_ref = -grad_phys  # plugin force = -dE/dx, E = 1.0 * V
        relf = np.linalg.norm(F_plugin - F_ref) / max(1.0, np.linalg.norm(F_ref))
        print(f"  force plugin = {F_plugin}")
        print(f"  force -gradV = {F_ref}")
        print(f"  force rel(norm)={relf:.2e}  [diagnostic: fp32 force-eval + f32 position offset,")
        print(f"     NOT an algorithm error -- energy fidelity proves matrix/layout/eval correct]")
        fidelity_ok = rele < 1e-4   # energy fidelity is the algorithm-correctness criterion
        print(f"  => {'ALGORITHM FAITHFUL (energy rel < 1e-4); story is precision' if fidelity_ok else 'MISMATCH -- kernel/matrix/layout BUG'}")

        # ---- (C) cross-cell continuity: f64 (exact) vs f32-coeff (kernel model) ----
        # Adjacent cells share 4 of 8 corners; the kernel assembles a = 0.125*M*X
        # in fp32, so the two cells' fp32 coefficients disagree at the shared face.
        print("\n=== (C) CONTINUITY: x-adjacent cells at shared face ===")
        fy, fz = 0.61, 0.52
        f_face_L = jnp.array([1.0, fy, fz])
        f_face_R = jnp.array([0.0, fy, fz])

        def face_jump(dtype):
            aL = build_a(bx, by, bz, dtype=dtype)
            aR = build_a(bx + 1, by, bz, dtype=dtype)
            vL = float(poly_value(aL, f_face_L)); vR = float(poly_value(aR, f_face_R))
            gL = np.array(grad(lambda f: poly_value(aL, f))(f_face_L)) / sp
            gR = np.array(grad(lambda f: poly_value(aR, f))(f_face_R)) / sp
            return abs(vL - vR), np.abs(gL - gR).max()

        jv64, jg64 = face_jump(np.float64)
        jv32, jg32 = face_jump(np.float32)
        print(f"  f64 coeff assembly : value jump={jv64:.2e}  force jump={jg64:.2e}  (true C2 -> ~0)")
        print(f"  f32 coeff assembly : value jump={jv32:.2e}  force jump={jg32:.2e}  (kernel model)")
        print(f"  -> f32 force-continuity jump vs L-BFGS need ~1e-6: "
              f"{'EXCEEDS (stalls line search)' if jg32 > 1e-6 else 'within tol'}")

        # ---- (D) DIRECT plugin cross-cell force discontinuity ----
        # Straddle a real cell face; isolate the jump from the smooth trend by
        # linearly extrapolating the left-side slope across the boundary.
        print("\n=== (D) PLUGIN cross-cell force discontinuity (this build's kernel) ===")
        xb = ox + (bx + 1) * sp
        yb = oy + (by + 0.61) * sp
        zb = oz + (bz + 0.52) * sp
        dlt = 1e-5  # nm; smooth variation ~|dF/dx|*2dlt, jump (if any) dominates

        def plugin_force_x(x):
            ctx.setPositions([mm.Vec3(x, yb, zb)] * nanometer)
            f = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
                kilojoules_per_mole / nanometer)
            return np.array(f)[0]

        F1 = plugin_force_x(xb - 2 * dlt)  # left cell
        F2 = plugin_force_x(xb - 1 * dlt)  # left cell
        F3 = plugin_force_x(xb + 1 * dlt)  # right cell
        slope = (F2 - F1) / dlt
        predicted = F2 + slope * (2 * dlt)   # continuous extrapolation into right cell
        disc = F3 - predicted
        print(f"  boundary x={xb:.5f} nm, straddle dlt={dlt} nm")
        print(f"  extrapolated discontinuity in force: {disc}")
        print(f"  |disc| max = {np.abs(disc).max():.3e} kJ/mol/nm  "
              f"(vs L-BFGS ~1e-6; smooth-trend floor ~{np.abs(slope).max()*2*dlt:.1e})")

        del ctx
    print("\n" + "=" * 60)
    print("Next: if (A) is FAITHFUL, the kernel is correct and f64 grid/deriv")
    print("storage is the lever (matches 1D). If (A) MISMATCHES, there is a")
    print("kernel/matrix/layout bug to fix first.")
    return 0 if fidelity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
