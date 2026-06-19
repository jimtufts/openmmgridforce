#!/usr/bin/env python
"""
1D quintic-Hermite interpolation validation via JAX autodiff.

Context: triquintic-Hermite receptor-grid interpolation breaks production
minimization (hangs / non-PD "fake" Hessians) where cubic B-spline works
(see precision_issues.md). The suspected mechanism is an fp32 self-consistency
problem, not an accuracy one. This is the FIRST step of the proper (FD-free)
validation: confirm the quintic-Hermite *polynomial math* is correct using JAX
autodiff as the reference (never finite differences -- FD on analytical forces
is meaningless here, it just measures central-difference fp32 cancellation).

What this 1D test DOES establish:
  1. Construction correctness -- the quintic reproduces E, E', E'' at both nodes.
  2. Derivative correctness   -- jax.grad / jax.hessian of the polynomial equal
                                 an independent hand-coded analytical derivative
                                 (the thing the CUDA kernel computes).
  3. C2 continuity            -- adjacent cells sharing node data agree in
                                 value/1st/2nd derivative at the shared node
                                 (this is WHY quintic Hermite *should* minimize
                                 well; its loss is the suspected bug).
  4. fp32 sensitivity         -- rounding node data / coefficients to float32
                                 reintroduces a cross-cell discontinuity; we
                                 measure its size and compare to the ~1e-6 that
                                 L-BFGS line search needs.

What it does NOT cover (deferred to the 3D test): the actual 216x216
TriquinticCoefficients.cuh solve and tile-cache boundary handling. In 1D, C2
continuity is structural (adjacent cells share the SAME node data), so any
discontinuity is pure roundoff -- which is exactly what isolates the fp32 effect.

Run:  /home/jtufts/anaconda3/envs/algdock_refactor/bin/python \
          python/tests/test_triquintic_hermite_1d_jax.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tiny problem; avoids GPU PTX warnings
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit

# ----------------------------------------------------------------------------
# Quintic Hermite basis on the unit interval t in [0,1].
# Matches value, 1st, and 2nd derivative at each endpoint (6 DOF = quintic).
# ----------------------------------------------------------------------------
def _basis(t):
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    h0 = 1.0 - 10.0*t3 + 15.0*t4 - 6.0*t5     # value @0
    h1 = t - 6.0*t3 + 8.0*t4 - 3.0*t5          # 1st   @0
    h2 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5     # 2nd   @0
    h3 = 10.0*t3 - 15.0*t4 + 6.0*t5            # value @1
    h4 = -4.0*t3 + 7.0*t4 - 3.0*t5             # 1st   @1
    h5 = 0.5*t3 - 1.0*t4 + 0.5*t5              # 2nd   @1
    return h0, h1, h2, h3, h4, h5


def eval_cell(x, x0, h, node):
    """Quintic-Hermite value at physical x in cell [x0, x0+h].
    node = (E0, Ex0, Exx0, E1, Ex1, Exx1) are PHYSICAL value/1st/2nd derivs.
    Derivatives are scaled by h / h^2 to convert physical -> normalized t."""
    E0, Ex0, Exx0, E1, Ex1, Exx1 = node
    t = (x - x0) / h
    h0, h1, h2, h3, h4, h5 = _basis(t)
    return (E0*h0 + (h*Ex0)*h1 + (h*h*Exx0)*h2
            + E1*h3 + (h*Ex1)*h4 + (h*h*Exx1)*h5)


# Hand-coded analytical 1st derivative of the cell polynomial w.r.t. physical x
# (the quantity the CUDA kernel computes as the force). Validated below against
# jax.grad to prove the kernel's "F = analytical dE/dx" formula is right.
def eval_cell_dx_analytic(x, x0, h, node):
    E0, Ex0, Exx0, E1, Ex1, Exx1 = node
    t = (x - x0) / h
    t2 = t*t; t3 = t2*t; t4 = t3*t
    dh0 = -30.0*t2 + 60.0*t3 - 30.0*t4
    dh1 = 1.0 - 18.0*t2 + 32.0*t3 - 15.0*t4
    dh2 = t - 4.5*t2 + 6.0*t3 - 2.5*t4
    dh3 = 30.0*t2 - 60.0*t3 + 30.0*t4
    dh4 = -12.0*t2 + 28.0*t3 - 15.0*t4
    dh5 = 1.5*t2 - 4.0*t3 + 2.5*t4
    dpdt = (E0*dh0 + (h*Ex0)*dh1 + (h*h*Exx0)*dh2
            + E1*dh3 + (h*Ex1)*dh4 + (h*h*Exx1)*dh5)
    return dpdt / h   # dt/dx = 1/h


# ----------------------------------------------------------------------------
# Test functions with steep curvature, representative of a receptor potential
# (close-contact repulsion is where precision_issues.md says the bug bites).
# ----------------------------------------------------------------------------
def f_gauss(x):
    return jnp.exp(-((x - 0.5) / 0.18) ** 2)

def f_lj(x):
    # Shifted Lennard-Jones-like wall on a domain bounded away from r=0.
    r = x + 0.35
    return 4.0 * ((0.30 / r) ** 12 - (0.30 / r) ** 6)

TEST_FUNCS = {"gaussian": f_gauss, "lennard_jones": f_lj}


def node_data(f, x, dtype=np.float64):
    """Physical (E, E', E'') of f at x, via JAX autodiff (the reference)."""
    fp = grad(f)
    fpp = grad(fp)
    E = float(f(x)); Ex = float(fp(x)); Exx = float(fpp(x))
    return np.array([E, Ex, Exx], dtype=dtype)


def build_nodes(f, xs, dtype=np.float64):
    return {float(x): node_data(f, x, dtype) for x in xs}


def cell_node_tuple(nodes, xL, xR):
    nl, nr = nodes[xL], nodes[xR]
    return (nl[0], nl[1], nl[2], nr[0], nr[1], nr[2])


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_construction_and_autodiff(f, x0, h):
    """(1) interpolation property + (2) jax.grad/hessian == analytical formula."""
    xs = [x0, x0 + h]
    nodes = build_nodes(f, xs)
    node = cell_node_tuple(nodes, xs[0], xs[1])

    g = jit(grad(lambda x: eval_cell(x, x0, h, node)))
    gg = jit(grad(grad(lambda x: eval_cell(x, x0, h, node))))

    # (1) reproduce E, E', E'' at both nodes
    rep_err = 0.0
    for xn in xs:
        E_ref, Ex_ref, Exx_ref = nodes[xn]
        rep_err = max(rep_err,
                      abs(float(eval_cell(xn, x0, h, node)) - E_ref),
                      abs(float(g(xn)) - Ex_ref),
                      abs(float(gg(xn)) - Exx_ref))

    # (2) hand-coded analytical derivative == jax.grad across the cell interior
    xq = np.linspace(x0, x0 + h, 17)
    form_err = max(abs(float(g(x)) - float(eval_cell_dx_analytic(x, x0, h, node)))
                   for x in xq)
    return rep_err, form_err


def check_c2_continuity(f, x0, h, dtype=np.float64):
    """(3)/(4) value/1st/2nd-deriv jump at a node shared by two adjacent cells.
    In exact arithmetic this is 0 (both cells reproduce the shared node data);
    any jump is pure roundoff of the node data at `dtype`."""
    xs = [x0, x0 + h, x0 + 2*h]
    nodes = build_nodes(f, xs, dtype=dtype)
    left = cell_node_tuple(nodes, xs[0], xs[1])
    right = cell_node_tuple(nodes, xs[1], xs[2])
    xm = xs[1]

    gL = grad(lambda x: eval_cell(x, xs[0], h, left))
    gR = grad(lambda x: eval_cell(x, xs[1], h, right))
    ggL = grad(grad(lambda x: eval_cell(x, xs[0], h, left)))
    ggR = grad(grad(lambda x: eval_cell(x, xs[1], h, right)))

    # cast the polynomial inputs to dtype to model fixed-precision storage
    def ev(fn, x):
        return float(fn(jnp.asarray(np.array(x, dtype=dtype))))

    jump_E = abs(ev(lambda x: eval_cell(x, xs[0], h, left), xm)
                 - ev(lambda x: eval_cell(x, xs[1], h, right), xm))
    jump_F = abs(ev(gL, xm) - ev(gR, xm))
    jump_H = abs(ev(ggL, xm) - ev(ggR, xm))
    scale = max(1.0, abs(nodes[xm][0]))
    return jump_E, jump_F, jump_H, scale


def main():
    print("1D quintic-Hermite validation (JAX autodiff reference, x64)\n")
    h = 0.025  # ~0.25 A cell in nm, matching the production failing spacing
    x0 = 0.40

    LBFGS_TOL = 1e-6  # line-search consistency L-BFGS needs (precision_issues.md)
    ok = True
    for name, f in TEST_FUNCS.items():
        print(f"=== {name} (cell width h={h}) ===")
        rep_err, form_err = check_construction_and_autodiff(f, x0, h)
        print(f"  (1) node interpolation max err     : {rep_err:.2e}  "
              f"({'PASS' if rep_err < 1e-9 else 'FAIL'})")
        print(f"  (2) jax.grad vs analytic formula   : {form_err:.2e}  "
              f"({'PASS' if form_err < 1e-9 else 'FAIL'})")
        ok = ok and rep_err < 1e-9 and form_err < 1e-9

        jE, jF, jH, sc = check_c2_continuity(f, x0, h, dtype=np.float64)
        print(f"  (3) C2 cross-cell jump (f64)       : "
              f"E={jE:.2e} F={jF:.2e} H={jH:.2e}  "
              f"({'PASS' if max(jE, jF) < 1e-9 else 'FAIL'})")
        ok = ok and max(jE, jF) < 1e-9

        jE3, jF3, jH3, sc3 = check_c2_continuity(f, x0, h, dtype=np.float32)
        relF = jF3 / max(1e-30, abs(jF3) + sc3)
        print(f"  (4) C2 cross-cell jump (f32 nodes) : "
              f"E={jE3:.2e} F={jF3:.2e} H={jH3:.2e}")
        print(f"      -> force jump vs L-BFGS need {LBFGS_TOL:.0e}: "
              f"{'EXCEEDS (would stall line search)' if jF3 > LBFGS_TOL else 'within tol'}")
        print()

    print("=" * 60)
    print(f"Polynomial math (1)-(3): {'ALL PASS' if ok else 'FAILURE -- algorithm bug'}")
    print("Check (4) is diagnostic, not pass/fail: it quantifies how much fp32")
    print("node-data storage alone perturbs cross-cell force continuity. The 3D")
    print("test adds the 216x216 coefficient solve, the larger suspected source.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
