#!/usr/bin/env python
"""
Probe: measure the magnitude range of Hessian entries to size the fixed-point
accumulator for the precision refactor (see DESIGN_PRECISION_SELECTION.md, Tier 4).

The Hessian is accumulated in CUDA via deterministic fixed-point atomicAdd with
HESSIAN_SCALE = 2^24 (bondedHessian.cu:20, isolatedNonbonded.cu:9). Two ends of
the dynamic range constrain a single 64-bit accumulator:

  * OVERFLOW   : scale * max|H_entry| must stay below 2^63.
  * RESOLUTION : the absolute quantum 2^-s (s = log2(scale)) must be well below
                 the smallest eigenvalue we care about, or soft modes (which
                 dominate vibrational entropy and the SoftAbs metric) get
                 corrupted.

This script measures, across representative Astex ligands:
  - max |entry| of the bonded Hessian (BondedHessian)        -> stiffest terms
  - max |entry| of the isolated nonbonded Hessian            -> close-contact LJ
  - per-atom 3x3 block eigenvalues (what metricAssembly /
    gridHessianAnalysis actually eigendecompose)             -> soft-mode floor
  - full-matrix raw eigenvalue range (informational)

then reports whether a single 64-bit fixed-point scale can hold the full range,
and recommends a scale s (or flags that a 128-bit / two-word accumulator is
needed).

The GridForce Hessian is intentionally NOT included: it is bounded by the grid
cap and is far softer than bonded/LJ terms, so it does not widen the range that
sizes the accumulator. (Bonded + LJ are the worst case on both ends.)

Usage:
    python probe_hessian_magnitude.py                 # default 12 systems
    python probe_hessian_magnitude.py --n 30          # more systems
    python probe_hessian_magnitude.py --systems 1g9v,1gkc
    python probe_hessian_magnitude.py --no-minimize   # skip minimization
"""

import argparse
import math
import os
import sys

import numpy as np
import openmm as mm
from openmm import Platform, Context, VerletIntegrator, LocalEnergyMinimizer
from openmm import NonbondedForce
from openmm.app import AmberPrmtopFile, AmberInpcrdFile
from openmm.unit import (nanometer, kilojoules_per_mole, elementary_charge,
                         kilojoule_per_mole)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridforceplugin as gfp
from gridforceplugin import BondedHessian
from benchmark_utils import (ASTEX_BASE, get_system_paths, validate_system_paths,
                             create_evaluation_system, load_systems_list)

# Must match the CUDA kernels.
CURRENT_SCALE_BITS = 24          # HESSIAN_SCALE = 0x1000000 = 2^24
ACC_BITS = 63                    # signed 64-bit accumulator
# Safety margins for the recommendation.
OVERFLOW_HEADROOM_BITS = 8       # guard against partial sums / cancellation
SOFTMODE_PRECISION_BITS = 20     # relative bits we want preserved on soft modes


def extract_lig_params(prmtop):
    """Return [(charge, sigma, epsilon), ...] from the prmtop's NonbondedForce."""
    sysm = prmtop.createSystem()
    nb = next(f for f in (sysm.getForce(i) for i in range(sysm.getNumForces()))
              if isinstance(f, NonbondedForce))
    params = []
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        params.append((q.value_in_unit(elementary_charge),
                       sig.value_in_unit(nanometer),
                       eps.value_in_unit(kilojoules_per_mole)))
    return params


def block_eigs_from_full(full_H, n_atoms):
    """Eigenvalues of each per-atom diagonal 3x3 block (mirrors metricAssembly)."""
    eigs = []
    for i in range(n_atoms):
        b = 3 * i
        blk = full_H[b:b+3, b:b+3]
        blk = 0.5 * (blk + blk.T)
        eigs.append(np.linalg.eigvalsh(blk))
    return np.array(eigs)  # (n_atoms, 3)


def analyze_system(name, minimize=True):
    """Build a gas-phase ligand (bonded + isolated NB), compute its Hessians,
    return a dict of magnitude statistics, or None on failure/skip."""
    paths = get_system_paths(name)
    needed = {'ligand_prmtop': paths['ligand_prmtop'],
              'ligand_inpcrd': paths['ligand_inpcrd']}
    missing = [p for p in needed.values() if not os.path.exists(p)]
    if missing:
        print(f"  [skip] {name}: missing {missing}")
        return None

    prmtop = AmberPrmtopFile(needed['ligand_prmtop'])
    inpcrd = AmberInpcrdFile(needed['ligand_inpcrd'])
    lig_params = extract_lig_params(prmtop)
    n_atoms = len(lig_params)

    # Exact production force path: bonded forces + IsolatedNonbondedForce, no grids.
    system, _, isolated_nb = create_evaluation_system(
        prmtop, lig_params, grid_files={}, method=3)

    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, VerletIntegrator(0.001), platform)
    context.setPositions(inpcrd.positions)
    if minimize:
        # Normal modes / entropy are evaluated at minima, so size the accumulator there.
        LocalEnergyMinimizer.minimize(context, 10.0, 2000)

    n3 = 3 * n_atoms

    # Bonded Hessian (full 3N x 3N).
    bonded = BondedHessian()
    bonded.initialize(system, context)
    H_bonded = np.array(bonded.computeHessian(context)).reshape(n3, n3)

    # Isolated nonbonded Hessian (full 3N x 3N).
    H_nb = (np.array(isolated_nb.computeHessian(context)).reshape(n3, n3)
            if isolated_nb is not None else np.zeros((n3, n3)))

    H_tot = H_bonded + H_nb
    H_tot = 0.5 * (H_tot + H_tot.T)

    # Per-atom 3x3 block eigenvalues (what the metric/curvature kernels consume).
    block_eigs = block_eigs_from_full(H_tot, n_atoms)
    abs_block_eigs = np.abs(block_eigs)
    # "Nonzero" relative to the largest block eigenvalue in this system.
    blk_max = abs_block_eigs.max() if abs_block_eigs.size else 0.0
    nz = abs_block_eigs[abs_block_eigs > blk_max * 1e-9] if blk_max > 0 else np.array([])

    # Full-matrix raw eigenvalues (informational); drop ~zero rigid-body modes.
    full_eigs = np.linalg.eigvalsh(H_tot)
    fmax = np.abs(full_eigs).max() if full_eigs.size else 0.0
    full_nz = np.abs(full_eigs)[np.abs(full_eigs) > fmax * 1e-9] if fmax > 0 else np.array([])

    del context
    return {
        'name': name,
        'n_atoms': n_atoms,
        'max_bonded': float(np.abs(H_bonded).max()),
        'max_nb': float(np.abs(H_nb).max()),
        'max_entry': float(np.abs(H_tot).max()),
        'block_eig_min_nz': float(nz.min()) if nz.size else float('nan'),
        'block_eig_max': float(blk_max),
        'full_eig_min_nz': float(full_nz.min()) if full_nz.size else float('nan'),
        'full_eig_max': float(fmax),
    }


def fmt(x):
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12, help='number of systems to probe')
    ap.add_argument('--systems', type=str, default=None,
                    help='comma-separated system names (overrides --n)')
    ap.add_argument('--no-minimize', action='store_true')
    args = ap.parse_args()

    if args.systems:
        names = [s.strip() for s in args.systems.split(',') if s.strip()]
    else:
        try:
            names = load_systems_list()
        except Exception:
            names = sorted(os.listdir(os.path.join(ASTEX_BASE, '1-build')))
        names = names[:args.n]

    print(f"Probing {len(names)} system(s) from {ASTEX_BASE}")
    print(f"minimize={'no' if args.no_minimize else 'yes'}, "
          f"current HESSIAN_SCALE = 2^{CURRENT_SCALE_BITS}\n")

    hdr = (f"{'system':<10}{'N':>5}{'max|bonded|':>14}{'max|nb|':>14}"
           f"{'max|entry|':>14}{'blk_eig_min':>14}{'full_eig_min':>14}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name in names:
        try:
            r = analyze_system(name, minimize=not args.no_minimize)
        except Exception as e:
            print(f"  [error] {name}: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"{r['name']:<10}{r['n_atoms']:>5}{fmt(r['max_bonded']):>14}"
              f"{fmt(r['max_nb']):>14}{fmt(r['max_entry']):>14}"
              f"{fmt(r['block_eig_min_nz']):>14}{fmt(r['full_eig_min_nz']):>14}")

    if not rows:
        print("\nNo systems analyzed. Check data paths / env.")
        return 1

    g_max_entry = max(r['max_entry'] for r in rows)
    g_max_bonded = max(r['max_bonded'] for r in rows)
    g_max_nb = max(r['max_nb'] for r in rows)
    blk_mins = [r['block_eig_min_nz'] for r in rows if not math.isnan(r['block_eig_min_nz'])]
    full_mins = [r['full_eig_min_nz'] for r in rows if not math.isnan(r['full_eig_min_nz'])]
    g_blk_min = min(blk_mins) if blk_mins else float('nan')
    g_full_min = min(full_mins) if full_mins else float('nan')

    print("\n" + "=" * 70)
    print("AGGREGATE (across all probed systems)")
    print("=" * 70)
    print(f"  max |entry|             : {fmt(g_max_entry)}  (~2^{math.log2(g_max_entry):.1f})")
    print(f"    of which bonded       : {fmt(g_max_bonded)}")
    print(f"    of which nonbonded    : {fmt(g_max_nb)}")
    print(f"  min |3x3-block eig|     : {fmt(g_blk_min)}"
          + (f"  (~2^{math.log2(g_blk_min):.1f})" if g_blk_min == g_blk_min and g_blk_min > 0 else ""))
    print(f"  min |full-matrix eig|   : {fmt(g_full_min)}"
          + (f"  (~2^{math.log2(g_full_min):.1f})" if g_full_min == g_full_min and g_full_min > 0 else ""))

    # --- Sizing analysis -------------------------------------------------
    print("\n" + "=" * 70)
    print("FIXED-POINT ACCUMULATOR SIZING")
    print("=" * 70)

    # Upper bound on scale before overflow.
    s_overflow = ACC_BITS - math.log2(g_max_entry) - OVERFLOW_HEADROOM_BITS
    # Lower bound on scale to preserve soft modes. Use the FULL-matrix softest
    # eigenvalue: it is the entropy/normal-mode-relevant floor and is far smaller
    # (stricter) than the per-atom 3x3 block min. Fall back to block min if the
    # full spectrum is unavailable.
    soft = g_full_min if (g_full_min == g_full_min and g_full_min > 0) else g_blk_min
    if soft == soft and soft > 0:
        s_resolution = -math.log2(soft) + SOFTMODE_PRECISION_BITS
    else:
        s_resolution = float('nan')

    print(f"  Current scale s = {CURRENT_SCALE_BITS}: absolute quantum = 2^-{CURRENT_SCALE_BITS} "
          f"= {2.0**-CURRENT_SCALE_BITS:.2e}")
    if soft == soft and soft > 0:
        rel = (2.0**-CURRENT_SCALE_BITS) / soft
        print(f"    -> quantum / softest block eig = {rel:.2e} "
              f"({'OK' if rel < 1e-3 else 'COARSE — soft modes at risk'})")
    print(f"  Max scale before overflow (with {OVERFLOW_HEADROOM_BITS}-bit headroom): "
          f"s <= {s_overflow:.1f}")
    if s_resolution == s_resolution:
        print(f"  Min scale to keep {SOFTMODE_PRECISION_BITS} bits on softest mode:    "
              f"s >= {s_resolution:.1f}")

    print()
    if s_resolution == s_resolution:
        if s_resolution <= s_overflow:
            rec = int(math.floor((s_resolution + s_overflow) / 2))
            print(f"  ==> A single 64-bit accumulator SUFFICES. Recommended scale: "
                  f"s = {rec}  (2^{rec})")
            print(f"      Feasible window: s in [{math.ceil(s_resolution)}, "
                  f"{math.floor(s_overflow)}].")
        else:
            print("  ==> No single 64-bit scale satisfies BOTH overflow and soft-mode")
            print(f"      resolution (need s>={s_resolution:.1f} but s<={s_overflow:.1f}).")
            print("      -> Use a 128-bit / two-word or block-relative-scaled accumulator.")
    else:
        print("  ==> Could not determine soft-mode floor (no nonzero block eigenvalues).")

    print("\nNotes:")
    print("  * Grid Hessian excluded (softer, grid-cap bounded) — does not widen range.")
    print("  * Run with more systems (--n) for a population max before committing a scale.")
    print("  * Resolution bound uses the FULL-matrix softest eigenvalue (entropy/")
    print("    normal-mode floor); block-eig min is the per-atom metric floor and is")
    print("    far larger (looser).")
    print("  * Measured at minimized poses; loosely-minimized / near-saddle poses can")
    print("    push eigenvalues toward 0 (a conditioning issue no finite scale fixes).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
