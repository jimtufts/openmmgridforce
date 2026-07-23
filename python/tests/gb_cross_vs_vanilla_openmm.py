#!/usr/bin/env python
"""Compare plugin IsolatedGBSAForce PAIRWISE (cutoff off) vs vanilla
OpenMM GBSAOBCForce on the SAME complex.

Vanilla OpenMM's OBC2 treats the whole system (ligand + receptor) as one
soluble entity: one HCT sum over all atoms → one set of Born radii → one
GB energy summed over ALL pairs.

Our plugin's PAIRWISE mode does the same physics but splits bookkeeping
into ligand/receptor. With no cutoff, the two should give IDENTICAL
E_gb_total (up to fp precision).

If they don't match, the plugin has a bug — fix that before any
reaction-field work.

For each pose:
  E_vanilla   = vanilla GBSAOBCForce total (whole system)
  E_plugin    = plugin IsolatedGBSAForce PAIRWISE with cutoff off:
                  ligand-self GB + cross-term GB + receptor-desolvation delta
                (equivalent decomposition of the same total)

Both with SA disabled for clean comparison.

Env:
  PDBS=1t46,1r1h,1jje
  N_POSES=5
  EVAL_START=60
  OUT_JSON=<path>
"""
import functools, os, sys, time, json
import numpy as np

print = functools.partial(print, flush=True)

sys.path.insert(0, '/home/jtufts/src/p312/openmmgridforce/python/tests')

import openmm as mm
import openmm.unit as unit
import gridforceplugin as gf

from jax_gb_cross_grid_v2 import load_prmtop_params, load_dock6_mol2


PDBS = tuple(os.environ.get('PDBS', '1t46,1r1h,1jje').split(','))
N_POSES = int(os.environ.get('N_POSES', '5'))
EVAL_START = int(os.environ.get('EVAL_START', '60'))
ASTEX = '/home/dminh/backup/AstexDiv_xtal'


def setup_vanilla(lig_pos, lig_q, lig_r, lig_s,
                  rec_pos, rec_q, rec_r, rec_s):
    """Build vanilla OpenMM system with all ligand + receptor atoms and
    a GBSAOBCForce. SA off. Nonbonded off (we want ONLY GB).
    """
    n_lig = len(lig_q); n_rec = len(rec_q)
    n_total = n_lig + n_rec

    system = mm.System()
    for _ in range(n_total):
        system.addParticle(12.0)

    gb = mm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    # ACE surface area off — set surface tension to zero
    gb.setSurfaceAreaEnergy(0.0)
    gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)

    for i in range(n_lig):
        gb.addParticle(float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    for i in range(n_rec):
        gb.addParticle(float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))

    system.addForce(gb)

    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.VerletIntegrator(0.001)
    props = {'Precision': 'double'}
    context = mm.Context(system, integrator, platform, props)
    return gb, context, n_lig, n_rec


def setup_plugin_pairwise(lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s):
    """Plugin IsolatedGBSAForce PAIRWISE, cutoff off, SA off."""
    n_lig = len(lig_q); n_rec = len(rec_q)
    system = mm.System()
    for _ in range(n_lig):
        system.addParticle(12.0)
    f = gf.IsolatedGBSAForce()
    f.setGBMethod(gf.IsolatedGBSAForce.OBC_II)
    f.setSoluteDielectric(1.0)
    f.setSolventDielectric(78.5)
    f.setIncludeSurfaceArea(False)
    f.setNumAtoms(n_lig)
    for i in range(n_lig):
        f.setAtomParameters(i, float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    f.setReceptorMode(gf.IsolatedGBSAForce.PAIRWISE)
    f.setNumReceptorAtoms(n_rec)
    for i in range(n_rec):
        f.setReceptorAtomParameters(i, float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    f.setReceptorPositions(np.asarray(rec_pos).flatten().tolist())
    # No locality cutoff — this is our reference
    f.addParticleGroup('ligand', list(range(n_lig)))
    f.setGroupScalingFactor(0, 1.0)
    system.addForce(f)

    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.VerletIntegrator(0.001)
    props = {'Precision': 'double'}
    context = mm.Context(system, integrator, platform, props)
    return f, context


def process_one(pdb):
    print(f'\n===== {pdb} =====')
    lig_pt = f'{ASTEX}/1-build/{pdb}/ligand.prmtop'
    lig_ic = f'{ASTEX}/3-grids/{pdb}/ligand.trans.inpcrd'
    rec_pt = f'{ASTEX}/1-build/{pdb}/receptor.prmtop'
    rec_ic = f'{ASTEX}/3-grids/{pdb}/receptor.trans.inpcrd'
    poses_file = f'{ASTEX}/4-UCSF_dock6/{pdb}/xtal_plus_dock6_scored.mol2'

    _, l_q, l_r, l_s = load_prmtop_params(lig_pt, lig_ic)
    r_pos, r_q, r_r, r_s = load_prmtop_params(rec_pt, rec_ic)
    all_poses = [p for p in load_dock6_mol2(poses_file) if p.shape[0] == len(l_q)]
    poses = all_poses[EVAL_START:EVAL_START + N_POSES]
    print(f'  N_lig={len(l_q)}  N_rec={len(r_q)}  poses={len(poses)}')

    # Fresh contexts for each pose — avoids any residual state
    per_pose = []
    print(f'  {"pose":>4s}  {"E_vanilla (kJ/mol)":>19s}  {"E_plugin (kJ/mol)":>18s}  '
          f'{"diff":>9s}   {"rel_ppm":>9s}')
    for pk, pose in enumerate(poses):
        # Vanilla OpenMM: whole system in one context.
        gb, ctx_v, n_lig, n_rec = setup_vanilla(
            None, l_q, l_r, l_s, r_pos, r_q, r_r, r_s)
        full_positions = np.vstack([np.asarray(pose), np.asarray(r_pos)])
        ctx_v.setPositions(full_positions * unit.nanometer)
        state_v = ctx_v.getState(getEnergy=True)
        E_vanilla_kJ = state_v.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        del ctx_v

        # Plugin: only ligand atoms as OpenMM particles; receptor is inside force.
        f, ctx_p = setup_plugin_pairwise(l_q, l_r, l_s, r_pos, r_q, r_r, r_s)
        ctx_p.setPositions(np.asarray(pose) * unit.nanometer)
        state_p = ctx_p.getState(getEnergy=True)
        E_plugin_kJ = state_p.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        del ctx_p

        diff = E_plugin_kJ - E_vanilla_kJ
        rel_ppm = 1e6 * diff / (abs(E_vanilla_kJ) + 1e-12)
        per_pose.append(dict(pose=pk + EVAL_START,
                              E_vanilla_kJ=E_vanilla_kJ,
                              E_plugin_kJ=E_plugin_kJ,
                              diff_kJ=diff, rel_ppm=rel_ppm))
        print(f'  {pk:>3d}   {E_vanilla_kJ:>+16.6f}   '
              f'{E_plugin_kJ:>+16.6f}   {diff:>+8.4f}   {rel_ppm:>+8.2f}')
    diffs = np.array([r['diff_kJ'] for r in per_pose])
    print(f'  == {pdb} ==  max|diff|={np.max(np.abs(diffs)):.4f} kJ/mol  '
          f'RMSE={np.sqrt(np.mean(diffs**2)):.4f} kJ/mol')
    return dict(pdb=pdb, n_lig=len(l_q), n_rec=len(r_q),
                per_pose=per_pose,
                max_abs_diff_kJ=float(np.max(np.abs(diffs))),
                rmse_kJ=float(np.sqrt(np.mean(diffs**2))))


def main():
    print(f'== plugin PAIRWISE vs vanilla OpenMM GBSAOBCForce ==')
    print(f'  cutoff=OFF   SA=OFF   precision=double')
    print(f'  PDBS={PDBS}  N_POSES={N_POSES}')
    results = []
    for pdb in PDBS:
        try:
            results.append(process_one(pdb))
        except Exception as e:
            print(f'  {pdb} FAILED: {e}')
            import traceback; traceback.print_exc()
    print(f'\n=== overall ===')
    print(f'  {"pdb":6s}  {"max|diff| kJ":>14s}  {"RMSE kJ":>9s}')
    for r in results:
        print(f'  {r["pdb"]:6s}  {r["max_abs_diff_kJ"]:>14.6f}  {r["rmse_kJ"]:>9.6f}')
    out = os.environ.get('OUT_JSON')
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as fh:
            json.dump({'results': results}, fh, indent=2)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
