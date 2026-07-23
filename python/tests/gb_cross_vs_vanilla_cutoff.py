#!/usr/bin/env python
"""Validate plugin PAIRWISE with cutoff == vanilla OpenMM GBSAOBCForce
CutoffNonPeriodic — for each cutoff distance R_c:

  Delta_vanilla(R_c) = E_v_full(R_c) - E_v_receptor_only(R_c)
  E_plugin(R_c)      = plugin PAIRWISE getPotentialEnergy at R_c

Both use identical R_c. If plugin implements the cutoff to match vanilla
(strict truncation on HCT and GB pair, plus -q_i*q_j/R_c shift on GB
pairs), then Delta_vanilla - E_plugin should be a POSE-INDEPENDENT
constant per system. That constant may vary WITH cutoff (because
receptor-vacuum-at-cutoff changes) but must not vary across poses for a
given cutoff.

Env:
  PDBS=1t46,1r1h,1jje
  N_POSES=5
  CUTOFFS=0.8,1.0,1.2,1.5,2.0    (nm)
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
CUTOFFS = tuple(float(x) for x in
                os.environ.get('CUTOFFS', '0.8,1.0,1.2,1.5,2.0').split(','))
ASTEX = '/home/dminh/backup/AstexDiv_xtal'


def setup_vanilla(lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s,
                   cutoff):
    n_lig = len(lig_q); n_rec = len(rec_q)
    system = mm.System()
    for _ in range(n_lig + n_rec):
        system.addParticle(12.0)
    gb = mm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setSurfaceAreaEnergy(0.0)
    if cutoff > 0:
        gb.setNonbondedMethod(mm.GBSAOBCForce.CutoffNonPeriodic)
        gb.setCutoffDistance(cutoff)
    else:
        gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
    for i in range(n_lig):
        gb.addParticle(float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    for i in range(n_rec):
        gb.addParticle(float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    system.addForce(gb)
    ctx = mm.Context(system,
                     mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    return ctx, n_lig, n_rec


def setup_vanilla_rec_only(rec_pos, rec_q, rec_r, rec_s, cutoff):
    n_rec = len(rec_q)
    system = mm.System()
    for _ in range(n_rec):
        system.addParticle(12.0)
    gb = mm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setSurfaceAreaEnergy(0.0)
    if cutoff > 0:
        gb.setNonbondedMethod(mm.GBSAOBCForce.CutoffNonPeriodic)
        gb.setCutoffDistance(cutoff)
    else:
        gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
    for i in range(n_rec):
        gb.addParticle(float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    system.addForce(gb)
    ctx = mm.Context(system,
                     mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    return ctx


def setup_plugin(lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s, cutoff):
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
    if cutoff > 0:
        f.setCutoffDistance(cutoff)
    f.addParticleGroup('ligand', list(range(n_lig)))
    f.setGroupScalingFactor(0, 1.0)
    system.addForce(f)
    ctx = mm.Context(system,
                     mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    return f, ctx


def process_pdb(pdb):
    print(f'\n===== {pdb} =====', flush=True)
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

    results = {}
    for cutoff in CUTOFFS:
        # Rec-only baseline (pose-independent — compute once)
        ctx_r = setup_vanilla_rec_only(r_pos, r_q, r_r, r_s, cutoff)
        ctx_r.setPositions(np.asarray(r_pos) * unit.nanometer)
        E_v_rec = ctx_r.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        del ctx_r

        print(f'  cutoff={cutoff:.2f} nm  E_v_rec_only={E_v_rec:+.4f}')
        per_pose = []
        for pk, pose in enumerate(poses):
            pose_arr = np.asarray(pose)
            # Vanilla full
            ctx_v, _, _ = setup_vanilla(l_q, l_r, l_s, r_pos, r_q, r_r, r_s, cutoff)
            full = np.vstack([pose_arr, np.asarray(r_pos)])
            ctx_v.setPositions(full * unit.nanometer)
            E_v_full = ctx_v.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            del ctx_v

            # Plugin
            f_, ctx_p = setup_plugin(l_q, l_r, l_s, r_pos, r_q, r_r, r_s, cutoff)
            ctx_p.setPositions(pose_arr * unit.nanometer)
            E_p = ctx_p.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            del ctx_p

            delta_v = E_v_full - E_v_rec
            gap = E_p - delta_v
            per_pose.append(dict(pose=pk + EVAL_START,
                                  E_v_full=E_v_full, E_v_rec=E_v_rec, E_p=E_p,
                                  delta_v=delta_v, gap=gap))
            print(f'    pose {pk}: E_v_full={E_v_full:+.3f}  E_p={E_p:+.3f}  '
                  f'delta_v={delta_v:+.3f}  gap={gap:+.6f}')
        gaps = np.array([r['gap'] for r in per_pose])
        results[cutoff] = dict(per_pose=per_pose,
                                E_v_rec=E_v_rec,
                                gap_span=float(gaps.max() - gaps.min()),
                                gap_mean=float(gaps.mean()),
                                gap_max_abs=float(np.max(np.abs(gaps))))
        print(f'    == cutoff={cutoff}  gap_span={gaps.max()-gaps.min():.6f}  '
              f'gap_max|abs|={np.max(np.abs(gaps)):.6f}', flush=True)
    return dict(pdb=pdb, n_lig=len(l_q), n_rec=len(r_q), results=results)


def main():
    print(f'== plugin vs vanilla under CutoffNonPeriodic ==')
    print(f'  PDBS={PDBS}  N_POSES={N_POSES}  CUTOFFS(nm)={CUTOFFS}')
    all_results = []
    for pdb in PDBS:
        try:
            all_results.append(process_pdb(pdb))
        except Exception as e:
            print(f'  {pdb} FAILED: {e}')
            import traceback; traceback.print_exc()

    print(f'\n=== overall ===')
    print(f'  {"pdb":6s}  ' + '  '.join(f'{"cut=%.2f"%c:>10s}' for c in CUTOFFS))
    for r in all_results:
        row = [r['pdb']]
        for c in CUTOFFS:
            span = r['results'][c]['gap_span']
            row.append(f'{span:.6f}')
        print(f'  {row[0]:6s}  ' + '  '.join(f'{s:>10s}' for s in row[1:]))
    print()
    print(f'  gap_span = max|gap| - min|gap| across poses at fixed cutoff.')
    print(f'  Ideal: 0 (pose-independent offset means match). If nonzero,')
    print(f'  we still have a pose-dependent bug.')

    out = os.environ.get('OUT_JSON')
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as fh:
            json.dump({'results': all_results}, fh, indent=2)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
