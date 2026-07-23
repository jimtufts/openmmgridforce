#!/usr/bin/env python
"""Measure pose-to-pose variance in the plugin cutoff truncation offset.

For each pose:
  E_cross_off(pose)  = plugin PAIRWISE E_cross with cutoff = OFF
  E_cross_cut(pose)  = plugin PAIRWISE E_cross at cutoff R_c
  offset(pose)       = E_cross_off - E_cross_cut

Report:
  - Distribution of offsets across poses
  - Std/Range as a fraction of the mean offset
  - std/kT to gauge Boltzmann-weight bias in BPMF

If std(offset) is small (<< kT ~= 0.6 kcal/mol at 300 K) relative to the
mean, the truncation acts as a constant shift that cancels in BPMF.
If std(offset) is comparable to or larger than kT, the truncation biases
the sampled ensemble and BPMF is affected.

Env:
  PDBS=1t46,1r1h,1jje
  N_POSES=30
  EVAL_START=0
  CUTOFFS=0.8,1.0,1.2,1.5,2.0
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
N_POSES = int(os.environ.get('N_POSES', '30'))
EVAL_START = int(os.environ.get('EVAL_START', '0'))
CUTOFFS = tuple(float(x) for x in
                 os.environ.get('CUTOFFS', '0.8,1.0,1.2,1.5,2.0').split(','))
ASTEX = '/home/dminh/backup/AstexDiv_xtal'
KCAL_PER_KJ = 1.0 / 4.184
KT_AT_300K = 0.5924  # kcal/mol


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
        f.setCutoffDistance(float(cutoff))
    f.addParticleGroup('ligand', list(range(n_lig)))
    f.setGroupScalingFactor(0, 1.0)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName('CUDA'),
                     {'Precision': 'double'})
    return f, ctx


def E_cross_at(pose, lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s, cutoff):
    f, ctx = setup_plugin(lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s, cutoff)
    ctx.setPositions(np.asarray(pose) * unit.nanometer)
    _ = ctx.getState(getEnergy=True)
    E = float(f.getGroupCrossTermEnergy(0)) * KCAL_PER_KJ
    del ctx
    return E


def process_pdb(pdb):
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

    # Compute E_cross(no cutoff) once per pose
    print(f'  computing reference E_cross (no cutoff) for {len(poses)} poses...')
    t0 = time.time()
    E_ref = np.array([E_cross_at(pose, l_q, l_r, l_s, r_pos, r_q, r_r, r_s, -1)
                       for pose in poses])
    print(f'    done in {time.time()-t0:.1f}s')
    print(f'    ref E_cross range: {E_ref.min():+.3f} to {E_ref.max():+.3f} kcal/mol'
          f'   mean={E_ref.mean():+.3f}  std={E_ref.std():.3f}')

    results = {}
    for cutoff in CUTOFFS:
        print(f'  cutoff={cutoff:.2f} nm ...', flush=True)
        t0 = time.time()
        E_cut = np.array([E_cross_at(pose, l_q, l_r, l_s, r_pos, r_q, r_r, r_s, cutoff)
                           for pose in poses])
        offset = E_ref - E_cut          # E_cross missing due to truncation
        print(f'    {time.time()-t0:.1f}s. E_cut range: '
              f'{E_cut.min():+.3f} to {E_cut.max():+.3f}   '
              f'offset mean={offset.mean():+.3f} std={offset.std():.3f} '
              f'(std/kT={offset.std()/KT_AT_300K:.1f})')
        results[cutoff] = dict(
            E_ref=E_ref.tolist(), E_cut=E_cut.tolist(),
            offset_mean=float(offset.mean()),
            offset_std=float(offset.std()),
            offset_min=float(offset.min()),
            offset_max=float(offset.max()),
            offset_span=float(offset.max() - offset.min()),
            std_over_kT=float(offset.std() / KT_AT_300K),
        )
    return dict(pdb=pdb, n_lig=len(l_q), n_rec=len(r_q), n_poses=len(poses),
                results=results)


def main():
    print(f'== truncation-offset pose variance ==')
    print(f'  PDBS={PDBS}  N_POSES={N_POSES}  EVAL_START={EVAL_START}')
    print(f'  CUTOFFS(nm)={CUTOFFS}   kT@300K = {KT_AT_300K:.4f} kcal/mol')
    all_r = []
    for pdb in PDBS:
        try:
            all_r.append(process_pdb(pdb))
        except Exception as e:
            print(f'  {pdb} FAILED: {e}')
            import traceback; traceback.print_exc()

    print(f'\n=== overall (offset std across {N_POSES} poses, kcal/mol) ===')
    print(f'  {"pdb":6s}  ' + '  '.join(f'{"cut=%.2f"%c:>10s}' for c in CUTOFFS))
    for r in all_r:
        row = ' '.join(f'{r["results"][c]["offset_std"]:>10.3f}' for c in CUTOFFS)
        print(f'  {r["pdb"]:6s}  {row}')
    print()
    print(f'  offset std / kT — >>1 means truncation biases the sampled ensemble')
    print(f'  {"pdb":6s}  ' + '  '.join(f'{"cut=%.2f"%c:>10s}' for c in CUTOFFS))
    for r in all_r:
        row = ' '.join(f'{r["results"][c]["std_over_kT"]:>10.1f}' for c in CUTOFFS)
        print(f'  {r["pdb"]:6s}  {row}')

    out = os.environ.get('OUT_JSON')
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as fh:
            json.dump({'results': all_r}, fh, indent=2)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
