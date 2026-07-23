#!/usr/bin/env python
"""Diagnostic decomposition: why does plugin-vs-vanilla diff vary by pose
on 1t46 but stay constant on 1r1h and 1jje?

For each pose, extract every intermediate the plugin exposes:
  E_lig_self  (getGroupLigandSelfEnergy)
  E_cross     (getGroupCrossTermEnergy)
  E_rec_del   (getGroupReceptorDesolvation)
  E_rec_contr (getGroupReceptorContribution)  ← the receptor-descreening part
  ligand Born radii  (getGroupBornRadii)
  receptor Born radii (getReceptorBornRadii)

And from vanilla OpenMM on the FULL system extract:
  E_total_vanilla
  Born radii for every atom  (recomputed manually via HCT, since GBSAOBCForce
                              doesn't expose them; we do it in numpy)

Compare per-pose which pieces vary in an unexpected way.

Env:
  PDB=1t46
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

PDB = os.environ.get('PDB', '1t46')
N_POSES = int(os.environ.get('N_POSES', '5'))
EVAL_START = int(os.environ.get('EVAL_START', '60'))
ASTEX = '/home/dminh/backup/AstexDiv_xtal'

DIELECTRIC_OFFSET = 0.009  # nm  (same as OBC)
OBC_ALPHA = 1.0
OBC_BETA = 0.8
OBC_GAMMA = 4.85


def numpy_hct_born(positions, radii, scales):
    """Vectorized HCT-integral Born radii (OBC-II). No cutoff.
    Formula matches ReferenceObc.cpp:158-208. Runs in seconds for N~5000.
    """
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    n = positions.shape[0]
    r_off = np.maximum(radii - DIELECTRIC_OFFSET, 1e-6)
    S = r_off * scales

    # (n, n) matrices
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.sqrt((diff * diff).sum(axis=-1))
    # avoid diagonal / self-pairs
    valid = (r > 1e-10) & (~np.eye(n, dtype=bool))

    offsetRi = r_off[:, None]        # (n, 1)
    scaledRj = S[None, :]            # (1, n)
    rScaled = r + scaledRj
    # active mask: pair contributes
    active = valid & (offsetRi < rScaled)

    r_safe = np.where(valid, r, 1.0)
    rInv = 1.0 / r_safe
    l_ij_denom = np.maximum(offsetRi, np.abs(r - scaledRj))
    l_ij_denom = np.where(l_ij_denom < 1e-12, 1e-12, l_ij_denom)
    l_ij = 1.0 / l_ij_denom
    u_ij = 1.0 / rScaled
    l_ij2 = l_ij * l_ij
    u_ij2 = u_ij * u_ij
    ratio = np.log(np.where(active, u_ij / np.maximum(l_ij, 1e-30), 1.0))
    term = (l_ij - u_ij + 0.25 * r * (u_ij2 - l_ij2)
            + 0.5 * rInv * ratio
            + 0.25 * scaledRj * scaledRj * rInv * (l_ij2 - u_ij2))

    # inside case: offsetRi < scaledRj - r
    inside = active & (offsetRi < (scaledRj - r))
    radiusIinv = 1.0 / r_off  # (n,)
    term = np.where(inside, term + 2.0 * (radiusIinv[:, None] - l_ij), term)

    term = np.where(active, term, 0.0)
    sum_ = term.sum(axis=1)

    sum_ *= 0.5 * r_off
    tanhArg = OBC_ALPHA * sum_ - OBC_BETA * sum_**2 + OBC_GAMMA * sum_**3
    born = 1.0 / (1.0 / r_off - np.tanh(tanhArg) / radii)
    return born


def setup_vanilla(lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s):
    n_lig = len(lig_q); n_rec = len(rec_q)
    system = mm.System()
    for _ in range(n_lig + n_rec):
        system.addParticle(12.0)
    gb = mm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setSurfaceAreaEnergy(0.0)
    gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
    for i in range(n_lig):
        gb.addParticle(float(lig_q[i]), float(lig_r[i]), float(lig_s[i]))
    for i in range(n_rec):
        gb.addParticle(float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    system.addForce(gb)
    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integrator, platform, {'Precision': 'double'})
    return ctx, n_lig, n_rec


def setup_vanilla_receptor_only(rec_pos, rec_q, rec_r, rec_s):
    """Vanilla OpenMM GBSAOBCForce with ONLY receptor atoms."""
    n_rec = len(rec_q)
    system = mm.System()
    for _ in range(n_rec):
        system.addParticle(12.0)
    gb = mm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setSurfaceAreaEnergy(0.0)
    gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
    for i in range(n_rec):
        gb.addParticle(float(rec_q[i]), float(rec_r[i]), float(rec_s[i]))
    system.addForce(gb)
    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integrator, platform, {'Precision': 'double'})
    return ctx


def setup_plugin_pairwise(lig_q, lig_r, lig_s, rec_pos, rec_q, rec_r, rec_s):
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
    f.addParticleGroup('ligand', list(range(n_lig)))
    f.setGroupScalingFactor(0, 1.0)
    # setDownloadBornRadii(True) crashes under double precision — see plugin bug.
    system.addForce(f)
    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integrator, platform, {'Precision': 'double'})
    return f, ctx


def main():
    print(f'== 1t46 decomposition diagnostic ==')
    lig_pt = f'{ASTEX}/1-build/{PDB}/ligand.prmtop'
    lig_ic = f'{ASTEX}/3-grids/{PDB}/ligand.trans.inpcrd'
    rec_pt = f'{ASTEX}/1-build/{PDB}/receptor.prmtop'
    rec_ic = f'{ASTEX}/3-grids/{PDB}/receptor.trans.inpcrd'
    poses_file = f'{ASTEX}/4-UCSF_dock6/{PDB}/xtal_plus_dock6_scored.mol2'
    _, l_q, l_r, l_s = load_prmtop_params(lig_pt, lig_ic)
    r_pos, r_q, r_r, r_s = load_prmtop_params(rec_pt, rec_ic)
    all_poses = [p for p in load_dock6_mol2(poses_file) if p.shape[0] == len(l_q)]
    poses = all_poses[EVAL_START:EVAL_START + N_POSES]
    print(f'  N_lig={len(l_q)}  N_rec={len(r_q)}  poses={len(poses)}')

    # ---- vanilla receptor-only baseline: pose-independent, compute ONCE ----
    ctx_vr = setup_vanilla_receptor_only(r_pos, r_q, r_r, r_s)
    ctx_vr.setPositions(np.asarray(r_pos) * unit.nanometer)
    E_v_rec_only = ctx_vr.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    del ctx_vr
    print(f'  E_vanilla_receptor_only = {E_v_rec_only:+.4f} kJ/mol (pose-independent)',
          flush=True)

    per_pose = []
    for pk, pose in enumerate(poses):
        pose_arr = np.asarray(pose)
        # Vanilla full
        ctx_v, n_lig, n_rec = setup_vanilla(l_q, l_r, l_s, r_pos, r_q, r_r, r_s)
        full = np.vstack([pose_arr, np.asarray(r_pos)])
        ctx_v.setPositions(full * unit.nanometer)
        E_v = ctx_v.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        del ctx_v

        # Plugin PAIRWISE (cutoff off)
        f, ctx_p = setup_plugin_pairwise(l_q, l_r, l_s, r_pos, r_q, r_r, r_s)
        ctx_p.setPositions(pose_arr * unit.nanometer)
        E_p = ctx_p.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Decomposition
        E_lig_self = float(f.getGroupLigandSelfEnergy(0))
        E_cross    = float(f.getGroupCrossTermEnergy(0))
        E_rec_del  = float(f.getGroupReceptorDesolvation(0))
        E_rec_ctr  = float(f.getGroupReceptorContribution(0))
        recon = E_lig_self + E_cross + E_rec_del
        del ctx_p

        # Numpy HCT-Born-radii reference — vectorized, ~seconds for N=4808.
        # Compute per pose so we can see whether the plugin's Born radii
        # deviate more per-pose than vanilla's (which would explain the
        # per-pose delta in the E_v - E_p offset).
        t0 = time.time()
        radii_all = np.concatenate([l_r, r_r])
        scales_all = np.concatenate([l_s, r_s])
        born_ref = numpy_hct_born(full, radii_all, scales_all)
        born_lig_ref = born_ref[:n_lig]
        born_rec_ref = born_ref[n_lig:]
        # GB energy from vanilla Born radii (numpy) for cross-check
        prefactor = -138.935485 * (1.0/1.0 - 1.0/78.5)  # kJ/mol/nm units
        q_all = np.concatenate([l_q, r_q])
        # Full N^2 GB energy for cross-check
        dx = full[:, None, :] - full[None, :, :]
        r2 = (dx * dx).sum(axis=-1)
        Ri = born_ref[:, None]; Rj = born_ref[None, :]
        RiRj = Ri * Rj
        f_gb = np.sqrt(r2 + RiRj * np.exp(-r2 / (4.0 * RiRj + 1e-30)))
        np.fill_diagonal(f_gb, 1.0)
        qq = q_all[:, None] * q_all[None, :]
        # Include self (i==j) — GB self term uses R_i = born radius
        # Standard: E = 0.5 sum_i q_i^2 * prefactor / R_i + sum_{i<j} q_i q_j prefactor / f_gb
        E_gb_pairs = 0.5 * prefactor * (qq / f_gb).sum()
        E_gb_self = 0.5 * prefactor * (q_all * q_all / born_ref).sum()
        E_gb_ref = E_gb_pairs  # already includes self (i==j) term via qq/born
        t_hct = time.time() - t0

        # THE key comparison: vanilla binding-like quantity vs plugin's E_p.
        delta_v = E_v - E_v_rec_only
        gap = E_p - delta_v
        # Also compute a stripped diff (no numpy reference)
        diff_v_p = E_v - E_p
        print(f'  pose {pk}: E_v={E_v:+.3f}  E_p={E_p:+.3f}  '
              f'delta_v=E_v-E_v_rec={delta_v:+.3f}  '
              f'gap=E_p-delta_v={gap:+.5f}',
              flush=True)
        print(f'    decomp: E_lig_self={E_lig_self:+.3f}  E_cross={E_cross:+.3f}  '
              f'E_rec_del={E_rec_del:+.3f}  E_rec_ctr={E_rec_ctr:+.3f}',
              flush=True)
        per_pose.append(dict(pose=pk + EVAL_START, E_v=E_v, E_p=E_p,
                              delta_v=delta_v, gap=gap,
                              diff_v_p=diff_v_p,
                              E_lig_self=E_lig_self, E_cross=E_cross,
                              E_rec_del=E_rec_del, E_rec_ctr=E_rec_ctr,
                              recon=recon))

    print()
    print(f'{"pose":>4s}  {"E_v":>12s}  {"E_p":>10s}  {"delta_v":>10s}  '
          f'{"gap":>10s}  {"E_lig_self":>11s}  {"E_cross":>10s}  {"E_rec_del":>10s}')
    for p in per_pose:
        print(f'  {p["pose"]:>3d}   {p["E_v"]:>11.3f}   {p["E_p"]:>+9.3f}  '
              f'{p["delta_v"]:>+9.3f}  {p["gap"]:>+9.5f}  '
              f'{p["E_lig_self"]:>+10.3f}  '
              f'{p["E_cross"]:>+9.3f}  {p["E_rec_del"]:>+9.3f}')
    print()
    gaps = np.array([p['gap'] for p in per_pose])
    print(f'  gap (E_p - delta_v) per pose:  min={gaps.min():+.4f}  '
          f'max={gaps.max():+.4f}  span={gaps.max()-gaps.min():.4f} kJ/mol')

    out = os.environ.get('OUT_JSON')
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as fh:
            json.dump({'per_pose': per_pose}, fh, indent=2)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
