#!/usr/bin/env python
"""Speed benchmark: grid G2 cross-term lookup vs atomistic pairwise cross term.

Both evaluated in the same JAX framework so numbers are apples-to-apples.
Reuses build code from jax_gb_cross_grid_v2.py.

Times, per pose:
  T1 = cross_energy_grid_vec         (G2 trilinear-linear-in-R lookup)
  T2 = cross_energy_atomistic_vec    (N_lig × N_rec pair sum)
  T3 = cross_energy_atomistic_vec, with per-pose R_i & R_j recompute
        (this is the true fully-coupled "pairwise" pipeline the plugin runs)
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, '/home/jtufts/src/p312/openmmgridforce/python/tests')
from jax_gb_cross_grid_v2 import (
    receptor_born_radii_isolated,
    receptor_born_radii_with_ligand,
    ligand_born_radii_with_receptor,
    build_g2_grid,
    phi_gb_lookup,
    cross_energy_atomistic_vec,
    cross_energy_grid_vec,
    load_prmtop_params,
    load_dock6_mol2,
)


def main():
    pdb = os.environ.get('PDB', '1r1h')
    astex = '/home/dminh/backup/AstexDiv_xtal'
    lig_pt = f'{astex}/1-build/{pdb}/ligand.prmtop'
    lig_ic = f'{astex}/3-grids/{pdb}/ligand.trans.inpcrd'
    rec_pt = f'{astex}/1-build/{pdb}/receptor.prmtop'
    rec_ic = f'{astex}/3-grids/{pdb}/receptor.trans.inpcrd'
    poses_file = f'{astex}/4-UCSF_dock6/{pdb}/xtal_plus_dock6_scored.mol2'
    n_poses = int(os.environ.get('N_POSES', '20'))
    n_reps = int(os.environ.get('N_REPS', '20'))

    print(f'PDB: {pdb}  n_poses: {n_poses}  n_reps: {n_reps}')
    l_pos, l_q, l_r, l_s = load_prmtop_params(lig_pt, lig_ic)
    r_pos, r_q, r_r, r_s = load_prmtop_params(rec_pt, rec_ic)
    poses = [p for p in load_dock6_mol2(poses_file)[:n_poses]
             if p.shape[0] == len(l_q)]
    N_l, N_r = len(l_q), len(r_q)
    print(f'  N_lig = {N_l}   N_rec = {N_r}   loaded {len(poses)} poses\n')

    lq_j = jnp.array(l_q); lr_j = jnp.array(l_r); ls_j = jnp.array(l_s)
    rp_j = jnp.array(r_pos); rq_j = jnp.array(r_q)
    rr_j = jnp.array(r_r); rs_j = jnp.array(r_s)

    # One-time offline: frozen receptor Born radii + G2
    print('[offline, once per receptor]', flush=True)
    t0 = time.time()
    R_j_frozen = receptor_born_radii_isolated(rp_j, rr_j, rs_j)
    R_j_frozen.block_until_ready()
    t_frozen = time.time() - t0
    print(f'  frozen R_j:  {t_frozen:.1f} s')

    all_pos = np.vstack(poses)
    margin = 0.5
    box_lo = (all_pos.min(axis=0) - margin).tolist()
    box_hi = (all_pos.max(axis=0) + margin).tolist()
    spacing = 0.05
    R_i_crystal = ligand_born_radii_with_receptor(
        jnp.array(l_pos), lr_j, ls_j, rp_j, rr_j, rs_j)
    probe_radii = np.linspace(float(R_i_crystal.min()) * 0.85,
                              float(R_i_crystal.max()) * 1.15, 6)
    t0 = time.time()
    Phi_stack, meta = build_g2_grid(rp_j, rq_j, R_j_frozen,
                                     box_lo, box_hi, spacing, probe_radii,
                                     chunk=1024)
    Phi_stack.block_until_ready()
    t_g2 = time.time() - t0
    print(f'  G2 build  :  {t_g2:.1f} s')
    print(f'  ONE-TIME cost: {t_frozen + t_g2:.1f} s (frozen R_j + G2)\n')

    # JIT-compile per-pose evaluators
    print('[JIT warmup]', flush=True)
    lp_j = jnp.array(poses[0])

    # Path 1: grid cross energy (assumes R_i already known)
    R_i_probe = ligand_born_radii_with_receptor(lp_j, lr_j, ls_j,
                                                  rp_j, rr_j, rs_j)
    jgrid = jax.jit(lambda lp, li, R: cross_energy_grid_vec(
        lp, li, R, Phi_stack, meta))
    _ = jgrid(lp_j, lq_j, R_i_probe).block_until_ready()

    # Path 2: atomistic cross sum with GIVEN R_i, R_j (just the pair sum)
    jatom = jax.jit(cross_energy_atomistic_vec)
    _ = jatom(lp_j, lq_j, R_i_probe, rp_j, rq_j, R_j_frozen).block_until_ready()

    # Path 3: full pairwise pipeline for cross term
    #   compute R_i(with rec)  +  compute R_j(with lig)  +  cross sum
    def full_pairwise_cross(lp, lq, lr, ls, rp, rq, rr, rs):
        R_i = ligand_born_radii_with_receptor(lp, lr, ls, rp, rr, rs)
        R_j = receptor_born_radii_with_ligand(rp, rr, rs, lp, lr, ls)
        return cross_energy_atomistic_vec(lp, lq, R_i, rp, rq, R_j)
    jfull = jax.jit(full_pairwise_cross)
    _ = jfull(lp_j, lq_j, lr_j, ls_j, rp_j, rq_j, rr_j, rs_j).block_until_ready()
    print('  JIT warmup done\n', flush=True)

    # Path 4: full pairwise cross WITHOUT the R_j recompute (uses frozen R_j).
    # Represents "you cache R_j via G1 style, only reduce lig-rec pair sum".
    def pw_with_frozen_Rj(lp, lq, lr, ls, rp, rq, rr, rs, R_j_frz):
        R_i = ligand_born_radii_with_receptor(lp, lr, ls, rp, rr, rs)
        return cross_energy_atomistic_vec(lp, lq, R_i, rp, rq, R_j_frz)
    jfrozRj = jax.jit(pw_with_frozen_Rj)
    _ = jfrozRj(lp_j, lq_j, lr_j, ls_j, rp_j, rq_j, rr_j, rs_j,
                 R_j_frozen).block_until_ready()

    # -------- benchmark loops --------
    def time_it(fn, args, n_reps, label):
        # warmup one more
        _ = fn(*args).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = fn(*args).block_until_ready()
        dt = (time.perf_counter() - t0) / n_reps
        print(f'  {label:44s}  {dt*1e3:8.3f} ms/pose   ({1.0/dt:6.1f} poses/s)')
        return dt

    print(f'[per-pose timing, n_reps={n_reps}, all JITted]\n')

    # Prepare per-pose data
    Ri_pose = [ligand_born_radii_with_receptor(jnp.array(p), lr_j, ls_j,
                                                 rp_j, rr_j, rs_j)
               for p in poses]

    # Loop over poses; time the WHOLE loop, average over poses × reps
    def bench_loop(label, fn_and_args_for_pose):
        # warmup
        for p_idx in range(min(3, len(poses))):
            _ = fn_and_args_for_pose(p_idx).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            for p_idx in range(len(poses)):
                _ = fn_and_args_for_pose(p_idx).block_until_ready()
        dt = (time.perf_counter() - t0) / (n_reps * len(poses))
        print(f'  {label:44s}  {dt*1e3:8.3f} ms/pose   ({1.0/dt:6.1f} poses/s)')
        return dt

    # 1. Grid path (given R_i)
    t1 = bench_loop('grid lookup only (R_i already known)',
                    lambda i: jgrid(jnp.array(poses[i]), lq_j, Ri_pose[i]))
    # 2. Atomistic pair sum only (given both R_i and R_j)
    t2 = bench_loop('atomistic pair sum (R_i, R_j given)',
                    lambda i: jatom(jnp.array(poses[i]), lq_j, Ri_pose[i],
                                     rp_j, rq_j, R_j_frozen))
    # 3. Full pairwise: recompute R_i AND R_j per pose
    t3 = bench_loop('FULL pairwise (recompute R_i, R_j, sum)',
                    lambda i: jfull(jnp.array(poses[i]), lq_j, lr_j, ls_j,
                                     rp_j, rq_j, rr_j, rs_j))
    # 4. Pairwise with frozen R_j (recompute R_i only, use frozen R_j)
    t4 = bench_loop('pairwise + frozen R_j (recompute R_i, sum)',
                    lambda i: jfrozRj(jnp.array(poses[i]), lq_j, lr_j, ls_j,
                                       rp_j, rq_j, rr_j, rs_j, R_j_frozen))

    print()
    print('=== speedups relative to full pairwise (path 3) ===')
    print(f'  grid lookup only          : {t3/t1:8.1f}x')
    print(f'  atomistic pair sum only   : {t3/t2:8.1f}x')
    print(f'  pairwise with frozen R_j  : {t3/t4:8.1f}x')

    print()
    print('=== notes ===')
    print('  Path 1 (grid) assumes R_i already comes from G1 (existing grid infra).')
    print('  Path 3 is the honest baseline for what our plugin computes today.')
    print('  Path 4 shows the benefit of frozen R_j alone (no G2), for reference.')
    print('  Real CUDA speedup will differ (JAX-vs-plugin overhead is not identical).')


if __name__ == '__main__':
    main()
