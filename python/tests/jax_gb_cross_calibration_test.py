#!/usr/bin/env python
"""Train/test evaluation of the linear correction on a target.

Loads up to N_POSES real docked poses per system.  Randomly splits into a
training subset of size N_TRAIN and a held-out test set.  Fits the linear
correction  E_full ~= m*E_grid + c  on training poses only.  Applies to
test poses.  Reports:
  - training slope, intercept, r
  - test residual RMS (predicted - true)
  - test Kendall tau (rank concordance) before and after correction

If the correction generalizes within a target, test RMS should be similar
to training residual RMS and rank correlation should stay high.
"""
import os, sys, json, time
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
    cross_energy_atomistic_vec,
    cross_energy_grid_vec,
    load_prmtop_params,
    load_dock6_mol2,
)


def kendall_tau(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / (n * (n - 1) / 2) if n >= 2 else float('nan')


def main():
    pdb = os.environ.get('PDB', '1r1h')
    astex = '/home/dminh/backup/AstexDiv_xtal'
    lig_pt = f'{astex}/1-build/{pdb}/ligand.prmtop'
    lig_ic = f'{astex}/3-grids/{pdb}/ligand.trans.inpcrd'
    rec_pt = f'{astex}/1-build/{pdb}/receptor.prmtop'
    rec_ic = f'{astex}/3-grids/{pdb}/receptor.trans.inpcrd'
    poses_file = f'{astex}/4-UCSF_dock6/{pdb}/xtal_plus_dock6_scored.mol2'
    n_train = int(os.environ.get('N_TRAIN', '30'))
    n_test = int(os.environ.get('N_TEST', '30'))
    seed = int(os.environ.get('SEED', '42'))

    print(f'PDB: {pdb}  n_train: {n_train}  n_test: {n_test}  seed: {seed}')
    l_pos, l_q, l_r, l_s = load_prmtop_params(lig_pt, lig_ic)
    r_pos, r_q, r_r, r_s = load_prmtop_params(rec_pt, rec_ic)
    all_poses = [p for p in load_dock6_mol2(poses_file)
                 if p.shape[0] == len(l_q)]
    N_l, N_r = len(l_q), len(r_q)
    print(f'  N_lig = {N_l}   N_rec = {N_r}   {len(all_poses)} poses available')

    if len(all_poses) < n_train + n_test:
        print(f'  WARN: only {len(all_poses)} poses; reducing splits')
        n_avail = len(all_poses)
        n_train = min(n_train, n_avail // 2)
        n_test = min(n_test, n_avail - n_train)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_poses))
    train_idx = perm[:n_train].tolist()
    test_idx = perm[n_train:n_train + n_test].tolist()
    print(f'  train indices: {train_idx[:8]}{"..." if len(train_idx)>8 else ""}')
    print(f'  test  indices: {test_idx[:8]}{"..." if len(test_idx)>8 else ""}\n')

    lq_j = jnp.array(l_q); lr_j = jnp.array(l_r); ls_j = jnp.array(l_s)
    rp_j = jnp.array(r_pos); rq_j = jnp.array(r_q)
    rr_j = jnp.array(r_r); rs_j = jnp.array(r_s)

    print('Frozen receptor Born radii...', flush=True)
    t0 = time.time()
    R_j_frozen = receptor_born_radii_isolated(rp_j, rr_j, rs_j)
    R_j_frozen.block_until_ready()
    print(f'  {time.time()-t0:.1f}s.  R_j range {float(R_j_frozen.min()):.3f}-{float(R_j_frozen.max()):.3f} nm')

    all_pose_arr = np.vstack([all_poses[i] for i in (train_idx + test_idx)])
    margin = 0.5
    box_lo = (all_pose_arr.min(axis=0) - margin).tolist()
    box_hi = (all_pose_arr.max(axis=0) + margin).tolist()
    R_i_crystal = ligand_born_radii_with_receptor(
        jnp.array(l_pos), lr_j, ls_j, rp_j, rr_j, rs_j)
    probe_radii = np.linspace(float(R_i_crystal.min()) * 0.85,
                              float(R_i_crystal.max()) * 1.15, 6)
    print('Building G2 (full receptor)...', flush=True)
    t0 = time.time()
    Phi_stack, meta = build_g2_grid(rp_j, rq_j, R_j_frozen,
                                     box_lo, box_hi, 0.05, probe_radii,
                                     chunk=1024)
    print(f'  {time.time()-t0:.1f}s\n')

    def energies_for(poses_idx):
        E_full = np.zeros(len(poses_idx))
        E_grid = np.zeros(len(poses_idx))
        for k, idx in enumerate(poses_idx):
            lp_j = jnp.array(all_poses[idx])
            R_i = ligand_born_radii_with_receptor(lp_j, lr_j, ls_j,
                                                    rp_j, rr_j, rs_j)
            R_j = receptor_born_radii_with_ligand(rp_j, rr_j, rs_j,
                                                    lp_j, lr_j, ls_j)
            E_full[k] = float(cross_energy_atomistic_vec(
                lp_j, lq_j, R_i, rp_j, rq_j, R_j)) / 4.184
            E_grid[k] = float(cross_energy_grid_vec(
                lp_j, lq_j, R_i, Phi_stack, meta)) / 4.184
        return E_full, E_grid

    print('Train set energies...', flush=True)
    t0 = time.time()
    E_full_tr, E_grid_tr = energies_for(train_idx)
    print(f'  {time.time()-t0:.1f}s')

    print('Test set energies...', flush=True)
    t0 = time.time()
    E_full_te, E_grid_te = energies_for(test_idx)
    print(f'  {time.time()-t0:.1f}s\n')

    # Fit linear correction on TRAIN
    m, c = np.polyfit(E_grid_tr, E_full_tr, 1)
    r_train = float(np.corrcoef(E_grid_tr, E_full_tr)[0, 1])
    resid_train = E_full_tr - (m * E_grid_tr + c)
    rms_train = float(np.sqrt(np.mean(resid_train ** 2)))
    print(f'=== TRAIN fit (n={len(train_idx)}) ===')
    print(f'  E_full = {m:.4f} * E_grid + {c:+.4f}')
    print(f'  Pearson r = {r_train:.4f}   fit residual RMS = {rms_train:.3f} kcal/mol')

    # Apply to TEST
    E_full_pred = m * E_grid_te + c
    resid_test = E_full_te - E_full_pred
    rms_test = float(np.sqrt(np.mean(resid_test ** 2)))
    r_test_raw = float(np.corrcoef(E_grid_te, E_full_te)[0, 1])
    r_test_corr = float(np.corrcoef(E_full_pred, E_full_te)[0, 1])
    tau_raw = kendall_tau(E_grid_te, E_full_te)
    tau_corr = kendall_tau(E_full_pred, E_full_te)
    print(f'\n=== TEST (n={len(test_idx)}) ===')
    print(f'  raw grid vs true:  Pearson r = {r_test_raw:.4f}, Kendall τ = {tau_raw:.4f}')
    print(f'  corrected vs true: Pearson r = {r_test_corr:.4f}, Kendall τ = {tau_corr:.4f}')
    print(f'  prediction residual RMS = {rms_test:.3f} kcal/mol')
    print(f'  train RMS was = {rms_train:.3f} kcal/mol')
    print(f'  overfitting gap = {rms_test - rms_train:+.3f} kcal/mol')

    out = os.environ.get('OUTPUT_JSON')
    if out:
        s = {
            'pdb': pdb,
            'N_lig': int(N_l), 'N_rec': int(N_r),
            'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
            'seed': int(seed),
            'train_idx': train_idx, 'test_idx': test_idx,
            'train_fit': {
                'slope': float(m), 'intercept': float(c),
                'pearson_r': r_train, 'residual_rms': rms_train,
            },
            'test': {
                'raw_pearson_r': r_test_raw,
                'raw_kendall_tau': tau_raw,
                'corrected_pearson_r': r_test_corr,
                'corrected_kendall_tau': tau_corr,
                'prediction_rms': rms_test,
                'overfitting_gap': rms_test - rms_train,
                'E_full': E_full_te.tolist(),
                'E_grid': E_grid_te.tolist(),
                'E_full_predicted': E_full_pred.tolist(),
            },
            'train': {
                'E_full': E_full_tr.tolist(),
                'E_grid': E_grid_tr.tolist(),
            },
        }
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(s, fh, indent=2)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
