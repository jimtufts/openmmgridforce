#!/usr/bin/env python
"""Physics-informed ML residual: fit  E_true - (m*E_grid + c) = f_θ(features)
where features are per-pose geometric / charge-density descriptors of the
ligand-receptor interface.

Reuses train/test splits + E_full/E_grid from SLURM 50267 calibration JSONs;
loads pose coordinates + prmtop parameters to compute features.

Model options (env var MODEL):
  ridge   — Ridge regression (default; simple, no hyperparameters worth tuning)
  mlp     — small MLP via scikit-learn
"""
import glob, json, os, sys, time
import numpy as np

# Reuse loaders from v2 prototype
sys.path.insert(0, '/home/jtufts/src/p312/openmmgridforce/python/tests')
from jax_gb_cross_grid_v2 import load_prmtop_params, load_dock6_mol2


D = '/home/jtufts/src/p312/algdock/AlGDock/mwe/results/gb_cross_calibration_2026-07-22'
OUT_PLOT = '/home/jtufts/src/p312/algdock/AlGDock/mwe/results/gb_cross_calibration_plots'
ASTEX = '/home/dminh/backup/AstexDiv_xtal'
MODEL = os.environ.get('MODEL', 'ridge')


def per_pose_features(lig_pos, lig_q, lig_r,
                      rec_pos, rec_q, rec_r,
                      cutoffs=(0.3, 0.4, 0.5, 0.7, 1.0)):
    """Compute per-pose geometric + charge-density features.

    lig_pos (N_l, 3), rec_pos (N_r, 3), all in nm.
    Returns 1D feature vector.
    """
    dx = lig_pos[:, None, :] - rec_pos[None, :, :]      # (Nl, Nr, 3)
    r = np.sqrt(np.sum(dx * dx, axis=-1))               # (Nl, Nr)  in nm
    inv_r = 1.0 / np.maximum(r, 0.05)

    feats = []
    # 1. Per-pose contact counts + charge density at multiple cutoffs
    for rc in cutoffs:
        mask = (r < rc).astype(np.float64)                            # (Nl, Nr)
        # per-lig-atom
        per_lig_count = mask.sum(axis=1)                              # (Nl,)
        per_lig_qsum = (mask * rec_q[None, :]).sum(axis=1)             # signed
        per_lig_qabs = (mask * np.abs(rec_q[None, :])).sum(axis=1)
        # per-lig atom, weighted by ligand's own charge
        per_lig_charged = np.abs(lig_q) * per_lig_qabs
        # aggregate across lig atoms
        for arr in (per_lig_count, per_lig_qsum, per_lig_qabs, per_lig_charged):
            feats.extend([arr.sum(), arr.mean(), arr.std(), arr.max()])

    # 2. Near-field electrostatic proxy (independent of cutoff): sum_ij q_i q_j / r
    #    (this is the direct Coulombic magnitude — the grid already knows this,
    #    but including it as a feature helps the network find its right scale)
    signed_ele = np.sum(lig_q[:, None] * rec_q[None, :] * inv_r)
    abs_ele    = np.sum(np.abs(lig_q[:, None]) * np.abs(rec_q[None, :]) * inv_r)
    feats.extend([signed_ele, abs_ele])

    # 3. Ligand pose descriptors: radius of gyration, spread of ligand-side charges
    lig_com = lig_pos.mean(axis=0)
    disp = lig_pos - lig_com
    rg = float(np.sqrt(np.mean(np.sum(disp * disp, axis=1))))
    feats.append(rg)
    feats.append(float(np.abs(lig_q).sum()))            # total polarity
    feats.append(float(np.sum(lig_q * lig_q)))          # sum q_i^2

    # 4. Per-atom nearest receptor distance stats
    min_r = np.min(r, axis=1)  # (Nl,) shortest receptor distance per lig atom
    feats.extend([min_r.min(), min_r.mean(), min_r.max(), np.median(min_r)])

    return np.asarray(feats)


def compute_features(pdb, pose_indices):
    """Load prmtops + poses for a PDB and return (N_poses, N_features) array."""
    lig_pt = f'{ASTEX}/1-build/{pdb}/ligand.prmtop'
    lig_ic = f'{ASTEX}/3-grids/{pdb}/ligand.trans.inpcrd'
    rec_pt = f'{ASTEX}/1-build/{pdb}/receptor.prmtop'
    rec_ic = f'{ASTEX}/3-grids/{pdb}/receptor.trans.inpcrd'
    poses_file = f'{ASTEX}/4-UCSF_dock6/{pdb}/xtal_plus_dock6_scored.mol2'
    _, l_q, l_r, _ = load_prmtop_params(lig_pt, lig_ic)
    r_pos, r_q, r_r, _ = load_prmtop_params(rec_pt, rec_ic)
    poses = load_dock6_mol2(poses_file)
    poses = [p for p in poses if p.shape[0] == len(l_q)]
    feats = []
    for idx in pose_indices:
        f = per_pose_features(poses[idx], l_q, l_r, r_pos, r_q, r_r)
        feats.append(f)
    return np.asarray(feats)


def fit_and_evaluate(X_tr, y_tr, X_te, y_te, model):
    """Fit residual model on train, return (train_rms, test_rms, y_te_pred)."""
    if model == 'ridge':
        from sklearn.linear_model import RidgeCV
        pipeline = _make_pipeline(RidgeCV(alphas=np.logspace(-4, 4, 20)))
    elif model == 'mlp':
        from sklearn.neural_network import MLPRegressor
        pipeline = _make_pipeline(MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=5000, random_state=42, early_stopping=True,
            validation_fraction=0.2))
    else:
        raise ValueError(model)
    pipeline.fit(X_tr, y_tr)
    y_tr_pred = pipeline.predict(X_tr)
    y_te_pred = pipeline.predict(X_te)
    tr_rms = float(np.sqrt(np.mean((y_tr_pred - y_tr) ** 2)))
    te_rms = float(np.sqrt(np.mean((y_te_pred - y_te) ** 2)))
    return tr_rms, te_rms, y_te_pred


def _make_pipeline(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([('scaler', StandardScaler()), ('est', estimator)])


def kendall_tau(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (n * (n - 1) / 2) if n >= 2 else float('nan')


def main():
    files = sorted(glob.glob(f'{D}/*.json'))
    if not files:
        raise SystemExit('no calibration JSONs found')

    print(f'Model: {MODEL}\n')
    rows = []
    for f in files:
        d = json.load(open(f))
        pdb = d['pdb']
        train_idx = d['train_idx']
        test_idx = d['test_idx']
        m, c = d['train_fit']['slope'], d['train_fit']['intercept']

        Eg_tr = np.array(d['train']['E_grid'])
        Ef_tr = np.array(d['train']['E_full'])
        Eg_te = np.array(d['test']['E_grid'])
        Ef_te = np.array(d['test']['E_full'])
        Ep_te_lin = m * Eg_te + c
        Ep_tr_lin = m * Eg_tr + c
        rms_lin_te = float(np.sqrt(np.mean((Ep_te_lin - Ef_te) ** 2)))

        # Residuals to predict
        r_tr = Ef_tr - Ep_tr_lin
        r_te = Ef_te - (m * Eg_te + c)

        # Pose features (loads prmtops + mol2 per PDB)
        t0 = time.time()
        X_tr = compute_features(pdb, train_idx)
        X_te = compute_features(pdb, test_idx)
        t_feat = time.time() - t0

        tr_rms, te_rms, r_te_pred = fit_and_evaluate(X_tr, r_tr, X_te, r_te,
                                                     MODEL)

        # Combined prediction: linear + ML residual
        Ep_te_ml = Ep_te_lin + r_te_pred
        combined_te_rms = float(np.sqrt(np.mean((Ep_te_ml - Ef_te) ** 2)))
        tau = kendall_tau(Ep_te_ml, Ef_te)

        rows.append({
            'pdb': pdb, 'n_feat': X_tr.shape[1],
            'lin_test_rms': rms_lin_te,
            'ml_resid_train_rms': tr_rms,
            'ml_resid_test_rms': te_rms,
            'combined_test_rms': combined_te_rms,
            'delta_vs_linear': combined_te_rms - rms_lin_te,
            'kendall_tau': tau,
            't_feat': t_feat,
        })

        # Save per-system JSON with predictions for plotting
        out_dir = os.environ.get(
            'ML_OUT',
            f'/home/jtufts/src/p312/algdock/AlGDock/mwe/results/gb_cross_ml_{MODEL}')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'{pdb}.json'), 'w') as fh:
            json.dump({
                'pdb': pdb, 'model': MODEL, 'n_features': int(X_tr.shape[1]),
                'linear_fit': {'slope': float(m), 'intercept': float(c)},
                'train': {
                    'E_full': Ef_tr.tolist(), 'E_grid': Eg_tr.tolist(),
                    'lin_pred': Ep_tr_lin.tolist(),
                },
                'test': {
                    'E_full': Ef_te.tolist(), 'E_grid': Eg_te.tolist(),
                    'lin_pred': Ep_te_lin.tolist(),
                    'ml_pred': Ep_te_ml.tolist(),
                    'lin_rms': rms_lin_te,
                    'ml_rms': combined_te_rms,
                    'kendall_tau': tau,
                },
            }, fh, indent=2)

    rows.sort(key=lambda r: r['delta_vs_linear'])
    print(f'{"pdb":6s}  {"n_feat":>6s}  {"lin RMS":>7s}  '
          f'{"ML resid tr":>11s}  {"ML resid te":>11s}  '
          f'{"lin+ML te":>9s}  {"Δ vs lin":>8s}  {"τ":>5s}')
    print()
    for r in rows:
        print(f'  {r["pdb"]:5s}  {r["n_feat"]:>6d}  '
              f'{r["lin_test_rms"]:>7.2f}  '
              f'{r["ml_resid_train_rms"]:>11.2f}  '
              f'{r["ml_resid_test_rms"]:>11.2f}  '
              f'{r["combined_test_rms"]:>9.2f}  '
              f'{r["delta_vs_linear"]:>+8.2f}  '
              f'{r["kendall_tau"]:>5.2f}')

    print('\nUnits: kcal/mol.  '
          'Δ vs lin  = combined_test_rms − lin_test_rms  '
          '(negative = ML residual improved over linear alone)\n')


if __name__ == '__main__':
    main()
