#!/usr/bin/env python
"""Refit gb_cross calibration with a quadratic form on the same train/test
splits used by the linear fit. Compares linear vs quadratic prediction RMSE
per system.

Uses the JSONs written by jax_gb_cross_calibration_test.py (SLURM 50267).
"""
import glob, json, os
import numpy as np


D = '/home/jtufts/src/p312/algdock/AlGDock/mwe/results/gb_cross_calibration_2026-07-22'
OUT_PLOT = '/home/jtufts/src/p312/algdock/AlGDock/mwe/results/gb_cross_calibration_plots'


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

    rows = []
    for f in files:
        d = json.load(open(f))
        Eg_tr = np.array(d['train']['E_grid'])
        Ef_tr = np.array(d['train']['E_full'])
        Eg_te = np.array(d['test']['E_grid'])
        Ef_te = np.array(d['test']['E_full'])

        # Linear (already fitted)
        m1, c1 = d['train_fit']['slope'], d['train_fit']['intercept']
        pred_lin_te = m1 * Eg_te + c1
        rms_lin_te = float(np.sqrt(np.mean((pred_lin_te - Ef_te) ** 2)))
        pred_lin_tr = m1 * Eg_tr + c1
        rms_lin_tr = float(np.sqrt(np.mean((pred_lin_tr - Ef_tr) ** 2)))

        # Quadratic: E_true ≈ a*Eg^2 + b*Eg + c
        a, b, c = np.polyfit(Eg_tr, Ef_tr, 2)
        pred_q_tr = a * Eg_tr ** 2 + b * Eg_tr + c
        pred_q_te = a * Eg_te ** 2 + b * Eg_te + c
        rms_q_tr = float(np.sqrt(np.mean((pred_q_tr - Ef_tr) ** 2)))
        rms_q_te = float(np.sqrt(np.mean((pred_q_te - Ef_te) ** 2)))
        tau_q = kendall_tau(pred_q_te, Ef_te)
        r_q = float(np.corrcoef(pred_q_te, Ef_te)[0, 1])

        rows.append({
            'pdb': d['pdb'],
            'n_train': len(Eg_tr), 'n_test': len(Eg_te),
            'lin_train_rms': rms_lin_tr, 'lin_test_rms': rms_lin_te,
            'quad_a': a, 'quad_b': b, 'quad_c': c,
            'quad_train_rms': rms_q_tr, 'quad_test_rms': rms_q_te,
            'quad_test_r': r_q, 'quad_test_tau': tau_q,
            'delta_test_rms': rms_q_te - rms_lin_te,
        })

    # Sort by improvement
    rows.sort(key=lambda r: r['delta_test_rms'])

    print(f'{"pdb":6s}  {"lin train":>9s}  {"lin test":>8s}  '
          f'{"quad a":>9s}  {"quad b":>7s}  {"quad c":>8s}  '
          f'{"q train":>7s}  {"q test":>6s}  {"Δ test":>7s}  {"q τ":>5s}')
    print()
    for r in rows:
        print(f'  {r["pdb"]:5s}  {r["lin_train_rms"]:>9.3f}  '
              f'{r["lin_test_rms"]:>8.3f}  '
              f'{r["quad_a"]:>+9.5f}  {r["quad_b"]:>+7.3f}  '
              f'{r["quad_c"]:>+8.2f}  '
              f'{r["quad_train_rms"]:>7.3f}  {r["quad_test_rms"]:>6.3f}  '
              f'{r["delta_test_rms"]:>+7.3f}  {r["quad_test_tau"]:>5.3f}')

    print('\nUnits: kcal/mol.  Δ test = quad_test_rms − lin_test_rms '
          '(negative = quadratic is better).\n')

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()
    for i, r in enumerate(rows):
        d = json.load(open(f'{D}/{r["pdb"]}.json'))
        Eg_tr = np.array(d['train']['E_grid'])
        Ef_tr = np.array(d['train']['E_full'])
        Eg_te = np.array(d['test']['E_grid'])
        Ef_te = np.array(d['test']['E_full'])
        a, b, c = r['quad_a'], r['quad_b'], r['quad_c']
        m1, c1 = d['train_fit']['slope'], d['train_fit']['intercept']

        ax = axes[i]
        lo = min(Eg_tr.min(), Eg_te.min()) - 5
        hi = max(Eg_tr.max(), Eg_te.max()) + 5
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, m1 * xs + c1, 'b-', lw=1.4, alpha=0.7, label='linear')
        ax.plot(xs, a * xs ** 2 + b * xs + c, 'r-', lw=1.4, alpha=0.7,
                label='quadratic')
        ax.scatter(Eg_tr, Ef_tr, s=20, c='grey', alpha=0.5, marker='s',
                   label=f'train (n={len(Eg_tr)})')
        ax.scatter(Eg_te, Ef_te, s=25, c='k', alpha=0.85, marker='o',
                   label=f'test (n={len(Eg_te)})')
        ax.set_title(f'{r["pdb"]}   test RMSE: lin={r["lin_test_rms"]:.2f}, '
                     f'quad={r["quad_test_rms"]:.2f} (Δ {r["delta_test_rms"]:+.2f})',
                     fontsize=9)
        ax.set_xlabel('E_grid (kcal/mol)')
        ax.set_ylabel('E_true (kcal/mol)')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    for j in range(len(rows), len(axes)):
        axes[j].axis('off')
    fig.suptitle('Linear vs quadratic correction per Astex receptor',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_PLOT, exist_ok=True)
    pdf = os.path.join(OUT_PLOT, 'linear_vs_quadratic.pdf')
    png = os.path.join(OUT_PLOT, 'linear_vs_quadratic.png')
    fig.savefig(pdf); fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f'wrote {pdf}\nwrote {png}')


if __name__ == '__main__':
    main()
