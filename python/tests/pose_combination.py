"""Multi-pose Boltzmann-weighted ΔG combination.

Per hessian_plan.md:821-918. The per-pose pipeline (compute_total_entropy_hybrid)
gives the vibrational entropy of ONE basin. With many poses we want a binding
free energy that includes the *configurational* entropy from pose multiplicity:
multiple distinct binding modes accessible via thermal fluctuation contribute
to the partition function via log-sum-exp over their per-pose ΔG values.

Complete-linkage agglomerative clustering on a heavy-atom pairwise-RMSD matrix
groups near-duplicate poses; Boltzmann-weighted log-sum-exp combines clusters.
"""
import numpy as np


def cluster_poses_by_rmsd(pose_geometries_A, rmsd_threshold_A=1.5):
    """Complete-linkage agglomerative clustering on pairwise heavy-atom RMSD.

    Merge the closest pair of clusters at each step until the next merge would
    exceed the threshold; cluster distance is the maximum pairwise distance
    between members (complete linkage). Equivalent to scipy/sklearn complete
    linkage at the same threshold.

    Args:
        pose_geometries_A: list of (N_heavy, 3) arrays in Angstrom.
        rmsd_threshold_A:  linkage cutoff (Å).

    Returns:
        list of clusters, each a sorted list of pose indices.
    """
    n = len(pose_geometries_A)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    geos = [np.asarray(g, dtype=float) for g in pose_geometries_A]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sqrt(np.mean(np.sum((geos[i] - geos[j]) ** 2, axis=1))))
            D[i, j] = D[j, i] = d

    clusters = [{i} for i in range(n)]
    while len(clusters) > 1:
        best = (None, None, np.inf)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                # Complete linkage: maximum pairwise distance between members
                d_ab = max(D[i, j] for i in clusters[a] for j in clusters[b])
                if d_ab < best[2]:
                    best = (a, b, d_ab)
        if best[2] > rmsd_threshold_A:
            break
        a, b, _ = best
        clusters[a] = clusters[a] | clusters[b]
        clusters.pop(b)

    return [sorted(c) for c in clusters]


def combine_poses_boltzmann(pose_dGs_kcal, geometries_A, T_K=300.0,
                              rmsd_threshold_A=1.5):
    """Combine per-pose ΔG values into one binding ΔG including configurational
    entropy from pose multiplicity.

    Cluster poses by RMSD; within a cluster, average ΔG (noisy estimates of the
    same basin); across clusters, Boltzmann-weight with log-sum-exp:
        ΔG_total = ΔG_min − (1/β) ln Σ_clusters exp(−β (ΔG_c − ΔG_min))

    The (ΔG_min − ΔG_total) difference is the configurational-entropy
    contribution (always ≥ 0; equals 0 for a single populated cluster, larger
    for many similar-energy clusters).

    Args:
        pose_dGs_kcal:    array of per-pose ΔG in kcal/mol.
        geometries_A:     list of (N_heavy, 3) arrays in Angstrom.
        T_K:              temperature (K).
        rmsd_threshold_A: clustering cutoff (Å).

    Returns:
        dict with:
          'dG_total_kcal':   total binding ΔG including configurational entropy
          'dG_lowest_kcal':  ΔG of the most stable cluster
          'TS_config_kcal':  configurational-entropy contribution (kcal/mol)
          'N_eff':           effective populated clusters (1/Σpᵢ²)
          'cluster_summaries': list of per-cluster {n_members, mean_dG_kcal}
    """
    dGs = np.asarray(pose_dGs_kcal, dtype=float)
    n = len(dGs)
    if n == 0:
        return {'dG_total_kcal': np.nan, 'dG_lowest_kcal': np.nan,
                'TS_config_kcal': np.nan, 'N_eff': 0, 'cluster_summaries': []}

    clusters = cluster_poses_by_rmsd(geometries_A, rmsd_threshold_A)
    cluster_dGs = np.array([float(np.mean(dGs[c])) for c in clusters])

    KB_KCAL = 1.987204e-3
    beta = 1.0 / (KB_KCAL * T_K)
    dG_min = float(cluster_dGs.min())
    shifted = cluster_dGs - dG_min
    log_sum_exp = float(np.log(np.sum(np.exp(-beta * shifted))))
    dG_total = dG_min - log_sum_exp / beta

    weights = np.exp(-beta * shifted)
    weights /= weights.sum()
    N_eff = float(1.0 / np.sum(weights ** 2))
    TS_config = float(dG_total - dG_min)   # ≤ 0; favorable contribution

    return {
        'dG_total_kcal': float(dG_total),
        'dG_lowest_kcal': dG_min,
        'TS_config_kcal': TS_config,
        'N_eff': N_eff,
        'cluster_summaries': [
            {'n_members': len(c), 'mean_dG_kcal': float(dG)}
            for c, dG in zip(clusters, cluster_dGs)
        ],
    }
