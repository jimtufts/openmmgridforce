"""Fourier DVR for 1D periodic Schrödinger problems (hindered rotors).

Lifted from AlGDock/mwe/toys/toys/toy_entropy_benchmark.py — validated to
machine precision on the 3-fold cosine toy.

Reference: Light & Carrington, Adv. Chem. Phys. 114, 263 (2000), Table II.
"""
import numpy as np

HBAR = 1.054571817e-34   # J·s
KB = 1.380649e-23        # J/K


def fourier_dvr_eigenvalues(V_func, I_r_kg_m2, N=128):
    """Solve [-ℏ²/(2I) d²/dφ² + V(φ)] ψ = E ψ on [0, 2π) by Fourier DVR.

    Args:
        V_func:    callable φ (rad) → V (J). Periodic on [0, 2π).
        I_r_kg_m2: reduced moment of inertia (kg·m²).
        N:         number of evenly-spaced grid points (even is fine).

    Returns:
        sorted eigenvalues E_n in Joules.
    """
    phi = 2 * np.pi * np.arange(N) / N
    prefactor = HBAR ** 2 / (2 * I_r_kg_m2)
    # Diagonal: (N²+2)/12 for even N, (N²-1)/12 for odd
    if N % 2 == 0:
        diag_val = prefactor * (N ** 2 + 2) / 12.0
    else:
        diag_val = prefactor * (N ** 2 - 1) / 12.0
    # Off-diagonal: prefactor * (-1)^(i-j) / (2 sin²(π(i-j)/N))
    idx = np.arange(N)
    di = idx[:, None] - idx[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        offdiag = prefactor * (-1.0) ** di / (2 * np.sin(np.pi * di / N) ** 2)
    T = np.where(di == 0, diag_val, offdiag)
    V_diag = np.array([V_func(p) for p in phi])
    H = T + np.diag(V_diag)
    return np.linalg.eigvalsh(H)


def entropy_dvr(V_func, I_r_kg_m2, T_K, N=None, N_safety=8):
    """Quantum 1D hindered-rotor entropy via Fourier DVR (in k_B units).

    Adaptively chooses N based on rotational constant and temperature; the
    grid must resolve momentum modes up to m_max ~ sqrt(kT/B).

    Args:
        V_func:     callable φ (rad) → V (J).
        I_r_kg_m2:  reduced moment of inertia (kg·m²).
        T_K:        temperature (K).
        N:          grid size; if None, choose adaptively (min 64, max 2048).
        N_safety:   safety factor for adaptive N (default 8 momentum modes/m_max).

    Returns:
        S in k_B units.
    """
    if N is None:
        B = HBAR ** 2 / (2 * I_r_kg_m2)
        kT = KB * T_K
        m_max = np.sqrt(kT / B) if B > 0 else 1.0
        N = int(np.ceil(N_safety * m_max))
        N = max(64, min(N, 2048))
        if N % 2 == 1:
            N += 1
    E_n = fourier_dvr_eigenvalues(V_func, I_r_kg_m2, N=N)
    kT = KB * T_K
    E_shift = E_n - E_n[0]
    w = np.exp(-E_shift / kT)
    Z = np.sum(w)
    E_avg = np.sum(E_shift * w) / Z
    return np.log(Z) + E_avg / kT
