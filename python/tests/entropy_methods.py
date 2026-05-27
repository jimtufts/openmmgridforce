"""Closed-form per-mode entropy formulas for the hybrid entropy pipeline.

Tier 2 of hessian_plan.md. Lifted from the validated 1D toys in
AlGDock/mwe/toys/toys/toy_entropy_benchmark.py (PG validated to <0.005 k_B vs
DVR across V_b = 0.01-1000 kJ/mol on the toy 3-fold cosine system).

All functions return entropy in units of k_B (dimensionless).
"""
import numpy as np
from scipy.special import iv

# Physical constants (SI)
HBAR = 1.054571817e-34       # J·s
KB = 1.380649e-23            # J/K
NA = 6.02214076e23           # mol^-1
H_PLANCK = 2 * np.pi * HBAR  # J·s
AMU_TO_KG = 1.66053907e-27   # kg/amu
ANG_TO_M = 1e-10             # m/Å


# ----------------------------------------------------------------------------
# Harmonic oscillator
# ----------------------------------------------------------------------------
def entropy_quantum_HO(omega, T_K):
    """Quantum harmonic oscillator entropy per mode (k_B units).

    S/k_B = x/(e^x - 1) - ln(1 - e^{-x})  where x = hbar*omega/(k_B*T)

    omega: angular frequency in rad/s. Non-positive returns 0 (degenerate mode).
    """
    if omega <= 0:
        return 0.0
    x = HBAR * omega / (KB * T_K)
    if x < 1e-12:
        return 1.0 - np.log(x)  # classical limit
    if x > 50:
        return x * np.exp(-x)
    return x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))


def _harmonic_well_frequency(V_b_J, n, I_r_kg_m2):
    """Angular frequency at the bottom of one well of V(phi) = (V_b/2)(1-cos(n*phi))."""
    return np.sqrt(V_b_J * n * n / (2.0 * I_r_kg_m2))


# ----------------------------------------------------------------------------
# Pitzer-Gwinn hindered rotor
# ----------------------------------------------------------------------------
def entropy_pitzer_gwinn(V_b_J, n, I_r_kg_m2, T_K, sigma=1):
    """Pitzer-Gwinn closed-form hindered-rotor entropy (k_B units).

    For V(phi) = (V_b/2) * (1 - cos(n*phi)):
        Z_HR = Z_FR * (Z_HO_q / Z_HO_cl) * exp(-u/2) * I_0(u/2)
    where u = V_b / (k_B*T), Z_FR is the 1D free-rotor partition function,
    and Z_HO_q/Z_HO_cl is the quantum/classical HO correction at the well
    frequency.

    Args:
        V_b_J:        barrier height in J (per molecule)
        n:            periodicity (integer >= 1)
        I_r_kg_m2:    reduced moment of inertia in kg·m^2
        T_K:          temperature in K
        sigma:        rotational symmetry number (1 unless known otherwise)

    Validated to < 0.005 k_B vs Fourier DVR across V_b = 0.01-1000 kJ/mol.
    """
    eps = 1e-4
    Tp, Tm = T_K * (1 + eps), T_K * (1 - eps)
    lnZ_T = _ln_Z_PG(V_b_J, n, I_r_kg_m2, T_K, sigma)
    lnZ_p = _ln_Z_PG(V_b_J, n, I_r_kg_m2, Tp,  sigma)
    lnZ_m = _ln_Z_PG(V_b_J, n, I_r_kg_m2, Tm,  sigma)
    dlnZ_dT = (lnZ_p - lnZ_m) / (Tp - Tm)
    return lnZ_T + T_K * dlnZ_dT


def _ln_Z_PG(V_b_J, n, I_r_kg_m2, T_K, sigma):
    omega = _harmonic_well_frequency(V_b_J, n, I_r_kg_m2)
    Z_FR = np.sqrt(2 * np.pi * I_r_kg_m2 * KB * T_K) / (HBAR * sigma)
    x = HBAR * omega / (KB * T_K)
    Q_corr = 1.0 if x < 1e-10 else x * np.exp(-x / 2) / (1 - np.exp(-x))
    u = V_b_J / (KB * T_K)
    f = np.exp(-u / 2) * iv(0, u / 2)
    return np.log(Z_FR * Q_corr * f)


def entropy_free_rotor(I_r_kg_m2, T_K, sigma=1):
    """Classical 1D free rotor entropy (k_B units).

    Z_FR = (1/sigma) * sqrt(2*pi*I_r*k_B*T) / hbar
    S/k_B = 1/2 + ln(Z_FR)
    """
    Z = np.sqrt(2 * np.pi * I_r_kg_m2 * KB * T_K) / (HBAR * sigma)
    return 0.5 + np.log(Z)


# ----------------------------------------------------------------------------
# Sackur-Tetrode translation
# ----------------------------------------------------------------------------
def sackur_tetrode(total_mass_kg, T_K=300.0, standard_state='1M'):
    """Translational entropy of a free molecule (k_B units).

    S_trans/k_B = ln[(2*pi*m*k_B*T/h^2)^(3/2) * V_per_molecule] + 5/2

    Args:
        total_mass_kg:   molecular mass in kg
        T_K:             temperature in K
        standard_state:  '1M' (biochem; V° = 1660 Å³/molecule) or '1atm'

    CRITICAL: For binding free energy comparisons to experimental K_d (M
    units), use '1M'. The 1 atm convention gives a S_trans that is larger
    by k_B*ln(24800/1660) ≈ 2.7 k_B, contributing a spurious ~1.6 kcal/mol
    to -TΔS_binding. See Swanson, Henchman, McCammon, Biophys J 86:67-74 (2004).
    """
    kT = KB * T_K
    if standard_state == '1M':
        V_per_molecule = 1660.0 * (ANG_TO_M ** 3)         # m^3 (= 1/N_A × 1 L/M)
    elif standard_state == '1atm':
        V_per_molecule = kT / 101325.0
    else:
        raise ValueError(f"Unknown standard_state {standard_state!r}; use '1M' or '1atm'.")
    thermal_wavelength = H_PLANCK / np.sqrt(2 * np.pi * total_mass_kg * kT)
    return np.log(V_per_molecule / thermal_wavelength ** 3) + 2.5


# ----------------------------------------------------------------------------
# Rigid rotor
# ----------------------------------------------------------------------------
def rigid_rotor(I_principal_kg_m2, T_K=300.0, sigma=1):
    """Rotational entropy of a non-linear rigid molecule (k_B units).

    S_rot/k_B = 3/2 + (1/2) ln[(8*pi^2*k_B*T)^3 * I_A*I_B*I_C * pi / (sigma^2 * h^6)]

    Args:
        I_principal_kg_m2: array-like of 3 principal moments of inertia (kg·m^2)
        T_K:               temperature in K
        sigma:             rotational symmetry number (1 asymmetric top, 2-12 symmetric)

    For linear molecules use the diatomic rigid-rotor formula instead (not implemented here).
    """
    I_A, I_B, I_C = sorted(I_principal_kg_m2)
    kT = KB * T_K
    inside = (8 * np.pi ** 2 * kT) ** 3 * I_A * I_B * I_C * np.pi / (sigma ** 2 * H_PLANCK ** 6)
    return 1.5 + 0.5 * np.log(inside)


def principal_moments_of_inertia(positions_m, masses_kg):
    """Principal moments of inertia about the centre of mass (kg·m^2).

    Args:
        positions_m: (N, 3) atomic positions in meters
        masses_kg:   (N,)   atomic masses in kg

    Returns:
        (3,) sorted ascending principal moments of inertia.
    """
    positions_m = np.asarray(positions_m)
    masses_kg = np.asarray(masses_kg)
    com = (masses_kg[:, None] * positions_m).sum(axis=0) / masses_kg.sum()
    r = positions_m - com
    I_tensor = np.zeros((3, 3))
    r2_sum = np.sum(r * r, axis=1)
    for a in range(3):
        for b in range(3):
            if a == b:
                I_tensor[a, b] = np.sum(masses_kg * (r2_sum - r[:, a] ** 2))
            else:
                I_tensor[a, b] = -np.sum(masses_kg * r[:, a] * r[:, b])
    return np.sort(np.linalg.eigvalsh(I_tensor))
