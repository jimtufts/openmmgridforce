#!/usr/bin/env python
"""
Comprehensive benchmark for Hessian analysis on Astex Diverse Set poses.

Tests the Hessian (second derivatives of potential energy) computation on
ligand poses after minimization. Computes:
- Per-atom eigenvalues (principal curvatures)
- Curvature metrics (mean, total, Gaussian)
- Fractional anisotropy (shape of potential well)
- Harmonic entropy estimates
- Saddle point indicators (negative eigenvalues)

Uses high-resolution grids (0.1 Angstrom = 0.01 nm spacing) with tiling
for memory efficiency.

Usage:
    # Test on single system
    python benchmark_hessian_analysis.py --systems 1g9v --methods triquintic

    # Test multiple systems with both methods
    python benchmark_hessian_analysis.py --systems 1g9v,1gkc --methods bspline,triquintic

    # With restraints (to study pose stability)
    python benchmark_hessian_analysis.py --systems 1g9v --restraint-strength 10.0

    # Full benchmark
    python benchmark_hessian_analysis.py --systems all --methods all
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gridforceplugin as gfp
from openmm.app import AmberPrmtopFile, AmberInpcrdFile
from openmm import Platform, Context, VerletIntegrator, LocalEnergyMinimizer
from openmm import NonbondedForce, Vec3
from gridforceplugin import NewtonMinimizer, BondedHessian
from openmm.unit import nanometer, kilojoules_per_mole, elementary_charge

# Import shared utilities from benchmark_utils
from benchmark_utils import (
    # Constants
    ONE_4PI_EPS0, METHOD_NAMES, GRID_TYPES, GRID_INV_POWER,
    GRID_ARCSINH_SCALE, GRID_PREFILTER_ORDER,
    ASTEX_BASE, DEFAULT_SYSTEMS_FILE,
    DEFAULT_GRID_SPACING, DEFAULT_GRID_CAP, TRIQUINTIC_GRID_CAP, METHOD_GRID_CAPS,
    TILED_THRESHOLD_GB, DEFAULT_TILE_SIZE, DEFAULT_TILE_MEMORY_MB,
    DEFAULT_MIN_TOL, DEFAULT_MIN_STEPS,
    # Utilities
    PersistentDirectory, aggressive_memory_cleanup,
    load_systems_list, get_system_paths, validate_system_paths,
    add_positional_restraints,
    generate_grid, create_evaluation_system, create_gas_phase_system,
    minimize_pose, compute_combined_hessian_analysis, compute_numerical_hessian,
    calculate_ref_energy_by_type, save_result,
)

# Import pose parser from example directory
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'example')
sys.path.insert(0, EXAMPLE_DIR)
from parse_dock6_poses import parse_mol2_file

# Local constants
DEFAULT_OUTPUT = 'benchmark_hessian_results.csv'
DEFAULT_MAX_POSES = 10
DEFAULT_RESTRAINT_STRENGTH = 0.0  # kcal/mol/A^2 - 0 means no restraints


def compute_rmsd(pos1, pos2):
    """Compute RMSD between two sets of positions (in nm). Returns RMSD in Angstroms."""
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    diff = pos1 - pos2
    rmsd_nm = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd_nm * 10.0  # Convert nm to Angstroms


def compute_gas_phase_nma(ligand_prmtop, lig_params, pose_coords_nm, platform,
                          temperature=300.0, min_tolerance=DEFAULT_MIN_TOL,
                          min_steps=DEFAULT_MIN_STEPS, use_newton=False,
                          skip_minimization=False, platform_properties=None):
    """Compute NMA entropy for ligand in gas phase (no receptor grids).

    Args:
        ligand_prmtop: AmberPrmtopFile for ligand
        lig_params: List of (charge, sigma, epsilon) tuples
        pose_coords_nm: Initial coordinates in nm (typically bound-state minimized geometry)
        platform: OpenMM Platform
        platform_properties: Dict of platform properties (e.g. precision)
        temperature: Temperature in K
        min_tolerance: Minimization tolerance
        min_steps: Max minimization steps
        use_newton: Use Newton-Raphson minimizer
        skip_minimization: If True, skip minimization and compute NMA at input geometry.
                          Use this when input is already the bound-state minimum to
                          ensure consistent geometry comparison.

    Returns:
        Dict with NMA results or None if failed
    """
    system, isolated_nb_force = create_gas_phase_system(ligand_prmtop, lig_params)

    if platform_properties is None:
        platform_properties = {}
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, platform, platform_properties)

    positions = [Vec3(x, y, z) * nanometer for x, y, z in pose_coords_nm]
    context.setPositions(positions)

    # Minimize (unless skipped to use bound-state geometry directly)
    if not skip_minimization:
        try:
            if use_newton:
                minimizer = NewtonMinimizer()
                minimizer.minimize(context, min_tolerance, min_steps)
            else:
                LocalEnergyMinimizer.minimize(context, min_tolerance, min_steps)
        except Exception as e:
            del context, integrator, system
            return None

    # Compute NMA (no grid forces, just bonded + IsolatedNonbonded)
    try:
        nma_result = compute_full_nma_entropy(
            context,
            grid_forces_dict={},  # No grids
            system=system,
            isolated_nb_force=isolated_nb_force,
            temperature=temperature,
            n_rigid_modes=6
        )
    except Exception as e:
        nma_result = None

    # Get final energy
    state = context.getState(getEnergy=True)
    E_gas = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    del context, integrator, system

    if nma_result:
        nma_result['E_gas'] = E_gas

    return nma_result


def compute_nma_with_and_without_grids(context, grid_forces_dict, system,
                                        isolated_nb_force=None, temperature=300.0,
                                        n_rigid_modes=6, eigenvalue_threshold=1e-6,
                                        use_cpu_gas_hessian=False, ligand_prmtop=None):
    """Compute NMA entropy both WITH and WITHOUT grid forces at the SAME geometry.

    This eliminates variability from different minimization endpoints by computing
    both Hessians at exactly the same atomic positions.

    Args:
        context: OpenMM Context (positions should already be at minimum)
        grid_forces_dict: Dict mapping grid_type -> GridForce object
        system: OpenMM System
        isolated_nb_force: IsolatedNonbondedForce (optional)
        temperature: Temperature in Kelvin
        n_rigid_modes: Number of rigid-body modes to exclude
        eigenvalue_threshold: Threshold for identifying zero eigenvalues
        use_cpu_gas_hessian: If True, compute gas phase Hessian numerically on CPU
                             for better determinism (slower but reproducible)
        ligand_prmtop: AmberPrmtopFile for ligand (required if use_cpu_gas_hessian=True)

    Returns:
        Dict with 'bound' and 'gas' sub-dicts, each containing NMA results,
        plus 'delta' values for the differences.
    """
    from openmm import unit

    # Get total energy and compute grid contribution for delta H
    state = context.getState(getEnergy=True)
    E_bound = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # Compute grid energy contribution per grid type (this is ΔH - the interaction with receptor)
    E_grid = 0.0
    E_grid_by_type = {}  # Track individual grid contributions
    for grid_type, grid_force in grid_forces_dict.items():
        # Get individual force group energy
        force_idx = grid_force.getForceGroup()
        grid_state = context.getState(getEnergy=True, groups={force_idx})
        E_this_grid = grid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        E_grid += E_this_grid
        E_grid_by_type[grid_type] = E_this_grid / 4.184  # Store in kcal/mol

    E_gas = E_bound - E_grid  # Energy without grid interactions
    delta_H_kJ = E_grid  # ΔH = E_bound - E_gas = E_grid
    delta_H_kcal = delta_H_kJ / 4.184

    n_atoms = context.getSystem().getNumParticles()
    n_dof = 3 * n_atoms

    # Build the base Hessian (bonded + nonbonded, no grids)
    H_base = np.zeros((n_dof, n_dof))

    # Add bonded Hessian
    bonded_hessian = BondedHessian()
    bonded_hessian.initialize(system, context)
    H_bonded = np.array(bonded_hessian.computeHessian(context)).reshape(n_dof, n_dof)
    H_base += H_bonded

    # Add IsolatedNonbondedForce Hessian
    if isolated_nb_force is not None:
        H_nb = np.array(isolated_nb_force.computeHessian(context)).reshape(n_dof, n_dof)
        H_base += H_nb

    # H_gas = gas phase Hessian (no grids)
    if use_cpu_gas_hessian and ligand_prmtop is not None:
        # Compute gas phase Hessian numerically on CPU for determinism
        # Create a gas-phase system with regular NonbondedForce
        from openmm import Platform, VerletIntegrator, Context

        gas_system = ligand_prmtop.createSystem()
        cpu_platform = Platform.getPlatformByName('Reference')
        cpu_integrator = VerletIntegrator(0.001)
        cpu_context = Context(gas_system, cpu_integrator, cpu_platform)

        # Copy positions from GPU context
        state = context.getState(getPositions=True)
        cpu_context.setPositions(state.getPositions())

        # Compute numerical Hessian on CPU
        H_gas = compute_numerical_hessian(cpu_context)
        H_gas = (H_gas + H_gas.T) / 2

        # Clean up
        del cpu_context, cpu_integrator
    else:
        # Use GPU-computed H_base (faster but potentially non-deterministic)
        H_gas = (H_base + H_base.T) / 2

    # H_bound = H_base + H_grids
    H_bound = H_base.copy()
    for grid_type, grid_force in grid_forces_dict.items():
        grid_force.computeHessian(context)
        blocks = np.array(grid_force.getHessianBlocks(context))
        for i in range(n_atoms):
            dxx, dyy, dzz, dxy, dxz, dyz = blocks[6*i:6*i+6]
            base = 3 * i
            H_bound[base+0, base+0] += dxx
            H_bound[base+1, base+1] += dyy
            H_bound[base+2, base+2] += dzz
            H_bound[base+0, base+1] += dxy
            H_bound[base+1, base+0] += dxy
            H_bound[base+0, base+2] += dxz
            H_bound[base+2, base+0] += dxz
            H_bound[base+1, base+2] += dyz
            H_bound[base+2, base+1] += dyz
    H_bound = (H_bound + H_bound.T) / 2

    # Get masses for mass-weighting
    from openmm import unit as omm_unit
    masses = np.zeros(n_atoms)
    for i in range(n_atoms):
        masses[i] = system.getParticleMass(i).value_in_unit(omm_unit.dalton)
    mass_3n = np.repeat(masses, 3)
    inv_sqrt_mass = 1.0 / np.sqrt(mass_3n)

    # Helper function to compute all three entropy measures from Hessian
    def compute_all_entropies(H):
        # Mass-weight the Hessian
        H_mw = H * np.outer(inv_sqrt_mass, inv_sqrt_mass)
        H_mw = (H_mw + H_mw.T) / 2

        # Eigendecomposition
        mw_eigs, _ = np.linalg.eigh(H_mw)

        # Skip rigid-body modes
        abs_eigs = np.abs(mw_eigs)
        n_zero = np.sum(abs_eigs < eigenvalue_threshold)
        n_skip = max(n_rigid_modes, n_zero)
        vib_eigs = mw_eigs[n_skip:]
        pos_vib_eigs = vib_eigs[vib_eigs > eigenvalue_threshold]

        if len(pos_vib_eigs) == 0:
            return {'quantum': np.nan, 'classical': np.nan, 'schlitter': np.nan,
                    'mean_x': np.nan, 'n_modes': 0}

        # Compute x = ℏω/(kBT)
        hbar_over_kB = 7.6382  # K·ps
        omega = np.sqrt(pos_vib_eigs)
        x = hbar_over_kB * omega / temperature

        # Quantum entropy: S/kB = Σ [x/(e^x-1) - ln(1-e^-x)]
        S_quantum = 0.0
        for xi in x:
            if xi < 1e-10:
                S_quantum += 1 - np.log(xi) if xi > 0 else 0
            elif xi > 30:
                S_quantum += xi * np.exp(-xi)
            else:
                S_quantum += xi / (np.exp(xi) - 1) - np.log(1 - np.exp(-xi))

        # Classical entropy: S/kB = Σ [1 - ln(ℏω/kBT)] = Σ [1 - ln(x)]
        S_classical = np.sum(1 - np.log(x))

        # Schlitter entropy: S/kB = (1/2) × Σ ln(1 + e²/x²)
        e_squared = np.e ** 2
        S_schlitter = 0.5 * np.sum(np.log(1 + e_squared / (x ** 2)))

        return {'quantum': S_quantum, 'classical': S_classical, 'schlitter': S_schlitter,
                'mean_x': np.mean(x), 'n_modes': len(pos_vib_eigs)}

    # Compute entropy for both bound and gas states
    ent_bound = compute_all_entropies(H_bound)
    ent_gas = compute_all_entropies(H_gas)

    # Delta entropies (bound - gas)
    def safe_delta(a, b):
        return a - b if (np.isfinite(a) and np.isfinite(b)) else np.nan

    delta_S_quantum = safe_delta(ent_bound['quantum'], ent_gas['quantum'])
    delta_S_classical = safe_delta(ent_bound['classical'], ent_gas['classical'])
    delta_S_schlitter = safe_delta(ent_bound['schlitter'], ent_gas['schlitter'])

    # Convert to -TΔS in kcal/mol
    kB_kcal = 1.987204e-3  # kcal/(mol·K)
    delta_minusTS_quantum = -delta_S_quantum * kB_kcal * temperature if np.isfinite(delta_S_quantum) else np.nan
    delta_minusTS_classical = -delta_S_classical * kB_kcal * temperature if np.isfinite(delta_S_classical) else np.nan
    delta_minusTS_schlitter = -delta_S_schlitter * kB_kcal * temperature if np.isfinite(delta_S_schlitter) else np.nan

    # Delta G = Delta H - T*Delta S = Delta H + (-T*Delta S)
    delta_G_quantum_kcal = delta_H_kcal + delta_minusTS_quantum if np.isfinite(delta_minusTS_quantum) else np.nan
    delta_G_classical_kcal = delta_H_kcal + delta_minusTS_classical if np.isfinite(delta_minusTS_classical) else np.nan
    delta_G_schlitter_kcal = delta_H_kcal + delta_minusTS_schlitter if np.isfinite(delta_minusTS_schlitter) else np.nan

    return {
        'bound': {
            'quantum_entropy_kB': ent_bound['quantum'],
            'classical_entropy_kB': ent_bound['classical'],
            'schlitter_entropy_kB': ent_bound['schlitter'],
            'mean_x': ent_bound['mean_x'],
            'n_vibrational_modes': ent_bound['n_modes'],
            'E_kJ_mol': E_bound,
        },
        'gas': {
            'quantum_entropy_kB': ent_gas['quantum'],
            'classical_entropy_kB': ent_gas['classical'],
            'schlitter_entropy_kB': ent_gas['schlitter'],
            'mean_x': ent_gas['mean_x'],
            'n_vibrational_modes': ent_gas['n_modes'],
            'E_kJ_mol': E_gas,
        },
        # Delta entropies (all three methods)
        'delta_quantum_entropy_kB': delta_S_quantum,
        'delta_classical_entropy_kB': delta_S_classical,
        'delta_schlitter_entropy_kB': delta_S_schlitter,
        # -TΔS in kcal/mol (all three methods)
        'delta_quantum_minusTS_kcal_mol': delta_minusTS_quantum,
        'delta_classical_minusTS_kcal_mol': delta_minusTS_classical,
        'delta_schlitter_minusTS_kcal_mol': delta_minusTS_schlitter,
        # Enthalpy and free energy (all three methods)
        'delta_H_kJ_mol': delta_H_kJ,
        'delta_H_kcal_mol': delta_H_kcal,
        'delta_G_quantum_kcal_mol': delta_G_quantum_kcal,
        'delta_G_classical_kcal_mol': delta_G_classical_kcal,
        'delta_G_schlitter_kcal_mol': delta_G_schlitter_kcal,
        'E_grid_by_type': E_grid_by_type,
        'n_atoms': n_atoms,
        'temperature': temperature,
    }


def compute_full_nma_entropy(context, grid_forces_dict, system=None,
                              isolated_nb_force=None, temperature=300.0,
                              n_rigid_modes=6, eigenvalue_threshold=1e-6,
                              use_numerical_hessian=False):
    """Compute proper Normal Mode Analysis entropy using full 3N×3N Hessian.

    This implements both classical and quantum entropy formulas:

    Classical (quasi-harmonic, Equation 16 from Bahar & Rader, 2005):
        S_classical = (k_B/2) * Σ_{k=1}^{3N-6} [1 + ln(2πk_BT/λ_k)]

    NOTE: Classical entropy has an arbitrary reference state and only
    relative values are meaningful.

    Quantum harmonic oscillator:
        S_quantum = k_B * Σ [(x/(e^x-1)) - ln(1-e^(-x))]

    where x = ℏω/(k_BT) and ω = sqrt(λ_mw) with λ_mw being eigenvalues
    of the mass-weighted Hessian.
    """
    # Get positions and trigger energy calculation
    context.getState(getEnergy=True)
    n_atoms = context.getSystem().getNumParticles()
    n_dof = 3 * n_atoms

    # Option 1: Use numerical Hessian (works for any force type)
    if use_numerical_hessian:
        H_full = compute_numerical_hessian(context)
    else:
        # Option 2: Analytical Hessian from supported force types
        H_full = np.zeros((n_dof, n_dof))

        # Add bonded Hessian contribution
        if system is not None:
            bonded_hessian = BondedHessian()
            bonded_hessian.initialize(system, context)
            H_bonded = np.array(bonded_hessian.computeHessian(context)).reshape(n_dof, n_dof)
            H_full += H_bonded

        # Add IsolatedNonbondedForce Hessian
        if isolated_nb_force is not None:
            H_nb = np.array(isolated_nb_force.computeHessian(context)).reshape(n_dof, n_dof)
            H_full += H_nb

        # Add GridForce Hessian contributions
        for grid_type, grid_force in grid_forces_dict.items():
            if grid_type == 'gbsa':
                # GBSAGridForce: full 3N×3N Hessian (cross-atom coupling via Born radii)
                grid_force.computeHessian(context)
                H_gbsa = np.array(grid_force.getFullHessian(context)).reshape(n_dof, n_dof)
                H_full += H_gbsa
            else:
                # Regular GridForce: diagonal blocks only
                grid_force.computeHessian(context)
                blocks = np.array(grid_force.getHessianBlocks(context))
                for i in range(n_atoms):
                    dxx = blocks[6*i + 0]
                    dyy = blocks[6*i + 1]
                    dzz = blocks[6*i + 2]
                    dxy = blocks[6*i + 3]
                    dxz = blocks[6*i + 4]
                    dyz = blocks[6*i + 5]

                    base = 3 * i
                    H_full[base+0, base+0] += dxx
                    H_full[base+1, base+1] += dyy
                    H_full[base+2, base+2] += dzz
                    H_full[base+0, base+1] += dxy
                    H_full[base+1, base+0] += dxy
                    H_full[base+0, base+2] += dxz
                    H_full[base+2, base+0] += dxz
                    H_full[base+1, base+2] += dyz
                    H_full[base+2, base+1] += dyz

        # Symmetrize
        H_full = (H_full + H_full.T) / 2

    # Eigendecomposition of the regular Hessian
    eigenvalues, eigenvectors = np.linalg.eigh(H_full)

    # Build mass-weighted Hessian for quantum entropy calculation
    from openmm import unit
    masses = None
    frequencies_cm1 = []
    mw_eigenvalues = None
    if system is not None:
        masses = np.zeros(n_atoms)
        for i in range(n_atoms):
            masses[i] = system.getParticleMass(i).value_in_unit(unit.dalton)

        # Build 3N mass array
        mass_3n = np.repeat(masses, 3)
        inv_sqrt_mass = 1.0 / np.sqrt(mass_3n)

        # Mass-weight the Hessian
        H_mw = H_full * np.outer(inv_sqrt_mass, inv_sqrt_mass)
        H_mw = (H_mw + H_mw.T) / 2

        # Eigendecomposition of mass-weighted Hessian
        mw_eigenvalues, _ = np.linalg.eigh(H_mw)

        # Convert eigenvalues to frequencies in cm^-1
        c_cm_ps = 2.99792458e-2  # speed of light in cm/ps
        for lam in mw_eigenvalues:
            if lam > eigenvalue_threshold:
                omega = np.sqrt(lam)  # rad/ps
                freq_cm1 = omega / (2 * np.pi * c_cm_ps)
                frequencies_cm1.append(freq_cm1)
            elif lam < -eigenvalue_threshold:
                omega = np.sqrt(-lam)
                freq_cm1 = -omega / (2 * np.pi * c_cm_ps)  # Negative indicates imaginary
                frequencies_cm1.append(freq_cm1)

    # Identify rigid-body modes
    abs_eigs = np.abs(eigenvalues)
    n_zero = np.sum(abs_eigs < eigenvalue_threshold)
    n_modes_to_skip = max(n_rigid_modes, n_zero)

    # Vibrational eigenvalues
    vibrational_eigs = eigenvalues[n_modes_to_skip:]

    # Count negative eigenvalues
    n_negative = np.sum(vibrational_eigs < -eigenvalue_threshold)

    # Compute entropy for positive vibrational modes
    # Note: Classical entropy requires mass-weighted eigenvalues (ω²)
    kB = 8.314462618e-3  # kJ/(mol*K)
    kT = kB * temperature
    two_pi_kT = 2 * np.pi * kT

    # Non-mass-weighted eigenvalues (for condition number and det_log only)
    positive_vib_eigs = vibrational_eigs[vibrational_eigs > eigenvalue_threshold]

    if len(positive_vib_eigs) > 0:
        det_log = np.sum(np.log(positive_vib_eigs))
        condition_number = positive_vib_eigs[-1] / positive_vib_eigs[0]
    else:
        det_log = np.nan
        condition_number = np.nan

    # Initialize entropy values
    nma_entropy = np.nan
    nma_entropy_per_mode = np.nan
    nma_quantum_entropy = np.nan
    nma_quantum_entropy_per_mode = np.nan
    nma_schlitter = np.nan
    nma_schlitter_per_mode = np.nan
    x_values = []

    # Compute quantum and classical harmonic oscillator entropy using mass-weighted Hessian
    # The mass-weighted eigenvalues are ω² (angular frequency squared) in units of ps^-2
    hbar_over_kB = 7.6382  # K·ps (ℏ/kB)

    if mw_eigenvalues is not None:
        mw_vib_eigs = mw_eigenvalues[n_modes_to_skip:]
        positive_mw_vib_eigs = mw_vib_eigs[mw_vib_eigs > eigenvalue_threshold]

        if len(positive_mw_vib_eigs) > 0:
            omega = np.sqrt(positive_mw_vib_eigs)  # ω in rad/ps
            x = hbar_over_kB * omega / temperature  # x = ℏω/(kB*T), dimensionless
            x_values = x.tolist()

            # Classical entropy: S/kB = Σ [1 - ln(x)] = Σ [1 - ln(ℏω/kT)]
            # This is the high-temperature limit of the quantum formula
            classical_entropy_per_mode = 1 - np.log(x)
            nma_entropy = np.sum(classical_entropy_per_mode)
            nma_entropy_per_mode = np.mean(classical_entropy_per_mode)

            # Quantum entropy: S/kB = Σ [x/(e^x-1) - ln(1-e^(-x))]
            quantum_entropy_per_mode = np.zeros_like(x)
            for i, xi in enumerate(x):
                if xi < 1e-10:
                    quantum_entropy_per_mode[i] = 1 - np.log(xi) if xi > 0 else np.nan
                elif xi > 30:
                    quantum_entropy_per_mode[i] = xi * np.exp(-xi)
                else:
                    exp_x = np.exp(xi)
                    quantum_entropy_per_mode[i] = xi / (exp_x - 1) - np.log(1 - np.exp(-xi))

            nma_quantum_entropy = np.sum(quantum_entropy_per_mode)
            nma_quantum_entropy_per_mode = np.mean(quantum_entropy_per_mode)

            # Schlitter entropy: S/kB = (1/2) × Σ ln(1 + e²/x²)
            e_squared = np.e ** 2
            schlitter_per_mode = 0.5 * np.log(1 + e_squared / (x ** 2))
            nma_schlitter = np.sum(schlitter_per_mode)
            nma_schlitter_per_mode = np.mean(schlitter_per_mode)

    # Compute covariance matrix properties
    if len(positive_vib_eigs) > 0:
        H_pinv = np.zeros((n_dof, n_dof))
        for k in range(n_modes_to_skip, n_dof):
            if eigenvalues[k] > eigenvalue_threshold:
                u_k = eigenvectors[:, k]
                H_pinv += np.outer(u_k, u_k) / eigenvalues[k]

        C = kT * H_pinv
        covariance_trace = np.trace(C)

        atomic_msf = np.zeros(n_atoms)
        for i in range(n_atoms):
            base = 3 * i
            atomic_msf[i] = C[base, base] + C[base+1, base+1] + C[base+2, base+2]
        mean_fluctuation = np.sqrt(np.mean(atomic_msf))
        b_factors = (8 * np.pi**2 / 3) * atomic_msf * 100  # Convert nm² to Å²
    else:
        covariance_trace = np.nan
        mean_fluctuation = np.nan
        b_factors = np.full(n_atoms, np.nan)

    return {
        'nma_entropy': nma_entropy,
        'nma_entropy_per_mode': nma_entropy_per_mode,
        'nma_quantum_entropy': nma_quantum_entropy,
        'nma_quantum_entropy_per_mode': nma_quantum_entropy_per_mode,
        'nma_schlitter': nma_schlitter,
        'nma_schlitter_per_mode': nma_schlitter_per_mode,
        'eigenvalues': eigenvalues.tolist(),
        'vibrational_eigenvalues': vibrational_eigs.tolist(),
        'frequencies_cm1': frequencies_cm1,
        'x_values': x_values,
        'mean_x': np.mean(x_values) if x_values else np.nan,
        'n_vibrational_modes': len(positive_vib_eigs),
        'n_negative_modes': int(n_negative),
        'n_zero_modes': int(n_zero),
        'covariance_trace': covariance_trace,
        'mean_fluctuation': mean_fluctuation,
        'b_factors': b_factors.tolist() if not np.all(np.isnan(b_factors)) else [],
        'hessian_condition_number': condition_number,
        'det_log': det_log,
        'n_atoms': n_atoms,
        'n_dof': n_dof,
        'temperature': temperature,
    }


def create_result_record(system_name, pose_data, pose_idx, method,
                         E_initial, E_final, min_time_ms,
                         hessian_analysis, compute_time_ms,
                         gen_time_ms, grid_spacing, n_atoms,
                         nma_analysis=None, gas_nma_analysis=None,
                         movement_A=np.nan, rmsd_to_xtal_A=np.nan, rmsd_init_to_xtal_A=np.nan):
    """Create a result record with all metrics for a single pose."""

    eigenvalues = np.array(hessian_analysis['eigenvalues'])
    mean_curvature = np.array(hessian_analysis['mean_curvature'])
    total_curvature = np.array(hessian_analysis['total_curvature'])
    gaussian_curvature = np.array(hessian_analysis['gaussian_curvature'])
    frac_anisotropy = np.array(hessian_analysis['frac_anisotropy'])
    entropy = np.array(hessian_analysis['entropy'])
    min_eigenvalue = np.array(hessian_analysis['min_eigenvalue'])
    num_negative = np.array(hessian_analysis['num_negative'])

    # Reshape eigenvalues: [3*n_atoms] -> [n_atoms, 3]
    eigenvalues_reshaped = eigenvalues.reshape(-1, 3)

    # Compute aggregate statistics
    all_eigs = eigenvalues_reshaped.flatten()
    valid_eigs = all_eigs[np.isfinite(all_eigs)]

    # Per-atom statistics
    mean_min_eig = np.nanmean(min_eigenvalue)
    mean_mean_curv = np.nanmean(mean_curvature)
    mean_total_curv = np.nanmean(total_curvature)
    mean_gauss_curv = np.nanmean(gaussian_curvature)
    mean_fa = np.nanmean(frac_anisotropy)

    # Entropy statistics
    valid_entropy = entropy[np.isfinite(entropy)]
    total_entropy = np.sum(valid_entropy) if len(valid_entropy) > 0 else np.nan
    n_valid_entropy = len(valid_entropy)

    # Saddle point statistics
    n_saddle_atoms = np.sum(num_negative > 0)
    n_total_negative = np.sum(num_negative)

    # Eigenvalue distribution
    n_negative_eigs = np.sum(valid_eigs < 0)
    n_positive_eigs = np.sum(valid_eigs > 0)
    eig_mean = np.mean(valid_eigs) if len(valid_eigs) > 0 else np.nan
    eig_std = np.std(valid_eigs) if len(valid_eigs) > 0 else np.nan
    eig_min = np.min(valid_eigs) if len(valid_eigs) > 0 else np.nan
    eig_max = np.max(valid_eigs) if len(valid_eigs) > 0 else np.nan

    return {
        # Identifiers
        'system': system_name,
        'pose_idx': pose_idx,
        'pose_name': pose_data['name'],
        'method': method,
        'method_name': METHOD_NAMES[method],

        # Grid parameters
        'grid_spacing_A': grid_spacing,
        'n_atoms': n_atoms,

        # DOCK6 reference scores
        'dock6_grid_score': pose_data['scores'].get('grid_score', np.nan),
        'dock6_vdw': pose_data['scores'].get('grid_vdw', np.nan),
        'dock6_es': pose_data['scores'].get('grid_es', np.nan),
        'dock6_rmsd': pose_data['scores'].get('rmsd_heavy', np.nan),

        # Post-minimization RMSD (in Angstroms)
        'movement_A': movement_A,  # How much pose moved during minimization
        'rmsd_init_to_xtal_A': rmsd_init_to_xtal_A,  # Initial pose RMSD to crystal
        'rmsd_final_to_xtal_A': rmsd_to_xtal_A,  # Final (post-min) RMSD to crystal

        # Energies
        'E_initial': E_initial,
        'E_final': E_final,
        'E_delta': E_final - E_initial,

        # Aggregate eigenvalue statistics
        'eig_mean': eig_mean,
        'eig_std': eig_std,
        'eig_min': eig_min,
        'eig_max': eig_max,
        'n_negative_eigs': n_negative_eigs,
        'n_positive_eigs': n_positive_eigs,
        'pct_negative_eigs': 100.0 * n_negative_eigs / (3 * n_atoms) if n_atoms > 0 else np.nan,

        # Per-atom statistics (averages)
        'mean_min_eigenvalue': mean_min_eig,
        'mean_mean_curvature': mean_mean_curv,
        'mean_total_curvature': mean_total_curv,
        'mean_gaussian_curvature': mean_gauss_curv,
        'mean_frac_anisotropy': mean_fa,

        # Entropy statistics
        'total_entropy': total_entropy,
        'n_valid_entropy_atoms': n_valid_entropy,
        'pct_valid_entropy': 100.0 * n_valid_entropy / n_atoms if n_atoms > 0 else np.nan,

        # Saddle point statistics
        'n_saddle_atoms': n_saddle_atoms,
        'pct_saddle_atoms': 100.0 * n_saddle_atoms / n_atoms if n_atoms > 0 else np.nan,
        'n_total_negative_eigs': n_total_negative,

        # NMA entropy (classical and quantum)
        'nma_entropy_kB': nma_analysis['nma_entropy'] if nma_analysis else np.nan,
        'nma_entropy_J_mol_K': (nma_analysis['nma_entropy'] * 8.314462618) if nma_analysis else np.nan,
        'nma_entropy_cal_mol_K': (nma_analysis['nma_entropy'] * 1.987204) if nma_analysis else np.nan,
        'nma_minusTS_kJ_mol': (-nma_analysis['nma_entropy'] * 8.314462618e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,
        'nma_minusTS_kcal_mol': (-nma_analysis['nma_entropy'] * 1.987204e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,
        'nma_quantum_entropy_kB': nma_analysis.get('nma_quantum_entropy', np.nan) if nma_analysis else np.nan,
        'nma_quantum_entropy_J_mol_K': (nma_analysis.get('nma_quantum_entropy', np.nan) * 8.314462618) if nma_analysis else np.nan,
        'nma_quantum_entropy_cal_mol_K': (nma_analysis.get('nma_quantum_entropy', np.nan) * 1.987204) if nma_analysis else np.nan,
        'nma_quantum_minusTS_kJ_mol': (-nma_analysis.get('nma_quantum_entropy', np.nan) * 8.314462618e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,
        'nma_quantum_minusTS_kcal_mol': (-nma_analysis.get('nma_quantum_entropy', np.nan) * 1.987204e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,
        'nma_schlitter_kB': nma_analysis.get('nma_schlitter', np.nan) if nma_analysis else np.nan,
        'nma_schlitter_minusTS_kcal_mol': (-nma_analysis.get('nma_schlitter', np.nan) * 1.987204e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,
        'nma_mean_x': nma_analysis.get('mean_x', np.nan) if nma_analysis else np.nan,
        'nma_n_vibrational_modes': nma_analysis['n_vibrational_modes'] if nma_analysis else np.nan,
        'nma_n_negative_modes': nma_analysis['n_negative_modes'] if nma_analysis else np.nan,
        'nma_n_zero_modes': nma_analysis['n_zero_modes'] if nma_analysis else np.nan,
        'nma_covariance_trace': nma_analysis['covariance_trace'] if nma_analysis else np.nan,
        'nma_mean_fluctuation_nm': nma_analysis['mean_fluctuation'] if nma_analysis else np.nan,
        'nma_condition_number': nma_analysis['hessian_condition_number'] if nma_analysis else np.nan,
        'nma_det_log': nma_analysis['det_log'] if nma_analysis else np.nan,

        # Gas-phase (free ligand) NMA entropy
        'gas_classical_entropy_kB': gas_nma_analysis.get('nma_entropy', np.nan) if gas_nma_analysis else np.nan,
        'gas_classical_minusTS_kcal_mol': (-gas_nma_analysis.get('nma_entropy', np.nan) * 1.987204e-3 * gas_nma_analysis['temperature']) if gas_nma_analysis else np.nan,
        'gas_quantum_entropy_kB': gas_nma_analysis.get('nma_quantum_entropy', np.nan) if gas_nma_analysis else np.nan,
        'gas_quantum_minusTS_kcal_mol': (-gas_nma_analysis.get('nma_quantum_entropy', np.nan) * 1.987204e-3 * gas_nma_analysis['temperature']) if gas_nma_analysis else np.nan,
        'gas_schlitter_kB': gas_nma_analysis.get('nma_schlitter', np.nan) if gas_nma_analysis else np.nan,
        'gas_schlitter_minusTS_kcal_mol': (-gas_nma_analysis.get('nma_schlitter', np.nan) * 1.987204e-3 * gas_nma_analysis['temperature']) if gas_nma_analysis else np.nan,
        'gas_E_kJ_mol': gas_nma_analysis.get('E_gas', np.nan) if gas_nma_analysis else np.nan,
        'gas_mean_x': gas_nma_analysis.get('mean_x', np.nan) if gas_nma_analysis else np.nan,

        # Delta entropy and thermodynamics
        'delta_quantum_entropy_kB': nma_analysis.get('direct_delta_S_quantum', np.nan) if nma_analysis else np.nan,
        'delta_quantum_minusTS_kcal_mol': nma_analysis.get('direct_delta_minusTS_quantum', np.nan) if nma_analysis else np.nan,
        'delta_classical_minusTS_kcal_mol': nma_analysis.get('direct_delta_minusTS_classical', np.nan) if nma_analysis else np.nan,
        'delta_schlitter_minusTS_kcal_mol': nma_analysis.get('direct_delta_minusTS_schlitter', np.nan) if nma_analysis else np.nan,
        'delta_H_kcal_mol': nma_analysis.get('direct_delta_H_kcal', np.nan) if nma_analysis else np.nan,
        'delta_G_quantum_kcal_mol': nma_analysis.get('direct_delta_G_quantum_kcal', np.nan) if nma_analysis else np.nan,
        'delta_G_classical_kcal_mol': nma_analysis.get('direct_delta_G_classical_kcal', np.nan) if nma_analysis else np.nan,
        'delta_G_schlitter_kcal_mol': nma_analysis.get('direct_delta_G_schlitter_kcal', np.nan) if nma_analysis else np.nan,

        # Per-grid energy breakdown
        'E_charge_kcal_mol': nma_analysis.get('E_grid_by_type', {}).get('charge', np.nan) if nma_analysis else np.nan,
        'E_ljr_kcal_mol': nma_analysis.get('E_grid_by_type', {}).get('ljr', np.nan) if nma_analysis else np.nan,
        'E_lja_kcal_mol': nma_analysis.get('E_grid_by_type', {}).get('lja', np.nan) if nma_analysis else np.nan,

        # Reference pairwise energies
        'ref_charge_kcal_mol': nma_analysis.get('ref_energies', {}).get('charge', np.nan) if nma_analysis else np.nan,
        'ref_ljr_kcal_mol': nma_analysis.get('ref_energies', {}).get('ljr', np.nan) if nma_analysis else np.nan,
        'ref_lja_kcal_mol': nma_analysis.get('ref_energies', {}).get('lja', np.nan) if nma_analysis else np.nan,

        # Timing
        'gen_time_ms': gen_time_ms,
        'min_time_ms': min_time_ms,
        'hessian_time_ms': compute_time_ms,

        # Status
        'status': 'success'
    }


def process_system(system_name, paths, poses, lig_params, platform,
                   methods, grid_spacing, output_file,
                   min_tolerance, min_steps, use_newton=False,
                   rec_params=None, rec_positions_nm=None,
                   platform_properties=None, use_cpu_gas_hessian=False,
                   separate_gas_min=False, restraint_strength=0.0,
                   xtal_coords_nm=None, entropy_mode='cartesian',
                   tiled_threshold_gb=TILED_THRESHOLD_GB,
                   save_pdb_dir=None,
                   gbsa_info=None, gbsa_spacing=0.5):
    """Process all poses for a system with all methods.

    Generates grids per-method (different methods may use different caps),
    then evaluates all poses.

    Args:
        restraint_strength: If > 0, add positional restraints during minimization
            (in kcal/mol/A^2). Hessian is computed WITHOUT restraints at the
            restrained-minimized geometry.
        xtal_coords_nm: Crystal structure coordinates in nm for RMSD calculation.
    """
    if platform_properties is None:
        platform_properties = {}
    n_atoms = len(lig_params)
    ligand_prmtop = AmberPrmtopFile(paths['ligand_prmtop'])

    print(f"  Ligand has {n_atoms} atoms")
    print(f"  Processing {len(poses)} poses")

    with PersistentDirectory('/scratch/hessian_tmp') as tmpdir:
        # Process each method - generate grids per method since caps may differ
        for method in methods:
            method_name = METHOD_NAMES[method]
            method_cap = METHOD_GRID_CAPS.get(method, DEFAULT_GRID_CAP)
            print(f"\n    Method: {method_name} (cap={method_cap:.2e} kJ/mol)")

            # Generate grids for this method
            grid_files = {}
            total_gen_time = 0

            for grid_type in GRID_TYPES:
                grid_file = os.path.join(tmpdir, f'{method_name}_{grid_type}.grid')
                print(f"      Generating {grid_type} grid ({grid_spacing}A spacing)...", end=" ")
                sys.stdout.flush()

                gen_start = time.time()
                # For method 4 (quintic B-spline), use arcsinh + prefilter during generation
                gen_arcsinh = GRID_ARCSINH_SCALE.get(grid_type, 0.0) if method == 4 else 0.0
                gen_prefilter = GRID_PREFILTER_ORDER.get(grid_type, 0) if method == 4 else 0
                # B-spline methods (1, 4) don't need derivatives; skip to save memory
                need_derivs = (method not in (1, 4))
                grid_result = generate_grid(
                    paths, grid_spacing, method_cap, grid_type,
                    platform, grid_file, platform_properties,
                    arcsinh_scale=gen_arcsinh, prefilter_order=gen_prefilter,
                    compute_derivatives=need_derivs,
                    tiled_threshold_gb=tiled_threshold_gb
                )
                gen_time = (time.time() - gen_start) * 1000
                total_gen_time += gen_time

                grid_files[grid_type] = grid_result
                tiled_indicator = " [tiled]" if grid_result.get('is_tiled', False) else ""
                print(f"{gen_time:.0f} ms (grid: {grid_result['counts']}{tiled_indicator})")
                aggressive_memory_cleanup()

            gen_time_per_pose = total_gen_time / len(poses) if poses else 0

            # Build GBSA params if requested
            gbsa_params = None
            if gbsa_info is not None and grid_files:
                # Use grid bounds from the first generated grid
                first_grid = next(iter(grid_files.values()))
                grid_origin = first_grid['origin_nm']
                grid_spacing_nm_val = first_grid['spacing_nm']

                # GBSA uses its own (coarser) spacing
                gbsa_spacing_nm = gbsa_spacing * 0.1  # Angstroms to nm

                # Compute GBSA grid counts from the same bounding box
                first_counts = first_grid['counts']
                extent = [first_counts[d] * grid_spacing_nm_val for d in range(3)]
                gbsa_counts = tuple(int(extent[d] / gbsa_spacing_nm) + 1 for d in range(3))

                # Receptor positions as flat list [x0, y0, z0, x1, ...]
                rec_pos_flat = []
                for x, y, z in rec_positions_nm:
                    rec_pos_flat.extend([x, y, z])

                # Ligand charges from lig_params
                lig_charges = [p[0] for p in lig_params]

                gbsa_params = {
                    'lig_radii': gbsa_info['lig_radii'],
                    'lig_scales': gbsa_info['lig_scales'],
                    'lig_charges': lig_charges,
                    'rec_positions': rec_pos_flat,
                    'rec_radii': gbsa_info['rec_radii'],
                    'rec_scales': gbsa_info['rec_scales'],
                    'grid_origin': grid_origin,
                    'grid_counts': gbsa_counts,
                    'grid_spacing': gbsa_spacing_nm,
                    'exclusions': gbsa_info['exclusions'],
                }
                print(f"      GBSA grid: {gbsa_counts}, spacing={gbsa_spacing_nm:.3f} nm")

            # Create system with all grids
            try:
                system, grid_forces_dict, isolated_nb_force = create_evaluation_system(
                    ligand_prmtop, lig_params, grid_files, method,
                    gbsa_params=gbsa_params
                )
            except Exception as e:
                print(f"      ERROR creating system: {e}")
                import traceback
                traceback.print_exc()
                continue

            if not grid_forces_dict:
                print("      ERROR: No GridForce created for Hessian computation")
                continue

            if isolated_nb_force is not None:
                print(f"      Added IsolatedNonbondedForce with {isolated_nb_force.getNumAtoms()} atoms")

            # Process each pose
            for pose_idx, pose_data in enumerate(poses):
                pose_coords_nm = pose_data['coordinates'] * 0.1  # Angstroms to nm

                if len(pose_coords_nm) != n_atoms:
                    print(f"      Pose {pose_idx}: atom count mismatch ({len(pose_coords_nm)} vs {n_atoms})")
                    continue

                positions = [Vec3(x, y, z) * nanometer for x, y, z in pose_coords_nm]

                # Always create a fresh system for each pose when using tiled grids
                # to avoid CUDA state issues with context reuse (workaround for tiled mode bug)
                current_system, current_grid_forces, current_iso_nb = create_evaluation_system(
                    ligand_prmtop, lig_params, grid_files, method,
                    gbsa_params=gbsa_params
                )
                if restraint_strength > 0:
                    restraint_force = add_positional_restraints(current_system, positions, restraint_strength)
                integrator = VerletIntegrator(0.001)
                context = Context(current_system, integrator, platform, platform_properties)
                context.setPositions(positions)

                # Minimize
                restraint_note = f" (k={restraint_strength})" if restraint_strength > 0 else ""
                print(f"      Pose {pose_idx}/{len(poses)}: minimizing{restraint_note}...", end=" ")
                sys.stdout.flush()

                try:
                    E_initial, E_final, min_time, converged = minimize_pose(
                        context, min_tolerance, min_steps,
                        use_newton=use_newton, system=current_system
                    )
                    status = "converged" if converged else "NOT CONVERGED"
                    print(f"E: {E_initial:.1f} -> {E_final:.1f} kJ/mol ({min_time:.0f} ms, {status})", end=" ")

                    if E_final > E_initial:
                        print(f"\n        WARNING: Energy increased during minimization!", end=" ")
                    if not converged:
                        print(f"\n        WARNING: Minimizer did not converge after {min_steps} iterations", end=" ")

                    # Compute post-minimization RMSD
                    from openmm import unit
                    state = context.getState(getPositions=True)
                    final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    movement_A = compute_rmsd(pose_coords_nm, final_pos)
                    rmsd_to_xtal_A = compute_rmsd(xtal_coords_nm, final_pos) if xtal_coords_nm is not None else np.nan
                    rmsd_init_to_xtal_A = compute_rmsd(xtal_coords_nm, pose_coords_nm) if xtal_coords_nm is not None else np.nan

                    # Print RMSD summary
                    rmsd_parts = [f"move={movement_A:.2f}A"]
                    if not np.isnan(rmsd_init_to_xtal_A):
                        rmsd_parts.append(f"init->xtal={rmsd_init_to_xtal_A:.2f}A")
                    if not np.isnan(rmsd_to_xtal_A):
                        rmsd_parts.append(f"final->xtal={rmsd_to_xtal_A:.2f}A")
                    print(f"\n        RMSD: {', '.join(rmsd_parts)}", end=" ")

                    # Save PDB files if requested
                    if save_pdb_dir is not None:
                        from openmm.app import PDBFile
                        os.makedirs(save_pdb_dir, exist_ok=True)
                        topology = ligand_prmtop.topology
                        prefix = f"{system_name}_{method_name}_pose{pose_idx}"
                        # Initial structure
                        init_pdb = os.path.join(save_pdb_dir, f"{prefix}_initial.pdb")
                        init_positions = [Vec3(*p) for p in pose_coords_nm] * unit.nanometer
                        with open(init_pdb, 'w') as f:
                            PDBFile.writeFile(topology, init_positions, f)
                        # Final (minimized) structure
                        final_pdb = os.path.join(save_pdb_dir, f"{prefix}_minimized.pdb")
                        final_positions = state.getPositions()
                        with open(final_pdb, 'w') as f:
                            PDBFile.writeFile(topology, final_positions, f)
                        print(f"\n        PDB saved: {init_pdb}, {final_pdb}", end=" ")

                except Exception as e:
                    print(f"minimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clean up before continuing to next pose (always delete system since we create fresh for each pose)
                    del context, integrator
                    del current_system, current_grid_forces, current_iso_nb
                    gc.collect()
                    aggressive_memory_cleanup()
                    continue

                # Compute combined Hessian analysis
                print("Hessian...", end=" ")
                sys.stdout.flush()

                try:
                    hess_start = time.time()
                    analysis = compute_combined_hessian_analysis(
                        context, current_grid_forces, system=current_system,
                        isolated_nb_force=current_iso_nb, temperature=300.0
                    )
                    hess_time = (time.time() - hess_start) * 1000
                    print(f"{hess_time:.0f} ms", end=" ")
                except Exception as e:
                    print(f"failed: {e}")
                    # Clean up before continuing to next pose (always delete system since we create fresh for each pose)
                    del context, integrator
                    del current_system, current_grid_forces, current_iso_nb
                    gc.collect()
                    aggressive_memory_cleanup()
                    continue

                # Compute NMA entropy
                if entropy_mode == 'internal':
                    print("NMA(internal coords, bound+gas)...", end=" ")
                elif separate_gas_min:
                    print("NMA(bound+separate gas)...", end=" ")
                else:
                    print("NMA(bound+gas)...", end=" ")
                sys.stdout.flush()
                try:
                    nma_start = time.time()

                    if entropy_mode == 'internal':
                        # Internal coordinate entropy via Wilson GF-matrix
                        bonded_hessian = BondedHessian()
                        bonded_hessian.initialize(current_system, context)

                        # Bound state: bonded force constants + grid Hessian projected into internal coords
                        bound_internal = bonded_hessian.computeInternalEntropy(
                            context, current_system, grid_forces=current_grid_forces, temperature=300.0
                        )
                        # Gas state: bonded force constants only (no grid)
                        gas_internal = bonded_hessian.computeInternalEntropy(
                            context, current_system, grid_forces=None, temperature=300.0
                        )

                        # Compute grid energy for delta H
                        from openmm import unit
                        E_grid = 0.0
                        E_grid_by_type = {}
                        for grid_type, grid_force in current_grid_forces.items():
                            force_idx = grid_force.getForceGroup()
                            grid_state = context.getState(getEnergy=True, groups={force_idx})
                            E_this_grid = grid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                            E_grid += E_this_grid
                            E_grid_by_type[grid_type] = E_this_grid / 4.184

                        state = context.getState(getEnergy=True)
                        E_bound = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                        kB_kcal = 1.987204e-3
                        T = 300.0

                        bound_S_q = bound_internal['total_quantum_entropy_kB']
                        gas_S_q = gas_internal['total_quantum_entropy_kB']
                        bound_S_c = bound_internal['total_classical_entropy_kB']
                        gas_S_c = gas_internal['total_classical_entropy_kB']

                        delta_S_quantum = bound_S_q - gas_S_q
                        delta_S_classical = bound_S_c - gas_S_c
                        delta_H_kcal = E_grid / 4.184

                        # Package into nma_analysis dict matching the expected format
                        nma_analysis = {
                            'nma_entropy': bound_S_c,
                            'nma_entropy_per_mode': bound_S_c / max(bound_internal['n_valid_modes'], 1),
                            'nma_quantum_entropy': bound_S_q,
                            'nma_quantum_entropy_per_mode': bound_S_q / max(bound_internal['n_valid_modes'], 1),
                            'nma_schlitter': np.nan,
                            'nma_schlitter_per_mode': np.nan,
                            'n_vibrational_modes': bound_internal['n_valid_modes'],
                            'n_negative_modes': bound_internal['n_negative'],
                            'n_zero_modes': 0,
                            'mean_x': np.nan,
                            'covariance_trace': np.nan,
                            'mean_fluctuation': np.nan,
                            'hessian_condition_number': np.nan,
                            'det_log': np.nan,
                            'n_atoms': n_atoms,
                            'n_dof': bound_internal['n_dof'],
                            'temperature': T,
                            'entropy_mode': 'internal',
                            # Delta values
                            'direct_delta_S_quantum': delta_S_quantum,
                            'direct_delta_S_classical': delta_S_classical,
                            'direct_delta_S_schlitter': np.nan,
                            'direct_delta_minusTS_quantum': -delta_S_quantum * kB_kcal * T,
                            'direct_delta_minusTS_classical': -delta_S_classical * kB_kcal * T,
                            'direct_delta_minusTS_schlitter': np.nan,
                            'direct_delta_H_kcal': delta_H_kcal,
                            'direct_delta_G_quantum_kcal': delta_H_kcal + (-delta_S_quantum * kB_kcal * T),
                            'direct_delta_G_classical_kcal': delta_H_kcal + (-delta_S_classical * kB_kcal * T),
                            'direct_delta_G_schlitter_kcal': np.nan,
                            'E_grid_by_type': E_grid_by_type,
                            # Internal-specific fields
                            'internal_bound_force_constants': bound_internal['force_constants']['all'].tolist(),
                            'internal_gas_force_constants': gas_internal['force_constants']['all'].tolist(),
                            'internal_effective_masses': bound_internal['effective_masses'].tolist(),
                            'internal_frequencies_cm1': bound_internal['frequencies_cm1'].tolist(),
                        }
                        gas_nma_analysis = {
                            'nma_entropy': gas_S_c,
                            'nma_quantum_entropy': gas_S_q,
                            'nma_schlitter': np.nan,
                            'mean_x': np.nan,
                            'temperature': T,
                            'E_gas': E_bound - E_grid,
                        }

                    else:
                        # Cartesian NMA (existing code path)
                        nma_analysis = compute_full_nma_entropy(
                            context, current_grid_forces, system=current_system,
                            isolated_nb_force=current_iso_nb, temperature=300.0
                        )

                        if separate_gas_min:
                            from openmm import unit

                            E_grid = 0.0
                            E_grid_by_type = {}
                            for grid_type, grid_force in current_grid_forces.items():
                                force_idx = grid_force.getForceGroup()
                                grid_state = context.getState(getEnergy=True, groups={force_idx})
                                E_this_grid = grid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                                E_grid += E_this_grid
                                E_grid_by_type[grid_type] = E_this_grid / 4.184

                            state = context.getState(getPositions=True)
                            bound_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

                            gas_nma_analysis = compute_gas_phase_nma(
                                ligand_prmtop, lig_params, bound_positions, platform,
                                temperature=300.0, min_tolerance=min_tolerance, min_steps=min_steps,
                                use_newton=use_newton, skip_minimization=False,
                                platform_properties=platform_properties
                            )

                            if gas_nma_analysis is not None:
                                kB_kcal = 1.987204e-3
                                T = 300.0

                                bound_S_quantum = nma_analysis.get('nma_quantum_entropy', np.nan)
                                bound_S_classical = nma_analysis.get('nma_entropy', np.nan)
                                bound_S_schlitter = nma_analysis.get('nma_schlitter', np.nan)
                                gas_S_quantum = gas_nma_analysis.get('nma_quantum_entropy', np.nan)
                                gas_S_classical = gas_nma_analysis.get('nma_entropy', np.nan)
                                gas_S_schlitter = gas_nma_analysis.get('nma_schlitter', np.nan)

                                delta_S_quantum = bound_S_quantum - gas_S_quantum
                                delta_S_classical = bound_S_classical - gas_S_classical
                                delta_S_schlitter = bound_S_schlitter - gas_S_schlitter

                                nma_analysis['direct_delta_S_quantum'] = delta_S_quantum
                                nma_analysis['direct_delta_S_classical'] = delta_S_classical
                                nma_analysis['direct_delta_S_schlitter'] = delta_S_schlitter
                                nma_analysis['direct_delta_minusTS_quantum'] = -delta_S_quantum * kB_kcal * T
                                nma_analysis['direct_delta_minusTS_classical'] = -delta_S_classical * kB_kcal * T
                                nma_analysis['direct_delta_minusTS_schlitter'] = -delta_S_schlitter * kB_kcal * T

                                delta_H_kcal = E_grid / 4.184
                                nma_analysis['direct_delta_H_kcal'] = delta_H_kcal
                                nma_analysis['direct_delta_G_quantum_kcal'] = delta_H_kcal + nma_analysis['direct_delta_minusTS_quantum']
                                nma_analysis['direct_delta_G_classical_kcal'] = delta_H_kcal + nma_analysis['direct_delta_minusTS_classical']
                                nma_analysis['direct_delta_G_schlitter_kcal'] = delta_H_kcal + nma_analysis['direct_delta_minusTS_schlitter']
                                nma_analysis['E_grid_by_type'] = E_grid_by_type
                        else:
                            combined_nma = compute_nma_with_and_without_grids(
                                context, current_grid_forces, system=current_system,
                                isolated_nb_force=current_iso_nb, temperature=300.0,
                                use_cpu_gas_hessian=use_cpu_gas_hessian,
                                ligand_prmtop=ligand_prmtop
                            )

                            gas_nma_analysis = {
                                'nma_entropy': combined_nma['gas']['classical_entropy_kB'],
                                'nma_quantum_entropy': combined_nma['gas']['quantum_entropy_kB'],
                                'nma_schlitter': combined_nma['gas']['schlitter_entropy_kB'],
                                'mean_x': combined_nma['gas']['mean_x'],
                                'temperature': 300.0,
                                'E_gas': combined_nma['gas'].get('E_kJ_mol', np.nan),
                            }
                            nma_analysis['direct_delta_S_quantum'] = combined_nma['delta_quantum_entropy_kB']
                            nma_analysis['direct_delta_S_classical'] = combined_nma['delta_classical_entropy_kB']
                            nma_analysis['direct_delta_S_schlitter'] = combined_nma['delta_schlitter_entropy_kB']
                            nma_analysis['direct_delta_minusTS_quantum'] = combined_nma['delta_quantum_minusTS_kcal_mol']
                            nma_analysis['direct_delta_minusTS_classical'] = combined_nma['delta_classical_minusTS_kcal_mol']
                            nma_analysis['direct_delta_minusTS_schlitter'] = combined_nma['delta_schlitter_minusTS_kcal_mol']
                            nma_analysis['direct_delta_H_kcal'] = combined_nma.get('delta_H_kcal_mol', np.nan)
                            nma_analysis['direct_delta_G_quantum_kcal'] = combined_nma.get('delta_G_quantum_kcal_mol', np.nan)
                            nma_analysis['direct_delta_G_classical_kcal'] = combined_nma.get('delta_G_classical_kcal_mol', np.nan)
                            nma_analysis['direct_delta_G_schlitter_kcal'] = combined_nma.get('delta_G_schlitter_kcal_mol', np.nan)
                            nma_analysis['E_grid_by_type'] = combined_nma.get('E_grid_by_type', {})

                    # Compute reference pairwise energies
                    if rec_params is not None and rec_positions_nm is not None:
                        from openmm import unit
                        state = context.getState(getPositions=True)
                        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        lig_pos_nm = [(p[0], p[1], p[2]) for p in positions]
                        ref_energies = {}
                        for grid_type in ['charge', 'ljr', 'lja']:
                            ref_E = calculate_ref_energy_by_type(
                                lig_pos_nm, rec_positions_nm, lig_params, rec_params, grid_type
                            )
                            ref_energies[grid_type] = ref_E / 4.184
                        nma_analysis['ref_energies'] = ref_energies

                except Exception as e:
                    print(f"failed: {e}")
                    import traceback
                    traceback.print_exc()
                    nma_analysis = None
                    gas_nma_analysis = None

                # Create result record
                result = create_result_record(
                    system_name, pose_data, pose_idx, method,
                    E_initial, E_final, min_time,
                    analysis, hess_time,
                    gen_time_per_pose, grid_spacing, n_atoms,
                    nma_analysis=nma_analysis,
                    gas_nma_analysis=gas_nma_analysis,
                    movement_A=movement_A,
                    rmsd_to_xtal_A=rmsd_to_xtal_A,
                    rmsd_init_to_xtal_A=rmsd_init_to_xtal_A
                )

                # Save immediately
                save_result(output_file, result)

                # Print summary
                delta_H = result.get('delta_H_kcal_mol', np.nan)
                delta_minusTS = result.get('delta_quantum_minusTS_kcal_mol', np.nan)
                delta_G = result.get('delta_G_quantum_kcal_mol', np.nan)

                thermo_parts = []
                if np.isfinite(delta_H):
                    thermo_parts.append(f"ΔH={delta_H:.1f}")
                if np.isfinite(delta_minusTS):
                    thermo_parts.append(f"-TΔS={delta_minusTS:.1f}")
                if np.isfinite(delta_G):
                    thermo_parts.append(f"ΔG={delta_G:.1f}")
                thermo_str = ", ".join(thermo_parts) + " kcal/mol" if thermo_parts else "thermo=N/A"

                print(f"        -> eig_mean={result['eig_mean']:.1f}, "
                      f"eig_min={result['eig_min']:.1f}, "
                      f"FA={result['mean_frac_anisotropy']:.3f}, "
                      f"saddle={result['pct_saddle_atoms']:.1f}%")
                print(f"        -> {thermo_str}")

                # Clean up context for this pose (critical to avoid GPU memory accumulation)
                # We always create fresh system for each pose now to avoid CUDA state issues
                del context, integrator
                del current_system, current_grid_forces, current_iso_nb
                gc.collect()
                aggressive_memory_cleanup()

            # Clean up shared system (originally for when restraints == 0, but now unused)
            del system
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Hessian Analysis Benchmark')
    parser.add_argument('--systems', type=str, default='1g9v',
                        help='Systems to test: all, or comma-separated (1g9v,1gkc,...)')
    parser.add_argument('--methods', type=str, default='triquintic',
                        help='Methods: all, bspline, triquintic, or comma-separated')
    parser.add_argument('--grid-spacing', type=float, default=DEFAULT_GRID_SPACING,
                        help=f'Grid spacing in Angstroms (default: {DEFAULT_GRID_SPACING})')
    parser.add_argument('--max-poses', type=int, default=DEFAULT_MAX_POSES,
                        help=f'Max poses per system (default: {DEFAULT_MAX_POSES})')
    parser.add_argument('--min-tolerance', type=float, default=DEFAULT_MIN_TOL,
                        help=f'Minimization tolerance kJ/mol/nm (default: {DEFAULT_MIN_TOL})')
    parser.add_argument('--min-steps', type=int, default=DEFAULT_MIN_STEPS,
                        help=f'Max minimization steps (default: {DEFAULT_MIN_STEPS})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output CSV file (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--platform', type=str, default='CUDA',
                        help='OpenMM platform (default: CUDA)')
    parser.add_argument('--use-newton', action='store_true', default=True,
                        help='Use Newton-Raphson minimizer (default: True)')
    parser.add_argument('--use-lbfgs', action='store_true',
                        help='Use L-BFGS minimizer (not compatible with IsolatedNonbondedForce)')
    parser.add_argument('--native', action='store_true',
                        help='Use native (crystallographic) pose instead of docked poses')
    parser.add_argument('--cpu-gas-hessian', action='store_true',
                        help='Compute gas phase Hessian on CPU for determinism (slower)')
    parser.add_argument('--separate-gas-min', action='store_true',
                        help='Minimize gas phase separately (different geometry from bound state)')
    parser.add_argument('--restraint-strength', type=float, default=DEFAULT_RESTRAINT_STRENGTH,
                        help=f'Positional restraint strength in kcal/mol/A^2 (default: {DEFAULT_RESTRAINT_STRENGTH}). '
                             'If > 0, poses are minimized with restraints but Hessian is computed without.')
    parser.add_argument('--entropy-mode', type=str, default='cartesian',
                        choices=['cartesian', 'internal'],
                        help='Entropy computation mode: cartesian (full 3Nx3N Hessian NMA) or '
                             'internal (Wilson GF-matrix with scalar force constants). Default: cartesian')
    parser.add_argument('--save-pdb', type=str, default=None,
                        help='Directory to save PDB files of initial and minimized structures')
    parser.add_argument('--tiled-threshold', type=float, default=TILED_THRESHOLD_GB,
                        help='Grid size threshold in GB for tiled storage '
                             '(default: {:.1f}). Set low (e.g. 0.001) to force tiled mode.'.format(TILED_THRESHOLD_GB))
    parser.add_argument('--gbsa', action='store_true',
                        help='Include GBSAGridForce (grid-based GBSA solvation) in Hessian computation')
    parser.add_argument('--gbsa-spacing', type=float, default=0.5,
                        help='GBSA grid spacing in Angstroms (default: 0.5). Coarser than GridForce since HCT is smooth.')

    args = parser.parse_args()

    # Handle minimizer selection: --use-lbfgs overrides default Newton
    if args.use_lbfgs:
        args.use_newton = False

    # Parse systems
    all_systems = load_systems_list() if os.path.exists(DEFAULT_SYSTEMS_FILE) else []
    if args.systems.lower() == 'all':
        systems = all_systems
    else:
        systems = [s.strip() for s in args.systems.split(',')]

    # Parse methods
    if args.methods.lower() == 'all':
        methods = [1, 3, 4]  # bspline, triquintic, quintic_bspline
    else:
        method_map = {'bspline': 1, 'triquintic': 3, 'quintic_bspline': 4, 'quintic': 4}
        methods = []
        for m in args.methods.split(','):
            m = m.strip().lower()
            if m in method_map:
                methods.append(method_map[m])
            elif m.isdigit():
                methods.append(int(m))

    # Check platform
    try:
        platform = Platform.getPlatformByName(args.platform)
    except Exception:
        print(f"Platform {args.platform} not available")
        sys.exit(1)

    platform_properties = {}

    print("=" * 80)
    print("HESSIAN ANALYSIS BENCHMARK")
    print("=" * 80)
    print(f"Systems: {len(systems)} ({systems[0] if systems else 'none'}{'...' if len(systems) > 1 else ''})")
    print(f"Methods: {[METHOD_NAMES[m] for m in methods]}")
    print(f"Grid spacing: {args.grid_spacing} A")
    print(f"Grid caps: bspline={DEFAULT_GRID_CAP:.0f}, triquintic={TRIQUINTIC_GRID_CAP:.0e} kJ/mol")
    print(f"Poses: {'native (crystallographic)' if args.native else f'docked (max {args.max_poses})'}")
    minimizer_name = "Newton-Raphson" if args.use_newton else "L-BFGS"
    print(f"Minimizer: {minimizer_name}, tol={args.min_tolerance}, max_steps={args.min_steps}")
    if args.restraint_strength > 0:
        print(f"Restraints: {args.restraint_strength} kcal/mol/A^2")
    print(f"Output: {args.output}")
    print(f"Platform: {args.platform}")
    print(f"Entropy mode: {args.entropy_mode}")
    if args.gbsa:
        print(f"GBSA: enabled (grid spacing: {args.gbsa_spacing} A)")
    print(f"Tiled threshold: {args.tiled_threshold:.3f} GB")
    if args.cpu_gas_hessian:
        print("Gas phase Hessian: CPU (deterministic)")
    print("=" * 80)

    completed = 0
    errors = 0

    for system_name in systems:
        print(f"\n{'='*60}")
        print(f"SYSTEM: {system_name}")
        print(f"{'='*60}")

        # Get and validate paths
        paths = get_system_paths(system_name)
        missing = validate_system_paths(paths)
        if missing:
            print(f"SKIPPING - Missing files:")
            for m in missing:
                print(f"  - {m}")
            errors += 1
            continue

        # Load ligand parameters
        ligand_prmtop = AmberPrmtopFile(paths['ligand_prmtop'])
        lig_sys = ligand_prmtop.createSystem()
        lig_nb = [f for f in lig_sys.getForces() if isinstance(f, NonbondedForce)][0]

        lig_params = []
        for i in range(ligand_prmtop.topology.getNumAtoms()):
            q, s, e = lig_nb.getParticleParameters(i)
            lig_params.append((
                q.value_in_unit(elementary_charge),
                s.value_in_unit(nanometer),
                e.value_in_unit(kilojoules_per_mole)
            ))

        # Load receptor parameters
        receptor_prmtop = AmberPrmtopFile(paths['receptor_prmtop'])
        receptor_inpcrd = AmberInpcrdFile(paths['receptor_inpcrd'])
        rec_sys = receptor_prmtop.createSystem()
        rec_nb = [f for f in rec_sys.getForces() if isinstance(f, NonbondedForce)][0]

        rec_params = []
        for i in range(receptor_prmtop.topology.getNumAtoms()):
            q, s, e = rec_nb.getParticleParameters(i)
            rec_params.append((
                q.value_in_unit(elementary_charge),
                s.value_in_unit(nanometer),
                e.value_in_unit(kilojoules_per_mole)
            ))

        rec_positions_nm = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                             p[2].value_in_unit(nanometer)) for p in receptor_inpcrd.positions]

        # Extract GBSA parameters if requested
        gbsa_info = None
        if args.gbsa:
            try:
                from openmm.app import OBC2
                from openmm import GBSAOBCForce

                # Extract ligand GBSA radii and scale factors
                lig_gbsa_sys = ligand_prmtop.createSystem(implicitSolvent=OBC2)
                lig_gbsa_force = None
                for fi in range(lig_gbsa_sys.getNumForces()):
                    f = lig_gbsa_sys.getForce(fi)
                    if isinstance(f, GBSAOBCForce):
                        lig_gbsa_force = f
                        break

                if lig_gbsa_force is None:
                    print("  Warning: Could not find GBSAOBCForce for ligand, skipping GBSA")
                else:
                    lig_gbsa_radii = []
                    lig_gbsa_scales = []
                    for i in range(lig_gbsa_force.getNumParticles()):
                        q, r, s = lig_gbsa_force.getParticleParameters(i)
                        lig_gbsa_radii.append(r.value_in_unit(nanometer))
                        lig_gbsa_scales.append(s)

                    # Extract receptor GBSA radii and scale factors
                    rec_gbsa_sys = receptor_prmtop.createSystem(implicitSolvent=OBC2)
                    rec_gbsa_force = None
                    for fi in range(rec_gbsa_sys.getNumForces()):
                        f = rec_gbsa_sys.getForce(fi)
                        if isinstance(f, GBSAOBCForce):
                            rec_gbsa_force = f
                            break

                    rec_gbsa_radii = []
                    rec_gbsa_scales = []
                    for i in range(rec_gbsa_force.getNumParticles()):
                        q, r, s = rec_gbsa_force.getParticleParameters(i)
                        rec_gbsa_radii.append(r.value_in_unit(nanometer))
                        rec_gbsa_scales.append(s)

                    # Extract exclusions from ligand NonbondedForce (1-2, 1-3 pairs)
                    lig_nb_for_excl = None
                    for fi in range(lig_gbsa_sys.getNumForces()):
                        f = lig_gbsa_sys.getForce(fi)
                        if isinstance(f, NonbondedForce):
                            lig_nb_for_excl = f
                            break

                    exclusions = []
                    if lig_nb_for_excl is not None:
                        for i in range(lig_nb_for_excl.getNumExceptions()):
                            p1, p2, chargeProd, sigma, epsilon = lig_nb_for_excl.getExceptionParameters(i)
                            cp = chargeProd.value_in_unit(elementary_charge**2)
                            ep = epsilon.value_in_unit(kilojoules_per_mole)
                            if abs(cp) < 1e-10 and abs(ep) < 1e-10:
                                exclusions.append((p1, p2))

                    gbsa_info = {
                        'lig_radii': lig_gbsa_radii,
                        'lig_scales': lig_gbsa_scales,
                        'rec_radii': rec_gbsa_radii,
                        'rec_scales': rec_gbsa_scales,
                        'exclusions': exclusions,
                    }
                    print(f"  GBSA: {len(lig_gbsa_radii)} ligand atoms, "
                          f"{len(rec_gbsa_radii)} receptor atoms, "
                          f"{len(exclusions)} exclusions")

                    del lig_gbsa_sys, rec_gbsa_sys
            except Exception as e:
                print(f"  Warning: Could not extract GBSA parameters: {e}")
                import traceback
                traceback.print_exc()

        # Load crystal pose for RMSD reference
        xtal_coords_nm = None
        try:
            xtal_poses = parse_mol2_file(paths['xtal_pose'], max_poses=1)
            if xtal_poses and len(xtal_poses[0]['coordinates']) == len(lig_params):
                xtal_coords_nm = xtal_poses[0]['coordinates'] * 0.1  # Angstroms to nm
        except Exception as e:
            print(f"  Warning: Could not load crystal pose for RMSD: {e}")

        # Load poses
        try:
            if args.native:
                poses = parse_mol2_file(paths['xtal_pose'], max_poses=1)
                print(f"  Loaded native (crystallographic) pose")
            else:
                poses = parse_mol2_file(paths['docked_poses'], max_poses=args.max_poses)
                print(f"  Loaded {len(poses)} docked poses")
        except Exception as e:
            print(f"  SKIPPING - Error loading poses: {e}")
            errors += 1
            continue

        # Filter poses with valid atom counts
        valid_poses = [p for p in poses if len(p['coordinates']) == len(lig_params)]
        if len(valid_poses) < len(poses):
            print(f"  Filtered to {len(valid_poses)} poses with matching atom count")

        if not valid_poses:
            print(f"  SKIPPING - No valid poses")
            errors += 1
            continue

        # Process system
        try:
            process_system(
                system_name, paths, valid_poses, lig_params, platform,
                methods, args.grid_spacing, args.output,
                args.min_tolerance, args.min_steps, args.use_newton,
                rec_params=rec_params, rec_positions_nm=rec_positions_nm,
                platform_properties=platform_properties,
                use_cpu_gas_hessian=args.cpu_gas_hessian,
                separate_gas_min=args.separate_gas_min,
                restraint_strength=args.restraint_strength,
                xtal_coords_nm=xtal_coords_nm,
                entropy_mode=args.entropy_mode,
                tiled_threshold_gb=args.tiled_threshold,
                save_pdb_dir=args.save_pdb,
                gbsa_info=gbsa_info,
                gbsa_spacing=args.gbsa_spacing
            )
            completed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: Completed={completed} systems, Errors={errors}")
    print(f"Results saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
