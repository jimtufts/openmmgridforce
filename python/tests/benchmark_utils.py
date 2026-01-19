#!/usr/bin/env python
"""
Utility functions for benchmark_hessian_analysis.py.
Extracted to keep the main script smaller and more focused.
"""

import os
import sys
import numpy as np
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gridforceplugin as gfp
from openmm.app import AmberPrmtopFile
from openmm import Platform, Context, VerletIntegrator
from openmm import NonbondedForce, System, Vec3
from gridforceplugin import NewtonMinimizer, BondedHessian
from openmm.unit import nanometer, kilojoules_per_mole, elementary_charge, dalton
from openmm.unit import kelvin, picosecond

# Constants
ONE_4PI_EPS0 = 138.935456  # kJ/mol * nm / e^2
METHOD_NAMES = {1: 'bspline', 3: 'triquintic'}
GRID_TYPES = ['charge', 'ljr', 'lja']

# Inv-power settings per grid type
GRID_INV_POWER = {
    'charge': None,
    'ljr': -6.0,
    'lja': -2.0,
}


def calculate_ref_energy_by_type(lig_positions_nm, rec_positions_nm, lig_params, rec_params, grid_type):
    """Calculate reference pairwise energy for a single grid type."""
    total = 0.0
    for i, (lig_pos, lig_p) in enumerate(zip(lig_positions_nm, lig_params)):
        for j, (rec_pos, rec_p) in enumerate(zip(rec_positions_nm, rec_params)):
            r = np.linalg.norm(np.array(lig_pos) - np.array(rec_pos))
            if r < 0.01:
                continue
            if grid_type == 'charge':
                total += ONE_4PI_EPS0 * lig_p['charge'] * rec_p['charge'] / r
            elif grid_type == 'ljr':
                eps_ij = np.sqrt(lig_p['epsilon'] * rec_p['epsilon'])
                sig_ij = 0.5 * (lig_p['sigma'] + rec_p['sigma'])
                if eps_ij > 0 and sig_ij > 0:
                    total += 4 * eps_ij * (sig_ij / r) ** 12
            elif grid_type == 'lja':
                eps_ij = np.sqrt(lig_p['epsilon'] * rec_p['epsilon'])
                sig_ij = 0.5 * (lig_p['sigma'] + rec_p['sigma'])
                if eps_ij > 0 and sig_ij > 0:
                    total += -4 * eps_ij * (sig_ij / r) ** 6
    return total


def load_systems_list(systems_file):
    """Load list of system names from file."""
    with open(systems_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def get_system_paths(system_name, base_dir):
    """Get paths to system files."""
    system_dir = os.path.join(base_dir, system_name)
    return {
        'dir': system_dir,
        'ligand_prmtop': os.path.join(system_dir, 'ligand.prmtop'),
        'ligand_inpcrd': os.path.join(system_dir, 'ligand.inpcrd'),
        'receptor_prmtop': os.path.join(system_dir, 'receptor.prmtop'),
        'receptor_inpcrd': os.path.join(system_dir, 'receptor.inpcrd'),
        'rec_box_pdb': os.path.join(system_dir, 'rec_box.pdb'),
        'poses_mol2': os.path.join(system_dir, 'ligand_poses.mol2'),
    }


def validate_system_paths(paths):
    """Check that required system files exist."""
    required = ['ligand_prmtop', 'receptor_prmtop', 'rec_box_pdb', 'poses_mol2']
    missing = [k for k in required if not os.path.exists(paths.get(k, ''))]
    return len(missing) == 0, missing


def extract_ligand_params(ligand_prmtop):
    """Extract ligand parameters from prmtop file."""
    prmtop = AmberPrmtopFile(ligand_prmtop)
    system = prmtop.createSystem()
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        return None
    params = []
    for i in range(nonbonded.getNumParticles()):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        params.append({
            'charge': charge.value_in_unit(elementary_charge),
            'sigma': sigma.value_in_unit(nanometer),
            'epsilon': epsilon.value_in_unit(kilojoules_per_mole)
        })
    return params


def create_gas_phase_system(ligand_prmtop, lig_params):
    """Create a gas-phase system with only bonded + IsolatedNonbondedForce."""
    prmtop = AmberPrmtopFile(ligand_prmtop)
    system = prmtop.createSystem()

    # Remove NonbondedForce and add IsolatedNonbondedForce
    isolated_nb_force = None
    forces_to_remove = []
    for i, force in enumerate(system.getForces()):
        if isinstance(force, NonbondedForce):
            forces_to_remove.append(i)
            # Create IsolatedNonbondedForce
            n_atoms = force.getNumParticles()
            charges = [lig_params[j]['charge'] for j in range(n_atoms)]
            sigmas = [lig_params[j]['sigma'] for j in range(n_atoms)]
            epsilons = [lig_params[j]['epsilon'] for j in range(n_atoms)]

            # Collect exclusions
            exclusions = []
            for exc_idx in range(force.getNumExceptions()):
                p1, p2, _, _, _ = force.getExceptionParameters(exc_idx)
                exclusions.append((p1, p2))

            isolated_nb_force = gfp.IsolatedNonbondedForce(charges, sigmas, epsilons, exclusions)
            system.addForce(isolated_nb_force)
            break

    for i in reversed(forces_to_remove):
        system.removeForce(i)

    return system, isolated_nb_force


def compute_nma_entropy(eigenvalues, temperature=300.0, n_rigid_modes=6,
                        eigenvalue_threshold=1e-6, mass=1.0):
    """Compute NMA entropy from eigenvalues using quantum, classical, and Schlitter formulas."""
    kB = 1.380649e-23  # J/K
    h = 6.62607015e-34  # J·s
    NA = 6.02214076e23  # mol^-1
    kJ_to_J = 1000.0
    kBT = kB * temperature

    # Convert eigenvalues from kJ/(mol·nm²) to J/(molecule·m²)
    eigenvalues_SI = np.array(eigenvalues) * kJ_to_J / NA * 1e18

    # Sort and filter eigenvalues
    sorted_eigs = np.sort(eigenvalues_SI)
    vibrational_eigs = sorted_eigs[n_rigid_modes:]  # Skip rigid modes
    positive_eigs = vibrational_eigs[vibrational_eigs > eigenvalue_threshold]

    if len(positive_eigs) == 0:
        return {
            'quantum_entropy_kB': np.nan,
            'classical_entropy_kB': np.nan,
            'schlitter_entropy_kB': np.nan,
            'n_vibrational_modes': 0,
            'n_negative_modes': 0,
            'mean_x': np.nan,
        }

    # Angular frequencies (assuming unit mass)
    omega = np.sqrt(positive_eigs / mass)
    omega = np.clip(omega, 1e-10, None)

    # Dimensionless parameter x = ℏω/(kBT)
    hbar = h / (2 * np.pi)
    x = hbar * omega / kBT

    # Quantum entropy
    S_quantum = 0.0
    for xi in x:
        if xi > 100:
            S_quantum += xi * np.exp(-xi) - np.log(1 - np.exp(-xi))
        else:
            exp_neg_x = np.exp(-xi)
            S_quantum += xi * exp_neg_x / (1 - exp_neg_x) - np.log(1 - exp_neg_x)

    # Classical entropy (quasi-harmonic)
    S_classical = 0.5 * np.sum(1 + np.log(2 * np.pi * kBT / positive_eigs))

    # Schlitter entropy
    S_schlitter = 0.5 * np.sum(np.log(1 + kBT * np.e / (hbar**2 * omega**2)))

    # Count negative eigenvalues
    n_negative = np.sum(sorted_eigs[n_rigid_modes:] < -eigenvalue_threshold)

    return {
        'quantum_entropy_kB': S_quantum,
        'classical_entropy_kB': S_classical,
        'schlitter_entropy_kB': S_schlitter,
        'n_vibrational_modes': len(positive_eigs),
        'n_negative_modes': int(n_negative),
        'mean_x': float(np.mean(x)),
    }


def compute_per_atom_hessian_analysis(hessian, n_atoms):
    """Compute per-atom eigenvalue analysis from the full Hessian."""
    eigenvalues = []
    mean_curvature = []
    total_curvature = []
    gaussian_curvature = []
    frac_anisotropy = []
    entropy = []
    min_eigenvalue = []

    for i in range(n_atoms):
        block = hessian[3*i:3*i+3, 3*i:3*i+3]
        eigvals = np.linalg.eigvalsh(block)
        eigvals_sorted = np.sort(eigvals)[::-1]

        eigenvalues.append(eigvals_sorted.tolist())
        mean_curv = np.mean(eigvals)
        mean_curvature.append(mean_curv)
        total_curvature.append(np.sum(eigvals))
        gaussian_curvature.append(np.prod(eigvals))

        # Fractional anisotropy
        if mean_curv > 0:
            fa = np.sqrt(0.5 * np.sum((eigvals - mean_curv)**2) / np.sum(eigvals**2))
        else:
            fa = 0.0
        frac_anisotropy.append(fa)

        # Per-atom entropy
        valid_eigs = eigvals[eigvals > 0]
        if len(valid_eigs) == 3:
            S = 1.5 * (1 + np.log(2 * np.pi)) - 0.5 * np.sum(np.log(valid_eigs))
            entropy.append(S)
        else:
            entropy.append(np.nan)

        min_eigenvalue.append(eigvals_sorted[-1])

    return {
        'eigenvalues': eigenvalues,
        'mean_curvature': mean_curvature,
        'total_curvature': total_curvature,
        'gaussian_curvature': gaussian_curvature,
        'frac_anisotropy': frac_anisotropy,
        'entropy': entropy,
        'min_eigenvalue': min_eigenvalue,
    }
