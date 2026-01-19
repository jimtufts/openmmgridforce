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

    # Full benchmark
    python benchmark_hessian_analysis.py --systems all --methods all
"""

import argparse
import sys
import os
import time
import tempfile
import numpy as np
import pandas as pd
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gridforceplugin as gfp
from openmm.app import AmberPrmtopFile, AmberInpcrdFile
from openmm import Platform, Context, VerletIntegrator, LangevinMiddleIntegrator
from openmm import NonbondedForce, System, Vec3, LocalEnergyMinimizer
from gridforceplugin import NewtonMinimizer, BondedHessian
from openmm.unit import nanometer, kilojoules_per_mole, elementary_charge, dalton
from openmm.unit import kelvin, picosecond

# Import pose parser from example directory
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'example')
sys.path.insert(0, EXAMPLE_DIR)
from parse_dock6_poses import parse_mol2_file, parse_rec_box_pdb, get_grid_bounds_for_openmm

# Constants
ONE_4PI_EPS0 = 138.935456  # kJ/mol * nm / e^2
METHOD_NAMES = {1: 'bspline', 3: 'triquintic'}
GRID_TYPES = ['charge', 'ljr', 'lja']  # All 3 grid types

# Inv-power settings per grid type (for runtime transformation)
# This helps tame steep LJ repulsion
GRID_INV_POWER = {
    'charge': None,   # No transformation for electrostatics
    'ljr': -6.0,      # LJR: soften r^-12 -> r^-6 behavior
    'lja': -2.0,      # LJA: soften r^-6 -> r^-2 behavior
}

# Default paths
ASTEX_BASE = '/scratch/AstexDiv_comprehensive'
DEFAULT_SYSTEMS_FILE = os.path.join(ASTEX_BASE, 'systems.txt')
DEFAULT_OUTPUT = 'benchmark_hessian_results.csv'

# High-resolution grid settings
DEFAULT_GRID_SPACING = 0.5  # Angstroms (0.05 nm) - moderate resolution (0.1A causes memory issues)
DEFAULT_TILE_SIZE = 32      # Tile size for memory efficiency (if tiling enabled)
DEFAULT_MEMORY_BUDGET_MB = 4096  # 4 GB memory budget
DEFAULT_GRID_CAP = 41840.0  # kJ/mol - default value for bspline
TRIQUINTIC_GRID_CAP = 1e30  # kJ/mol - essentially no cap for triquintic
DEFAULT_MAX_POSES = 10      # Limit poses per system
USE_TILED_MODE = False      # Disable tiled mode to avoid stability issues

# Map method to grid cap
METHOD_GRID_CAPS = {
    1: DEFAULT_GRID_CAP,      # bspline uses default cap
    3: TRIQUINTIC_GRID_CAP,   # triquintic uses 1e30 (no cap) for accurate Hessians
}

# Minimization settings
DEFAULT_MIN_TOL = 1.0       # kJ/mol/nm - energy tolerance for minimization
DEFAULT_MIN_STEPS = 1000    # Max minimization steps


def calculate_ref_energy_by_type(lig_positions_nm, rec_positions_nm, lig_params, rec_params, grid_type):
    """Calculate total reference energy for a specific grid type using pairwise interactions.

    Args:
        lig_positions_nm: Ligand positions in nm, shape (n_lig, 3)
        rec_positions_nm: Receptor positions in nm, shape (n_rec, 3)
        lig_params: List of (charge, sigma, epsilon) for each ligand atom
        rec_params: List of (charge, sigma, epsilon) for each receptor atom
        grid_type: 'charge', 'ljr', or 'lja'

    Returns:
        Total energy in kJ/mol
    """
    lig_pos = np.asarray(lig_positions_nm)
    rec_pos = np.asarray(rec_positions_nm)

    lig_q = np.array([p[0] for p in lig_params])
    lig_s = np.array([p[1] for p in lig_params])
    lig_e = np.array([p[2] for p in lig_params])

    rec_q = np.array([p[0] for p in rec_params])
    rec_s = np.array([p[1] for p in rec_params])
    rec_e = np.array([p[2] for p in rec_params])

    # Compute pairwise distances: (n_lig, n_rec)
    diff = lig_pos[:, np.newaxis, :] - rec_pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2))
    r_safe = np.where(r < 0.01, 1e10, r)

    if grid_type == 'charge':
        E_pair = ONE_4PI_EPS0 * lig_q[:, np.newaxis] * rec_q[np.newaxis, :] / r_safe
    elif grid_type == 'ljr':
        sig_ij = np.sqrt(lig_s[:, np.newaxis] * rec_s[np.newaxis, :])
        eps_ij = np.sqrt(lig_e[:, np.newaxis] * rec_e[np.newaxis, :])
        sr = sig_ij / r_safe
        E_pair = 4.0 * eps_ij * (sr ** 12)
    elif grid_type == 'lja':
        sig_ij = np.sqrt(lig_s[:, np.newaxis] * rec_s[np.newaxis, :])
        eps_ij = np.sqrt(lig_e[:, np.newaxis] * rec_e[np.newaxis, :])
        sr = sig_ij / r_safe
        E_pair = -4.0 * eps_ij * (sr ** 6)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    return np.sum(E_pair)


def load_systems_list(systems_file=DEFAULT_SYSTEMS_FILE):
    """Load list of system names from file."""
    with open(systems_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_system_paths(system_name, base_dir=ASTEX_BASE):
    """Get paths to all files for a system."""
    return {
        'ligand_prmtop': os.path.join(base_dir, '1-build', system_name, 'ligand.prmtop'),
        'receptor_prmtop': os.path.join(base_dir, '1-build', system_name, 'receptor.prmtop'),
        'ligand_inpcrd': os.path.join(base_dir, '3-grids', system_name, 'ligand.trans.inpcrd'),
        'receptor_inpcrd': os.path.join(base_dir, '3-grids', system_name, 'receptor.trans.inpcrd'),
        'docked_poses': os.path.join(base_dir, '4-UCSF_dock6', system_name, 'anchor_and_grow_scored.mol2'),
        'xtal_pose': os.path.join(base_dir, '4-UCSF_dock6', system_name, 'xtal_scored.mol2'),
        'rec_box': os.path.join(base_dir, '4-UCSF_dock6', system_name, 'rec_box.pdb'),
    }


def validate_system_paths(paths):
    """Check that all required files exist."""
    missing = []
    for key, path in paths.items():
        if not os.path.exists(path):
            missing.append(f"{key}: {path}")
    return missing


def generate_grid(paths, grid_spacing, grid_cap, grid_type, platform, grid_file, platform_properties=None):
    """Generate a grid and save to file.

    Returns:
        Dict with grid parameters: counts, origin_nm, spacing_nm
    """
    if platform_properties is None:
        platform_properties = {}
    receptor_prmtop = AmberPrmtopFile(paths['receptor_prmtop'])
    receptor_inpcrd = AmberInpcrdFile(paths['receptor_inpcrd'])
    receptor_atoms = list(range(receptor_prmtop.topology.getNumAtoms()))
    rec_pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                     p[2].value_in_unit(nanometer)) for p in receptor_inpcrd.positions]

    box_params = parse_rec_box_pdb(paths['rec_box'])
    grid_bounds = get_grid_bounds_for_openmm(box_params, grid_spacing)

    system = receptor_prmtop.createSystem()
    grid = gfp.GridForce()
    grid.setGridOrigin(*grid_bounds['origin_nm'])
    grid.addGridCounts(*grid_bounds['grid_counts'])
    grid.addGridSpacing(grid_bounds['spacing_nm'], grid_bounds['spacing_nm'], grid_bounds['spacing_nm'])
    grid.setAutoGenerateGrid(True)
    grid.setGridType(grid_type)
    grid.setGridCap(grid_cap)

    # Always compute derivatives for Hessian analysis
    grid.setComputeDerivatives(True)

    # No inv_power for simplicity - use raw values
    grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

    grid.setReceptorAtoms(receptor_atoms)
    grid.setReceptorPositionsFromLists(rec_pos_list)

    system.addForce(grid)
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, platform, platform_properties)
    context.setPositions(receptor_inpcrd.positions)
    context.getState(getEnergy=True)  # Trigger generation

    # Save grid to standard file
    grid.saveToFile(grid_file)

    # Return all grid parameters needed for evaluation
    grid_params = {
        'counts': grid_bounds['grid_counts'],
        'origin_nm': grid_bounds['origin_nm'],
        'spacing_nm': grid_bounds['spacing_nm']
    }

    del context, integrator, system, grid
    gc.collect()

    return grid_params


def create_evaluation_system(ligand_prmtop, lig_params, grid_files, method):
    """Create an OpenMM system with GridForce for ligand evaluation.

    Args:
        ligand_prmtop: AmberPrmtopFile for ligand
        lig_params: List of (charge, sigma, epsilon) tuples
        grid_files: Dict mapping grid_type -> grid_file_path
        method: Interpolation method (1=bspline, 3=triquintic)

    Returns:
        Tuple of (system, grid_forces_dict, isolated_nb_force) where grid_forces_dict maps
        grid_type -> GridForce object for combined Hessian computation
    """
    n_atoms = len(lig_params)

    # Create system WITH bonded forces from prmtop (bonds, angles, torsions)
    # This is critical - without internal forces, atoms can collapse on each other
    system = ligand_prmtop.createSystem()

    # Assign all bonded forces to group 4 (before we modify the force list)
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if not isinstance(f, NonbondedForce):
            f.setForceGroup(4)  # Bonded forces in group 4

    # Find and extract NonbondedForce parameters before removing it
    nb_force = None
    nb_force_idx = None
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), NonbondedForce):
            nb_force = system.getForce(i)
            nb_force_idx = i
            break

    # Create IsolatedNonbondedForce from the NonbondedForce parameters
    isolated_nb_force = None
    if nb_force is not None:
        isolated_nb_force = gfp.IsolatedNonbondedForce()
        isolated_nb_force.setNumAtoms(n_atoms)
        isolated_nb_force.setParticles(list(range(n_atoms)))

        # Copy atom parameters
        for i in range(n_atoms):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            isolated_nb_force.setAtomParameters(
                i,
                charge.value_in_unit(elementary_charge),
                sigma.value_in_unit(nanometer),
                epsilon.value_in_unit(kilojoules_per_mole)
            )

        # Copy exceptions (1-4 scaled interactions)
        for i in range(nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
            chargeProd_val = chargeProd.value_in_unit(elementary_charge**2)
            sigma_val = sigma.value_in_unit(nanometer)
            epsilon_val = epsilon.value_in_unit(kilojoules_per_mole)

            # If both are zero, it's a complete exclusion (1-2 or 1-3)
            if abs(chargeProd_val) < 1e-10 and abs(epsilon_val) < 1e-10:
                isolated_nb_force.addExclusion(p1, p2)
            else:
                # It's a 1-4 exception with scaled parameters
                isolated_nb_force.addException(p1, p2, chargeProd_val, sigma_val, epsilon_val)

        isolated_nb_force.setForceGroup(3)  # IsolatedNonbondedForce in group 3
        system.addForce(isolated_nb_force)

        # Now remove the original NonbondedForce
        # Note: force index may have changed after adding IsolatedNonbondedForce
        for i in range(system.getNumForces()):
            if isinstance(system.getForce(i), NonbondedForce):
                system.removeForce(i)
                break

    # Keep references to all GridForces for combined Hessian computation
    grid_forces_dict = {}

    # Add a GridForce for each grid type
    for grid_type in GRID_TYPES:
        if grid_type not in grid_files:
            continue

        grid_file = grid_files[grid_type]

        # Build scaling factors based on grid type
        if grid_type == 'charge':
            scaling_factors = [p[0] for p in lig_params]  # charges
        elif grid_type == 'ljr':
            # LJR scaling: sqrt(epsilon) * 2 * sigma^6
            scaling_factors = [np.sqrt(p[2]) * 2.0 * (p[1]**6) for p in lig_params]
        elif grid_type == 'lja':
            # LJA scaling: sqrt(epsilon) * sqrt(2) * sigma^3
            scaling_factors = [np.sqrt(p[2]) * np.sqrt(2.0) * (p[1]**3) for p in lig_params]

        grid_force = gfp.GridForce()
        grid_force.loadFromFile(grid_file)
        grid_force.setInterpolationMethod(method)

        # Apply inv_power transformation for LJ grids (runtime mode)
        # Explicitly set NONE for charge to ensure no transformation
        inv_power = GRID_INV_POWER.get(grid_type)
        if inv_power is not None:
            grid_force.setInvPowerMode(gfp.InvPowerMode_RUNTIME, inv_power)
        else:
            grid_force.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        # Add all atoms as a single particle group
        particle_indices = list(range(n_atoms))
        grid_force.addParticleGroup(f'{grid_type}', particle_indices, scaling_factors)

        # Assign force groups: charge=0, ljr=1, lja=2
        force_group_map = {'charge': 0, 'ljr': 1, 'lja': 2}
        grid_force.setForceGroup(force_group_map[grid_type])

        system.addForce(grid_force)
        grid_forces_dict[grid_type] = grid_force

    return system, grid_forces_dict, isolated_nb_force


def create_gas_phase_system(ligand_prmtop, lig_params):
    """Create an OpenMM system for gas-phase ligand (bonded + intramolecular nonbonded only).

    This is used as a reference state for computing ΔS_binding = S_bound - S_gas.

    Args:
        ligand_prmtop: AmberPrmtopFile for ligand
        lig_params: List of (charge, sigma, epsilon) tuples

    Returns:
        Tuple of (system, isolated_nb_force)
    """
    n_atoms = len(lig_params)

    # Create system with bonded forces from prmtop
    system = ligand_prmtop.createSystem()

    # Assign all bonded forces to group 4
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if not isinstance(f, NonbondedForce):
            f.setForceGroup(4)

    # Find NonbondedForce and create IsolatedNonbondedForce
    nb_force = None
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), NonbondedForce):
            nb_force = system.getForce(i)
            break

    isolated_nb_force = None
    if nb_force is not None:
        isolated_nb_force = gfp.IsolatedNonbondedForce()
        isolated_nb_force.setNumAtoms(n_atoms)
        isolated_nb_force.setParticles(list(range(n_atoms)))

        for i in range(n_atoms):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            isolated_nb_force.setAtomParameters(
                i,
                charge.value_in_unit(elementary_charge),
                sigma.value_in_unit(nanometer),
                epsilon.value_in_unit(kilojoules_per_mole)
            )

        for i in range(nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
            chargeProd_val = chargeProd.value_in_unit(elementary_charge**2)
            sigma_val = sigma.value_in_unit(nanometer)
            epsilon_val = epsilon.value_in_unit(kilojoules_per_mole)

            if abs(chargeProd_val) < 1e-10 and abs(epsilon_val) < 1e-10:
                isolated_nb_force.addExclusion(p1, p2)
            else:
                isolated_nb_force.addException(p1, p2, chargeProd_val, sigma_val, epsilon_val)

        isolated_nb_force.setForceGroup(3)
        system.addForce(isolated_nb_force)

        # Remove original NonbondedForce
        for i in range(system.getNumForces()):
            if isinstance(system.getForce(i), NonbondedForce):
                system.removeForce(i)
                break

    return system, isolated_nb_force


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


def minimize_pose(context, tolerance=DEFAULT_MIN_TOL, max_iterations=DEFAULT_MIN_STEPS,
                  use_newton=False, system=None):
    """Minimize a pose using L-BFGS or Newton-Raphson.

    Args:
        context: OpenMM Context
        tolerance: Energy tolerance (kJ/mol/nm for L-BFGS, kJ/mol for Newton)
        max_iterations: Maximum iterations
        use_newton: If True, use Newton-Raphson minimizer with analytical Hessians
        system: OpenMM System (required for Newton minimizer)

    Returns:
        Tuple of (initial_energy, final_energy, minimization_time_ms, converged)
    """
    initial_state = context.getState(getEnergy=True)
    E_initial = initial_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    start_time = time.time()

    if use_newton:
        if system is None:
            raise ValueError("System required for Newton minimizer")
        minimizer = NewtonMinimizer()
        converged = minimizer.minimize(context, tolerance, max_iterations)
    else:
        LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
        converged = True  # L-BFGS doesn't return convergence status

    min_time = (time.time() - start_time) * 1000

    final_state = context.getState(getEnergy=True)
    E_final = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    return E_initial, E_final, min_time, converged


def compute_combined_hessian_analysis(context, grid_forces_dict, system=None,
                                       isolated_nb_force=None, temperature=300.0):
    """Compute combined Hessian analysis from bonded + grid + isolated nonbonded forces.

    Sums the Hessian contributions from:
    - BondedHessian (bonds, angles, torsions) - diagonal blocks extracted
    - GridForce (ligand-receptor grid interactions) - diagonal blocks
    - IsolatedNonbondedForce (intra-ligand nonbonded) - diagonal blocks extracted

    Args:
        context: OpenMM Context
        grid_forces_dict: Dict mapping grid_type -> GridForce object
        system: OpenMM System (required for bonded Hessian)
        isolated_nb_force: IsolatedNonbondedForce (optional, for intra-ligand nonbonded Hessian)
        temperature: Temperature in Kelvin for entropy calculation

    Returns:
        Dict with per-atom metrics (same format as getHessianAnalysis)
    """
    # Trigger energy calculation to ensure positions are current
    context.getState(getEnergy=True)

    n_atoms = context.getSystem().getNumParticles()
    combined_blocks = np.zeros(n_atoms * 6)

    # Add bonded Hessian contribution (diagonal blocks)
    if system is not None:
        bonded_hessian = BondedHessian()
        bonded_hessian.initialize(system, context)
        full_H = np.array(bonded_hessian.computeHessian(context))

        # Extract diagonal 3x3 blocks for each atom
        n3 = 3 * n_atoms
        full_H = full_H.reshape(n3, n3)
        for i in range(n_atoms):
            base = 3 * i
            # Extract diagonal block elements: dxx, dyy, dzz, dxy, dxz, dyz
            combined_blocks[6*i + 0] += full_H[base+0, base+0]  # dxx
            combined_blocks[6*i + 1] += full_H[base+1, base+1]  # dyy
            combined_blocks[6*i + 2] += full_H[base+2, base+2]  # dzz
            combined_blocks[6*i + 3] += full_H[base+0, base+1]  # dxy
            combined_blocks[6*i + 4] += full_H[base+0, base+2]  # dxz
            combined_blocks[6*i + 5] += full_H[base+1, base+2]  # dyz

    # Add IsolatedNonbondedForce Hessian contribution (diagonal blocks)
    if isolated_nb_force is not None:
        full_H_nb = np.array(isolated_nb_force.computeHessian(context))
        n3 = 3 * n_atoms
        full_H_nb = full_H_nb.reshape(n3, n3)
        for i in range(n_atoms):
            base = 3 * i
            combined_blocks[6*i + 0] += full_H_nb[base+0, base+0]  # dxx
            combined_blocks[6*i + 1] += full_H_nb[base+1, base+1]  # dyy
            combined_blocks[6*i + 2] += full_H_nb[base+2, base+2]  # dzz
            combined_blocks[6*i + 3] += full_H_nb[base+0, base+1]  # dxy
            combined_blocks[6*i + 4] += full_H_nb[base+0, base+2]  # dxz
            combined_blocks[6*i + 5] += full_H_nb[base+1, base+2]  # dyz

    # Add GridForce Hessian contributions (diagonal blocks)
    for grid_type, grid_force in grid_forces_dict.items():
        grid_force.computeHessian(context)
        blocks = np.array(grid_force.getHessianBlocks(context))
        if len(blocks) == len(combined_blocks):
            combined_blocks += blocks

    if n_atoms == 0:
        raise ValueError("No atoms in system")

    # Reshape to [n_atoms, 6] where 6 = (dxx, dyy, dzz, dxy, dxz, dyz)
    blocks_2d = combined_blocks.reshape(n_atoms, 6)

    # Perform eigenanalysis in Python
    kB = 8.314462618e-3  # kJ/(mol*K)
    kT = kB * temperature

    eigenvalues = np.zeros((n_atoms, 3))
    mean_curvature = np.zeros(n_atoms)
    total_curvature = np.zeros(n_atoms)
    gaussian_curvature = np.zeros(n_atoms)
    frac_anisotropy = np.zeros(n_atoms)
    entropy = np.zeros(n_atoms)
    min_eigenvalue = np.zeros(n_atoms)
    num_negative = np.zeros(n_atoms, dtype=int)

    for i in range(n_atoms):
        # Reconstruct 3x3 symmetric Hessian matrix
        dxx, dyy, dzz, dxy, dxz, dyz = blocks_2d[i]
        H = np.array([
            [dxx, dxy, dxz],
            [dxy, dyy, dyz],
            [dxz, dyz, dzz]
        ])

        # Compute eigenvalues (sorted ascending by np.linalg.eigh)
        eigs, _ = np.linalg.eigh(H)
        eigenvalues[i] = eigs

        # Curvature metrics
        mean_curvature[i] = np.mean(eigs)
        total_curvature[i] = np.sum(eigs)
        gaussian_curvature[i] = np.prod(eigs)
        min_eigenvalue[i] = eigs[0]  # smallest eigenvalue

        # Count negative eigenvalues
        num_negative[i] = np.sum(eigs < 0)

        # Fractional anisotropy
        mean_eig = np.mean(eigs)
        diff = eigs - mean_eig
        numerator = np.sum(diff**2)
        denominator = np.sum(eigs**2)
        if denominator > 1e-20:
            frac_anisotropy[i] = np.sqrt(numerator / (2 * denominator))

        # Entropy (harmonic approximation, only valid if all eigenvalues positive)
        if np.all(eigs > 1e-10):
            two_pi_kT = 2 * np.pi * kT
            S = 0.5 * np.sum(1 + np.log(two_pi_kT / eigs))
            entropy[i] = S
        else:
            entropy[i] = np.nan

    # Return in same format as getHessianAnalysis
    return {
        'eigenvalues': eigenvalues.flatten().tolist(),
        'mean_curvature': mean_curvature.tolist(),
        'total_curvature': total_curvature.tolist(),
        'gaussian_curvature': gaussian_curvature.tolist(),
        'frac_anisotropy': frac_anisotropy.tolist(),
        'entropy': entropy.tolist(),
        'min_eigenvalue': min_eigenvalue.tolist(),
        'num_negative': num_negative.tolist(),
    }


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
    masses = np.zeros(n_atoms)
    for i in range(n_atoms):
        masses[i] = system.getParticleMass(i).value_in_unit(unit.dalton)
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
        # This has arbitrary reference but differences are meaningful
        S_classical = np.sum(1 - np.log(x))

        # Schlitter entropy: S/kB = (1/2) × Σ ln(1 + e²/x²)
        # where e is Euler's number. This is derived from the covariance matrix
        # formulation and is numerically more stable for low frequencies.
        # Reference: Schlitter, J. Chem. Phys. Lett. 215, 617-621 (1993)
        e_squared = np.e ** 2  # Euler's number squared
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
    # Compute for all three entropy methods
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


def compute_numerical_hessian(context, h=1e-5):
    """Compute Hessian numerically using central differences on forces.

    This is useful for forces not supported by BondedHessian (e.g., CustomExternalForce).

    Args:
        context: OpenMM Context
        h: Finite difference step size (nm)

    Returns:
        H: 3N×3N Hessian matrix (kJ/mol/nm^2)
    """
    from openmm import unit
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    n_atoms = len(positions)
    n_dof = 3 * n_atoms
    H = np.zeros((n_dof, n_dof))

    for i in range(n_dof):
        atom_i, dim_i = i // 3, i % 3

        # +h perturbation
        pos_plus = positions.copy()
        pos_plus[atom_i, dim_i] += h
        context.setPositions(pos_plus * unit.nanometer)
        f_plus = context.getState(getForces=True).getForces(asNumpy=True)
        f_plus = f_plus.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        # -h perturbation
        pos_minus = positions.copy()
        pos_minus[atom_i, dim_i] -= h
        context.setPositions(pos_minus * unit.nanometer)
        f_minus = context.getState(getForces=True).getForces(asNumpy=True)
        f_minus = f_minus.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        # H[i,j] = -dF_j/dx_i
        for j in range(n_dof):
            atom_j, dim_j = j // 3, j % 3
            H[i, j] = -(f_plus[atom_j, dim_j] - f_minus[atom_j, dim_j]) / (2 * h)

    # Restore original positions
    context.setPositions(positions * unit.nanometer)

    # Symmetrize
    return (H + H.T) / 2


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
    of the mass-weighted Hessian. This gives physically meaningful
    absolute entropy values.

    Also computes the covariance matrix C = k_BT * H^-1 (Equation 18).

    Args:
        context: OpenMM Context
        grid_forces_dict: Dict mapping grid_type -> GridForce object
        system: OpenMM System (required for bonded Hessian and masses)
        isolated_nb_force: IsolatedNonbondedForce (optional)
        temperature: Temperature in Kelvin
        n_rigid_modes: Number of rigid-body modes to exclude (default 6: 3 trans + 3 rot)
        eigenvalue_threshold: Threshold for identifying zero eigenvalues
        use_numerical_hessian: If True, compute Hessian numerically (slower but works
                               for any force type including CustomExternalForce)

    Returns:
        Dict with:
            - 'nma_entropy': Total classical configurational entropy (k_B units)
            - 'nma_entropy_per_mode': Average classical entropy per vibrational mode
            - 'nma_quantum_entropy': Quantum harmonic oscillator entropy (k_B units)
            - 'nma_quantum_entropy_per_mode': Average quantum entropy per mode
            - 'eigenvalues': All 3N eigenvalues (sorted ascending)
            - 'vibrational_eigenvalues': 3N-6 non-zero eigenvalues
            - 'frequencies_cm1': Vibrational frequencies in wavenumbers (cm^-1)
            - 'n_negative_modes': Number of negative eigenvalues (indicates instability)
            - 'n_zero_modes': Number of near-zero eigenvalues found
            - 'covariance_trace': Trace of covariance matrix (sum of variances)
            - 'mean_fluctuation': Average atomic fluctuation sqrt(<Δr²>)
            - 'hessian_condition_number': Condition number of Hessian (excluding zeros)
            - 'det_log': ln(det(H)) for the vibrational modes
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
        # Build full 3N×3N Hessian matrix
        H_full = np.zeros((n_dof, n_dof))

        # Add bonded Hessian contribution (full matrix)
        if system is not None:
            bonded_hessian = BondedHessian()
            bonded_hessian.initialize(system, context)
            H_bonded = np.array(bonded_hessian.computeHessian(context)).reshape(n_dof, n_dof)
            H_full += H_bonded

        # Add IsolatedNonbondedForce Hessian (full matrix)
        if isolated_nb_force is not None:
            H_nb = np.array(isolated_nb_force.computeHessian(context)).reshape(n_dof, n_dof)
            H_full += H_nb

        # Add GridForce Hessian contributions (diagonal blocks only - no cross-particle terms)
        for grid_type, grid_force in grid_forces_dict.items():
            grid_force.computeHessian(context)
            blocks = np.array(grid_force.getHessianBlocks(context))
            # blocks is [n_atoms * 6] with (dxx, dyy, dzz, dxy, dxz, dyz) per atom
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

        # Symmetrize (should already be symmetric, but numerical safety)
        H_full = (H_full + H_full.T) / 2

    # Eigendecomposition of the regular (non-mass-weighted) Hessian
    eigenvalues, eigenvectors = np.linalg.eigh(H_full)

    # Build mass-weighted Hessian for quantum entropy calculation
    # H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
    # Eigenvalues of H_mw are ω² in units of ps^-2 (OpenMM internal units)
    masses = None
    frequencies_cm1 = []
    mw_eigenvalues = None
    if system is not None:
        from openmm import unit
        masses = np.zeros(n_atoms)
        for i in range(n_atoms):
            masses[i] = system.getParticleMass(i).value_in_unit(unit.dalton)

        # Build 3N mass array (each atom's mass appears 3 times for x,y,z)
        mass_3n = np.repeat(masses, 3)
        inv_sqrt_mass = 1.0 / np.sqrt(mass_3n)

        # Mass-weight the Hessian: H_mw = M^(-1/2) H M^(-1/2)
        H_mw = H_full * np.outer(inv_sqrt_mass, inv_sqrt_mass)
        H_mw = (H_mw + H_mw.T) / 2  # Ensure symmetry

        # Eigendecomposition of mass-weighted Hessian
        mw_eigenvalues, _ = np.linalg.eigh(H_mw)

        # Convert eigenvalues (ω² in ps^-2) to frequencies in cm^-1
        # ω = sqrt(λ) in rad/ps (for positive eigenvalues)
        # ν = ω/(2π) in ps^-1 = THz
        # ν̃ = ν/c in cm^-1 where c = 2.998e10 cm/s = 2.998e-2 cm/ps
        # So ν̃ = ω / (2π * 2.998e-2) = ω * 5.309 cm^-1/(rad/ps)
        c_cm_ps = 2.99792458e-2  # speed of light in cm/ps
        for lam in mw_eigenvalues:
            if lam > eigenvalue_threshold:
                omega = np.sqrt(lam)  # rad/ps
                freq_cm1 = omega / (2 * np.pi * c_cm_ps)
                frequencies_cm1.append(freq_cm1)
            elif lam < -eigenvalue_threshold:
                # Imaginary frequency (negative eigenvalue)
                omega = np.sqrt(-lam)
                freq_cm1 = -omega / (2 * np.pi * c_cm_ps)  # Negative indicates imaginary
                frequencies_cm1.append(freq_cm1)

    # Identify rigid-body modes (smallest absolute eigenvalues)
    abs_eigs = np.abs(eigenvalues)
    n_zero = np.sum(abs_eigs < eigenvalue_threshold)

    # Use at least n_rigid_modes, but if more are near-zero, note that
    n_modes_to_skip = max(n_rigid_modes, n_zero)

    # Vibrational eigenvalues (skip the smallest n_modes_to_skip)
    vibrational_eigs = eigenvalues[n_modes_to_skip:]

    # Count negative eigenvalues in vibrational modes (indicates instability/saddle point)
    n_negative = np.sum(vibrational_eigs < -eigenvalue_threshold)

    # Compute entropy for positive vibrational modes only
    # S = (k_B/2) * Σ [1 + ln(2πk_BT/λ)]
    kB = 8.314462618e-3  # kJ/(mol*K) - Boltzmann constant
    kT = kB * temperature
    two_pi_kT = 2 * np.pi * kT

    positive_vib_eigs = vibrational_eigs[vibrational_eigs > eigenvalue_threshold]

    if len(positive_vib_eigs) > 0:
        # Entropy contribution from each mode
        entropy_per_mode = 0.5 * (1 + np.log(two_pi_kT / positive_vib_eigs))
        nma_entropy = np.sum(entropy_per_mode)
        nma_entropy_per_mode = np.mean(entropy_per_mode)

        # Log determinant for vibrational modes
        det_log = np.sum(np.log(positive_vib_eigs))

        # Condition number (ratio of largest to smallest positive eigenvalue)
        condition_number = positive_vib_eigs[-1] / positive_vib_eigs[0]
    else:
        nma_entropy = np.nan
        nma_entropy_per_mode = np.nan
        det_log = np.nan
        condition_number = np.nan

    # Compute quantum harmonic oscillator entropy
    # S_quantum/k_B = Σ [(x/(e^x-1)) - ln(1-e^(-x))]
    # where x = ℏω/(k_BT)
    # Constants in OpenMM internal units (kJ/mol, nm, ps, amu):
    # ℏ/k_B = 7.6382 K·ps
    hbar_over_kB = 7.6382  # K·ps
    nma_quantum_entropy = np.nan
    nma_quantum_entropy_per_mode = np.nan

    if mw_eigenvalues is not None:
        # Get vibrational mass-weighted eigenvalues (skip rigid-body modes)
        mw_vib_eigs = mw_eigenvalues[n_modes_to_skip:]
        positive_mw_vib_eigs = mw_vib_eigs[mw_vib_eigs > eigenvalue_threshold]

        if len(positive_mw_vib_eigs) > 0:
            # ω = sqrt(λ) in rad/ps
            omega = np.sqrt(positive_mw_vib_eigs)

            # x = ℏω/(k_BT) = (ℏ/k_B) * ω / T
            x = hbar_over_kB * omega / temperature

            # Quantum entropy per mode: S/k_B = x/(e^x-1) - ln(1-e^(-x))
            # For numerical stability, handle small and large x separately
            quantum_entropy_per_mode = np.zeros_like(x)
            for i, xi in enumerate(x):
                if xi < 1e-10:
                    # Classical limit: S/k_B ≈ 1 - ln(x) as x → 0
                    quantum_entropy_per_mode[i] = 1 - np.log(xi) if xi > 0 else np.nan
                elif xi > 30:
                    # Low-temperature limit: S/k_B ≈ x*e^(-x) ≈ 0
                    quantum_entropy_per_mode[i] = xi * np.exp(-xi)
                else:
                    # General formula
                    exp_x = np.exp(xi)
                    quantum_entropy_per_mode[i] = xi / (exp_x - 1) - np.log(1 - np.exp(-xi))

            nma_quantum_entropy = np.sum(quantum_entropy_per_mode)
            nma_quantum_entropy_per_mode = np.mean(quantum_entropy_per_mode)

            # Schlitter entropy: S/kB = (1/2) × Σ ln(1 + e²/x²)
            # Reference: Schlitter, J. Chem. Phys. Lett. 215, 617-621 (1993)
            # Numerically more stable for low frequencies, derived from covariance matrix
            e_squared = np.e ** 2
            schlitter_per_mode = 0.5 * np.log(1 + e_squared / (x ** 2))
            nma_schlitter = np.sum(schlitter_per_mode)
            nma_schlitter_per_mode = np.mean(schlitter_per_mode)
    else:
        nma_schlitter = np.nan
        nma_schlitter_per_mode = np.nan

    # Store x values for analysis
    x_values = x.tolist() if mw_eigenvalues is not None and len(positive_mw_vib_eigs) > 0 else []

    # Compute covariance matrix properties via pseudoinverse
    # C = k_BT * H^-1 (Equation 18)
    # Using pseudoinverse: H^-1 = Σ_k u_k u_k^T / λ_k (for non-zero eigenvalues)
    if len(positive_vib_eigs) > 0:
        # Build pseudoinverse using only vibrational modes
        H_pinv = np.zeros((n_dof, n_dof))
        for k in range(n_modes_to_skip, n_dof):
            if eigenvalues[k] > eigenvalue_threshold:
                u_k = eigenvectors[:, k]
                H_pinv += np.outer(u_k, u_k) / eigenvalues[k]

        # Covariance matrix C = k_BT * H^-1
        C = kT * H_pinv

        # Trace of covariance = sum of variances = Σ <Δx_i²>
        covariance_trace = np.trace(C)

        # Mean atomic fluctuation: average of sqrt(<Δr_i²>) where <Δr_i²> = C[3i,3i] + C[3i+1,3i+1] + C[3i+2,3i+2]
        atomic_msf = np.zeros(n_atoms)
        for i in range(n_atoms):
            base = 3 * i
            atomic_msf[i] = C[base, base] + C[base+1, base+1] + C[base+2, base+2]
        mean_fluctuation = np.sqrt(np.mean(atomic_msf))  # in nm

        # Per-atom B-factors (crystallographic): B = 8π²/3 * <Δr²>
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
        'x_values': x_values,  # ℏω/(k_BT) for each mode
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
                         nma_analysis=None, gas_nma_analysis=None):
    """Create a result record with all metrics for a single pose.

    Args:
        ... (existing args)
        nma_analysis: Optional dict from compute_full_nma_entropy() for bound state
        gas_nma_analysis: Optional dict from compute_gas_phase_nma() for reference state
    """

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
    # Eigenvalue statistics (over all atoms)
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

        # NMA entropy (full system, proper normal mode analysis)
        # These use the full 3N×3N Hessian and exclude 6 rigid-body modes
        # Classical entropy has arbitrary reference; quantum entropy gives absolute values
        # Unit conversions: k_B = 8.314e-3 kJ/(mol·K) = 1.987e-3 kcal/(mol·K)
        'nma_entropy_kB': nma_analysis['nma_entropy'] if nma_analysis else np.nan,  # Classical, in k_B units
        'nma_entropy_J_mol_K': (nma_analysis['nma_entropy'] * 8.314462618) if nma_analysis else np.nan,  # Classical, J/(mol·K)
        'nma_entropy_cal_mol_K': (nma_analysis['nma_entropy'] * 1.987204) if nma_analysis else np.nan,  # Classical, cal/(mol·K)
        'nma_minusTS_kJ_mol': (-nma_analysis['nma_entropy'] * 8.314462618e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,  # Classical, -TS in kJ/mol
        'nma_minusTS_kcal_mol': (-nma_analysis['nma_entropy'] * 1.987204e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,  # Classical, -TS in kcal/mol
        # Quantum harmonic oscillator entropy (physically meaningful absolute values)
        'nma_quantum_entropy_kB': nma_analysis.get('nma_quantum_entropy', np.nan) if nma_analysis else np.nan,  # Quantum, in k_B units
        'nma_quantum_entropy_J_mol_K': (nma_analysis.get('nma_quantum_entropy', np.nan) * 8.314462618) if nma_analysis else np.nan,  # Quantum, J/(mol·K)
        'nma_quantum_entropy_cal_mol_K': (nma_analysis.get('nma_quantum_entropy', np.nan) * 1.987204) if nma_analysis else np.nan,  # Quantum, cal/(mol·K)
        'nma_quantum_minusTS_kJ_mol': (-nma_analysis.get('nma_quantum_entropy', np.nan) * 8.314462618e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,  # Quantum, -TS in kJ/mol
        'nma_quantum_minusTS_kcal_mol': (-nma_analysis.get('nma_quantum_entropy', np.nan) * 1.987204e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,  # Quantum, -TS in kcal/mol
        # Schlitter entropy (covariance-based, numerically stable)
        'nma_schlitter_kB': nma_analysis.get('nma_schlitter', np.nan) if nma_analysis else np.nan,
        'nma_schlitter_minusTS_kcal_mol': (-nma_analysis.get('nma_schlitter', np.nan) * 1.987204e-3 * nma_analysis['temperature']) if nma_analysis else np.nan,
        'nma_mean_x': nma_analysis.get('mean_x', np.nan) if nma_analysis else np.nan,  # Mean ℏω/(k_BT)
        'nma_n_vibrational_modes': nma_analysis['n_vibrational_modes'] if nma_analysis else np.nan,
        'nma_n_negative_modes': nma_analysis['n_negative_modes'] if nma_analysis else np.nan,
        'nma_n_zero_modes': nma_analysis['n_zero_modes'] if nma_analysis else np.nan,
        'nma_covariance_trace': nma_analysis['covariance_trace'] if nma_analysis else np.nan,
        'nma_mean_fluctuation_nm': nma_analysis['mean_fluctuation'] if nma_analysis else np.nan,
        'nma_condition_number': nma_analysis['hessian_condition_number'] if nma_analysis else np.nan,
        'nma_det_log': nma_analysis['det_log'] if nma_analysis else np.nan,

        # Gas-phase (free ligand) NMA entropy for delta calculation
        'gas_quantum_entropy_kB': gas_nma_analysis.get('nma_quantum_entropy', np.nan) if gas_nma_analysis else np.nan,
        'gas_quantum_minusTS_kcal_mol': (-gas_nma_analysis.get('nma_quantum_entropy', np.nan) * 1.987204e-3 * gas_nma_analysis['temperature']) if gas_nma_analysis else np.nan,
        'gas_E_kJ_mol': gas_nma_analysis.get('E_gas', np.nan) if gas_nma_analysis else np.nan,
        'gas_mean_x': gas_nma_analysis.get('mean_x', np.nan) if gas_nma_analysis else np.nan,

        # Delta entropy: ΔS = S_bound - S_gas (negative = entropy loss upon binding)
        # Use direct computation from same geometry (more reliable than separate calculations)
        'delta_quantum_entropy_kB': nma_analysis.get('direct_delta_S_quantum', np.nan) if nma_analysis else np.nan,
        # Delta entropy (-TΔS) for all three methods (kcal/mol)
        'delta_quantum_minusTS_kcal_mol': nma_analysis.get('direct_delta_minusTS_quantum', np.nan) if nma_analysis else np.nan,
        'delta_classical_minusTS_kcal_mol': nma_analysis.get('direct_delta_minusTS_classical', np.nan) if nma_analysis else np.nan,
        'delta_schlitter_minusTS_kcal_mol': nma_analysis.get('direct_delta_minusTS_schlitter', np.nan) if nma_analysis else np.nan,

        # Thermodynamic quantities (ΔH = grid interaction energy, ΔG = ΔH - TΔS)
        'delta_H_kcal_mol': nma_analysis.get('direct_delta_H_kcal', np.nan) if nma_analysis else np.nan,
        'delta_G_quantum_kcal_mol': nma_analysis.get('direct_delta_G_quantum_kcal', np.nan) if nma_analysis else np.nan,
        'delta_G_classical_kcal_mol': nma_analysis.get('direct_delta_G_classical_kcal', np.nan) if nma_analysis else np.nan,
        'delta_G_schlitter_kcal_mol': nma_analysis.get('direct_delta_G_schlitter_kcal', np.nan) if nma_analysis else np.nan,

        # Per-grid energy breakdown (kcal/mol)
        'E_charge_kcal_mol': nma_analysis.get('E_grid_by_type', {}).get('charge', np.nan) if nma_analysis else np.nan,
        'E_ljr_kcal_mol': nma_analysis.get('E_grid_by_type', {}).get('ljr', np.nan) if nma_analysis else np.nan,
        'E_lja_kcal_mol': nma_analysis.get('E_grid_by_type', {}).get('lja', np.nan) if nma_analysis else np.nan,

        # Reference pairwise energies (for validation)
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


def save_result(output_file, result):
    """Append a single result to the CSV file."""
    df_new = pd.DataFrame([result])

    if os.path.exists(output_file):
        df_new.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(output_file, index=False)


def process_system(system_name, paths, poses, lig_params, platform,
                   methods, grid_spacing, output_file,
                   min_tolerance, min_steps, use_newton=False,
                   rec_params=None, rec_positions_nm=None,
                   platform_properties=None, use_cpu_gas_hessian=False,
                   separate_gas_min=False):
    """Process all poses for a system with all methods.

    Generates grids per-method (different methods may use different caps),
    then evaluates all poses.
    """
    if platform_properties is None:
        platform_properties = {}
    n_atoms = len(lig_params)
    ligand_prmtop = AmberPrmtopFile(paths['ligand_prmtop'])

    print(f"  Ligand has {n_atoms} atoms")
    print(f"  Processing {len(poses)} poses")

    with tempfile.TemporaryDirectory() as tmpdir:
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
                params = generate_grid(
                    paths, grid_spacing, method_cap, grid_type,
                    platform, grid_file, platform_properties
                )
                gen_time = (time.time() - gen_start) * 1000
                total_gen_time += gen_time

                grid_files[grid_type] = grid_file
                print(f"{gen_time:.0f} ms (grid: {params['counts']})")

            gen_time_per_pose = total_gen_time / len(poses) if poses else 0

            # Create system with all grids
            try:
                system, grid_forces_dict, isolated_nb_force = create_evaluation_system(
                    ligand_prmtop, lig_params, grid_files, method
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

            # Create context
            integrator = VerletIntegrator(0.001)
            context = Context(system, integrator, platform, platform_properties)

            # Process each pose
            for pose_idx, pose_data in enumerate(poses):
                pose_coords_nm = pose_data['coordinates'] * 0.1  # Angstroms to nm

                if len(pose_coords_nm) != n_atoms:
                    print(f"      Pose {pose_idx}: atom count mismatch ({len(pose_coords_nm)} vs {n_atoms})")
                    continue

                # Set initial positions
                positions = [Vec3(x, y, z) * nanometer for x, y, z in pose_coords_nm]
                context.setPositions(positions)

                # Minimize
                print(f"      Pose {pose_idx}/{len(poses)}: minimizing...", end=" ")
                sys.stdout.flush()

                try:
                    E_initial, E_final, min_time, converged = minimize_pose(
                        context, min_tolerance, min_steps,
                        use_newton=use_newton, system=system
                    )
                    status = "converged" if converged else "NOT CONVERGED"
                    print(f"E: {E_initial:.1f} -> {E_final:.1f} kJ/mol ({min_time:.0f} ms, {status})", end=" ")

                    # Warn if energy increased or minimizer didn't converge
                    if E_final > E_initial:
                        print(f"\n        WARNING: Energy increased during minimization!", end=" ")
                    if not converged:
                        print(f"\n        WARNING: Minimizer did not converge after {min_steps} iterations", end=" ")
                except Exception as e:
                    print(f"minimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Compute combined Hessian analysis from bonded + grid forces
                print("Hessian...", end=" ")
                sys.stdout.flush()

                try:
                    hess_start = time.time()
                    analysis = compute_combined_hessian_analysis(
                        context, grid_forces_dict, system=system,
                        isolated_nb_force=isolated_nb_force, temperature=300.0
                    )
                    hess_time = (time.time() - hess_start) * 1000
                    print(f"{hess_time:.0f} ms", end=" ")
                except Exception as e:
                    print(f"failed: {e}")
                    continue

                # Compute NMA entropy for bound and gas phase
                if separate_gas_min:
                    print("NMA(bound+separate gas)...", end=" ")
                else:
                    print("NMA(bound+gas)...", end=" ")
                sys.stdout.flush()
                try:
                    nma_start = time.time()
                    # First compute the full bound-state NMA (for backward compatibility)
                    nma_analysis = compute_full_nma_entropy(
                        context, grid_forces_dict, system=system,
                        isolated_nb_force=isolated_nb_force, temperature=300.0
                    )

                    if separate_gas_min:
                        # Compute gas phase with SEPARATE minimization (different geometry)
                        # This gives larger entropy changes reflecting conformational relaxation
                        from openmm import unit

                        # Get grid energies BEFORE creating new context (avoids CUDA context issues)
                        E_grid = 0.0
                        E_grid_by_type = {}
                        for grid_type, grid_force in grid_forces_dict.items():
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
                            # Compute delta values from separate calculations
                            kB_kcal = 1.987204e-3  # kcal/(mol·K)
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
                        # Compute both bound and gas at same geometry for delta
                        combined_nma = compute_nma_with_and_without_grids(
                            context, grid_forces_dict, system=system,
                            isolated_nb_force=isolated_nb_force, temperature=300.0,
                            use_cpu_gas_hessian=use_cpu_gas_hessian,
                            ligand_prmtop=ligand_prmtop
                        )

                        # Create a gas_nma_analysis dict for compatibility with result record
                        gas_nma_analysis = {
                            'nma_quantum_entropy': combined_nma['gas']['quantum_entropy_kB'],
                            'mean_x': combined_nma['gas']['mean_x'],
                            'temperature': 300.0,
                            'E_gas': combined_nma['gas'].get('E_kJ_mol', np.nan),
                        }
                        # Override the delta values with the direct computation (all three methods)
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
                        # Per-grid energy contributions
                        nma_analysis['E_grid_by_type'] = combined_nma.get('E_grid_by_type', {})

                    # Compute reference pairwise energies at minimized geometry
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
                            ref_energies[grid_type] = ref_E / 4.184  # kJ/mol -> kcal/mol
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
                    gas_nma_analysis=gas_nma_analysis
                )

                # Save immediately
                save_result(output_file, result)

                # Print summary including thermodynamic quantities
                # ΔH = grid interaction energy, -TΔS = entropy penalty, ΔG = ΔH - TΔS
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

            # Clean up context
            del context, integrator, system
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Hessian Analysis Benchmark')
    parser.add_argument('--systems', type=str, default='1g9v',
                        help='Systems to test: all, or comma-separated (1g9v,1gkc,...)')
    parser.add_argument('--methods', type=str, default='bspline,triquintic',
                        help='Methods: all, bspline, triquintic, or comma-separated')
    parser.add_argument('--grid-spacing', type=float, default=DEFAULT_GRID_SPACING,
                        help=f'Grid spacing in Angstroms (default: {DEFAULT_GRID_SPACING})')
    # Note: grid cap is set per-method (bspline=41840, triquintic=1e30)
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
                        help='Use Newton-Raphson minimizer (default: True, required for IsolatedNonbondedForce)')
    parser.add_argument('--use-lbfgs', action='store_true',
                        help='Use L-BFGS minimizer (not compatible with IsolatedNonbondedForce)')
    parser.add_argument('--native', action='store_true',
                        help='Use native (crystallographic) pose instead of docked poses')
    parser.add_argument('--cpu-gas-hessian', action='store_true',
                        help='Compute gas phase Hessian on CPU for determinism (slower)')
    parser.add_argument('--separate-gas-min', action='store_true',
                        help='Minimize gas phase separately (different geometry from bound state). '
                             'This gives larger entropy changes that reflect conformational relaxation.')

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
        methods = [1, 3]  # bspline, triquintic
    else:
        method_map = {'bspline': 1, 'triquintic': 3}
        methods = []
        for m in args.methods.split(','):
            m = m.strip().lower()
            if m in method_map:
                methods.append(method_map[m])
            elif m.isdigit():
                methods.append(int(m))

    # Check platform and set precision
    try:
        platform = Platform.getPlatformByName(args.platform)
    except Exception:
        print(f"Platform {args.platform} not available")
        sys.exit(1)

    # Platform properties - don't set precision to avoid kernel compatibility issues
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
    print(f"Output: {args.output}")
    print(f"Platform: {args.platform}")
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

        # Load receptor parameters and positions for reference energy calculation
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

        # Load poses (native or docked)
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
                separate_gas_min=args.separate_gas_min
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
