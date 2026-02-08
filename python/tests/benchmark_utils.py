#!/usr/bin/env python
"""
Utility functions for Hessian analysis benchmarks.

This module provides all the utility functions needed for benchmark_hessian_analysis.py,
organized into logical sections:
- Constants and configuration
- Directory and memory utilities
- Path and file utilities
- System creation (grids, forces)
- Minimization
- Hessian analysis and NMA
- Result handling
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gridforceplugin as gfp
from openmm.app import AmberPrmtopFile, AmberInpcrdFile
from openmm import Platform, Context, VerletIntegrator, LocalEnergyMinimizer
from openmm import NonbondedForce, CustomExternalForce, Vec3
from gridforceplugin import NewtonMinimizer, BondedHessian
from openmm.unit import nanometer, kilojoules_per_mole, elementary_charge

# Import pose parser from example directory
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'example')
sys.path.insert(0, EXAMPLE_DIR)
from parse_dock6_poses import parse_mol2_file, parse_rec_box_pdb, get_grid_bounds_for_openmm

# =============================================================================
# CONSTANTS
# =============================================================================

ONE_4PI_EPS0 = 138.935456  # kJ/mol * nm / e^2
METHOD_NAMES = {1: 'bspline', 3: 'triquintic', 4: 'quintic_bspline'}
GRID_TYPES = ['charge', 'ljr', 'lja']

# Inv-power settings per grid type (None = no transformation)
# LJ grids need transformation to tame steep r^-12 and r^-6 potentials
GRID_INV_POWER = {
    'charge': None,
    'ljr': -6.0,   # Soften r^-12 -> r^-2 behavior
    'lja': -2.0,   # Soften r^-6 -> r^-3 behavior
}

# Arcsinh scale per grid type (0.0 = disabled)
# For quintic B-spline: arcsinh compression helps LJ grids with large dynamic range
GRID_ARCSINH_SCALE = {
    'charge': 0.0,   # No arcsinh for charge (moderate dynamic range)
    'ljr': 100.0,    # Compress LJ repulsive dynamic range
    'lja': 100.0,    # Compress LJ attractive dynamic range
}

# B-spline prefilter order per grid type (0 = none, 5 = quintic)
# Prefiltering converts grid samples to B-spline coefficients for exact interpolation
GRID_PREFILTER_ORDER = {
    'charge': 0,     # No prefiltering for charge (raw quintic bspline)
    'ljr': 5,        # Quintic prefilter for LJ repulsive
    'lja': 5,        # Quintic prefilter for LJ attractive
}

# Default paths
ASTEX_BASE = '/scratch/AstexDiv_comprehensive'
DEFAULT_SYSTEMS_FILE = os.path.join(ASTEX_BASE, 'systems.txt')

# Grid settings
DEFAULT_GRID_SPACING = 0.2  # Angstroms
DEFAULT_GRID_CAP = 41840.0  # kJ/mol - for bspline
TRIQUINTIC_GRID_CAP = 1e30  # kJ/mol - for triquintic (uncapped to avoid NaN with inv_power)

QUINTIC_BSPLINE_GRID_CAP = 1e30  # kJ/mol - for quintic bspline (uncapped)

METHOD_GRID_CAPS = {
    1: DEFAULT_GRID_CAP,
    3: TRIQUINTIC_GRID_CAP,
    4: QUINTIC_BSPLINE_GRID_CAP,
}

# Tiled grid settings
TILED_THRESHOLD_GB = 4.0
DEFAULT_TILE_SIZE = 32
DEFAULT_TILE_MEMORY_MB = 3072

# Minimization settings
DEFAULT_MIN_TOL = 1.0
DEFAULT_MIN_STEPS = 1000


# =============================================================================
# DIRECTORY AND MEMORY UTILITIES
# =============================================================================

class PersistentDirectory:
    """Context manager for using a fixed directory (won't auto-delete)."""
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def __enter__(self):
        return self.path

    def __exit__(self, *args):
        pass


def aggressive_memory_cleanup():
    """Aggressively free memory after grid operations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def calculate_expected_grid_memory_gb(grid_counts, compute_derivatives=True):
    """Calculate expected GPU memory for a grid before generation."""
    nx, ny, nz = grid_counts
    n_points = nx * ny * nz
    bytes_per_float = 4
    values_bytes = n_points * bytes_per_float
    derivs_bytes = n_points * 27 * bytes_per_float if compute_derivatives else 0
    return (values_bytes + derivs_bytes) / (1024 ** 3)


# =============================================================================
# PATH AND FILE UTILITIES
# =============================================================================

def load_systems_list(systems_file=None):
    """Load list of system names from file."""
    if systems_file is None:
        systems_file = DEFAULT_SYSTEMS_FILE
    with open(systems_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_system_paths(system_name, base_dir=None):
    """Get paths to all files for a system."""
    if base_dir is None:
        base_dir = ASTEX_BASE
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
    """Check that all required files exist. Returns list of missing paths."""
    missing = []
    for key, path in paths.items():
        if not os.path.exists(path):
            missing.append(f"{key}: {path}")
    return missing


# =============================================================================
# RESTRAINT UTILITIES
# =============================================================================

def add_positional_restraints(system, positions, force_constant_kcal=10.0):
    """Add harmonic positional restraints to keep atoms near initial positions.

    Args:
        system: OpenMM System to modify
        positions: Initial positions (list of Vec3 in nm, or numpy array in nm)
        force_constant_kcal: Force constant in kcal/mol/A^2

    Returns:
        The CustomExternalForce object
    """
    # Convert to OpenMM units (kJ/mol/nm^2)
    k_kj_nm2 = force_constant_kcal * 418.4

    restraint_force = CustomExternalForce(
        'k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)'
    )
    restraint_force.addGlobalParameter('k', k_kj_nm2)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')

    for i in range(len(positions)):
        if hasattr(positions[i], 'value_in_unit'):
            x0 = positions[i][0].value_in_unit(nanometer)
            y0 = positions[i][1].value_in_unit(nanometer)
            z0 = positions[i][2].value_in_unit(nanometer)
        elif hasattr(positions[i], '__iter__'):
            x0, y0, z0 = positions[i][0], positions[i][1], positions[i][2]
        else:
            raise ValueError(f"Unknown position format: {type(positions[i])}")
        restraint_force.addParticle(i, [x0, y0, z0])

    system.addForce(restraint_force)
    return restraint_force


# =============================================================================
# GRID GENERATION
# =============================================================================

def generate_grid(paths, grid_spacing, grid_cap, grid_type, platform, grid_file,
                  platform_properties=None, buffer_nm=2.0,
                  tiled_threshold_gb=TILED_THRESHOLD_GB, tile_size=DEFAULT_TILE_SIZE,
                  arcsinh_scale=0.0, prefilter_order=0, compute_derivatives=True):
    """Generate a grid and save to file, using tiled mode for large grids.

    Args:
        arcsinh_scale: If > 0, apply arcsinh(V/scale) compression before prefiltering.
        prefilter_order: B-spline prefilter order (0=none, 5=quintic).
        compute_derivatives: Whether to compute and store derivatives. B-spline methods
            (1, 4) don't need derivatives, so setting False saves memory/time.

    Returns:
        Dict with grid parameters: counts, origin_nm, spacing_nm, is_tiled, output_file
    """
    if platform_properties is None:
        platform_properties = {}

    receptor_prmtop = AmberPrmtopFile(paths['receptor_prmtop'])
    receptor_inpcrd = AmberInpcrdFile(paths['receptor_inpcrd'])
    receptor_atoms = list(range(receptor_prmtop.topology.getNumAtoms()))
    rec_pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                     p[2].value_in_unit(nanometer)) for p in receptor_inpcrd.positions]

    box_params = parse_rec_box_pdb(paths['rec_box'])
    grid_bounds = get_grid_bounds_for_openmm(box_params, grid_spacing, buffer_nm=buffer_nm)

    expected_size_gb = calculate_expected_grid_memory_gb(
        grid_bounds['grid_counts'], compute_derivatives=compute_derivatives)
    use_tiled_generation = (tiled_threshold_gb > 0 and expected_size_gb > tiled_threshold_gb)

    if use_tiled_generation:
        if grid_file.endswith('.grid'):
            tiled_file = grid_file.replace('.grid', '.tiled')
        else:
            tiled_file = grid_file + '.tiled'
        actual_output_file = tiled_file
        print("[Grid {:.1f} GB > {:.1f} GB, using tiled mode]".format(expected_size_gb, tiled_threshold_gb), end=" ")
    else:
        actual_output_file = grid_file

    system = receptor_prmtop.createSystem()
    grid = gfp.GridForce()
    grid.setGridOrigin(*grid_bounds['origin_nm'])
    grid.addGridCounts(*grid_bounds['grid_counts'])
    grid.addGridSpacing(grid_bounds['spacing_nm'], grid_bounds['spacing_nm'], grid_bounds['spacing_nm'])
    grid.setAutoGenerateGrid(True)
    grid.setGridType(grid_type)
    grid.setGridCap(grid_cap)
    grid.setComputeDerivatives(compute_derivatives)
    grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
    grid.setReceptorAtoms(receptor_atoms)
    grid.setReceptorPositionsFromLists(rec_pos_list)

    # Set arcsinh and prefilter for generation (applied after grid values are computed)
    if arcsinh_scale > 0.0:
        grid.setArcsinhScale(arcsinh_scale)
    if prefilter_order > 0:
        grid.setBSplinePrefilterOrder(prefilter_order)

    if use_tiled_generation:
        grid.setTiledOutputFile(actual_output_file, tile_size)

    system.addForce(grid)
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, platform, platform_properties)
    context.setPositions(receptor_inpcrd.positions)
    context.getState(getEnergy=True)  # Trigger generation

    if not use_tiled_generation:
        grid.saveToFile(grid_file)

    grid_params = {
        'counts': grid_bounds['grid_counts'],
        'origin_nm': grid_bounds['origin_nm'],
        'spacing_nm': grid_bounds['spacing_nm'],
        'is_tiled': use_tiled_generation,
        'output_file': actual_output_file
    }

    del context, integrator, system, grid
    gc.collect()

    return grid_params


# =============================================================================
# SYSTEM CREATION
# =============================================================================

def create_evaluation_system(ligand_prmtop, lig_params, grid_files, method,
                             grid_params_dict=None, tile_size=DEFAULT_TILE_SIZE,
                             tile_memory_mb=DEFAULT_TILE_MEMORY_MB,
                             gbsa_params=None):
    """Create an OpenMM system with GridForce for ligand evaluation.

    Args:
        gbsa_params: If provided, dict with keys:
            'lig_radii': list of ligand intrinsic radii (nm)
            'lig_scales': list of ligand OBC scale factors
            'lig_charges': list of ligand charges (e)
            'rec_positions': flat list of receptor positions [x0,y0,z0,x1,...] (nm)
            'rec_radii': list of receptor intrinsic radii (nm)
            'rec_scales': list of receptor OBC scale factors
            'grid_origin': (x, y, z) in nm
            'grid_counts': (nx, ny, nz)
            'grid_spacing': float in nm
            'exclusions': list of (i, j) pairs

    Returns:
        Tuple of (system, grid_forces_dict, isolated_nb_force)
    """
    if grid_params_dict is None:
        grid_params_dict = {}
    n_atoms = len(lig_params)

    system = ligand_prmtop.createSystem()

    # Assign bonded forces to group 4
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

    # Add GridForces
    grid_forces_dict = {}
    for grid_type in GRID_TYPES:
        if grid_type not in grid_files:
            continue

        grid_info = grid_files[grid_type]
        if isinstance(grid_info, dict):
            grid_file = grid_info['output_file']
            is_tiled = grid_info.get('is_tiled', False)
        else:
            grid_file = grid_info
            is_tiled = grid_file.endswith('.tiled')

        # Build scaling factors
        if grid_type == 'charge':
            scaling_factors = [p[0] for p in lig_params]
        elif grid_type == 'ljr':
            scaling_factors = [np.sqrt(p[2]) * 2.0 * (p[1]**6) for p in lig_params]
        elif grid_type == 'lja':
            scaling_factors = [np.sqrt(p[2]) * np.sqrt(2.0) * (p[1]**3) for p in lig_params]

        grid_force = gfp.GridForce()

        if is_tiled:
            grid_params = grid_params_dict.get(grid_type) or (grid_info if isinstance(grid_info, dict) else None)
            if grid_params is None:
                raise ValueError(f"grid_params required for tiled grid: {grid_file}")

            grid_force.setGridOrigin(*grid_params['origin_nm'])
            grid_force.addGridCounts(*grid_params['counts'])
            grid_force.addGridSpacing(grid_params['spacing_nm'], grid_params['spacing_nm'], grid_params['spacing_nm'])
            grid_force.setTiledInputFile(grid_file)
            grid_force.setTiledMode(True, tile_size, tile_memory_mb)
        else:
            grid_force.loadFromFile(grid_file)

        grid_force.setInterpolationMethod(method)

        if method == 4:
            # Quintic B-spline: use arcsinh + prefilter instead of inv_power
            arcsinh_scale = GRID_ARCSINH_SCALE.get(grid_type, 0.0)
            prefilter_order = GRID_PREFILTER_ORDER.get(grid_type, 0)
            if arcsinh_scale > 0.0:
                grid_force.setArcsinhScale(arcsinh_scale)
            if prefilter_order > 0:
                grid_force.setBSplinePrefilterOrder(prefilter_order)
            grid_force.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        else:
            inv_power = GRID_INV_POWER.get(grid_type)
            if inv_power is not None:
                grid_force.setInvPowerMode(gfp.InvPowerMode_RUNTIME, inv_power)
            else:
                grid_force.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        particle_indices = list(range(n_atoms))
        grid_force.addParticleGroup(f'{grid_type}', particle_indices, scaling_factors)

        force_group_map = {'charge': 0, 'ljr': 1, 'lja': 2}
        grid_force.setForceGroup(force_group_map[grid_type])

        system.addForce(grid_force)
        grid_forces_dict[grid_type] = grid_force

    # Add GBSAGridForce if GBSA params provided
    if gbsa_params is not None:
        gbsa_force = gfp.GBSAGridForce()
        gbsa_force.setNumAtoms(n_atoms)
        gbsa_force.setParticles(list(range(n_atoms)))

        for i in range(n_atoms):
            gbsa_force.setAtomParameters(
                i, gbsa_params['lig_charges'][i],
                gbsa_params['lig_radii'][i],
                gbsa_params['lig_scales'][i]
            )

        # Add exclusions
        for i, j in gbsa_params['exclusions']:
            gbsa_force.addExclusion(i, j)

        # Auto-generate grid on GPU
        gbsa_force.setAutoGenerateGrid(True)
        n_rec = len(gbsa_params['rec_radii'])
        gbsa_force.setReceptorAtoms(list(range(n_rec)))
        gbsa_force.setReceptorPositions(gbsa_params['rec_positions'])
        gbsa_force.setReceptorRadii(gbsa_params['rec_radii'])
        gbsa_force.setReceptorScaleFactors(gbsa_params['rec_scales'])

        gbsa_force.setGridOrigin(*gbsa_params['grid_origin'])
        gbsa_force.setGridCounts(*gbsa_params['grid_counts'])
        gbsa_force.setGridSpacing(gbsa_params['grid_spacing'])
        gbsa_force.setInterpolationMethod(0)  # trilinear for GBSA grids
        gbsa_force.setForceGroup(5)
        system.addForce(gbsa_force)
        grid_forces_dict['gbsa'] = gbsa_force

    return system, grid_forces_dict, isolated_nb_force


def create_gas_phase_system(ligand_prmtop, lig_params):
    """Create a gas-phase system (bonded + intramolecular nonbonded only).

    Returns:
        Tuple of (system, isolated_nb_force)
    """
    n_atoms = len(lig_params)
    system = ligand_prmtop.createSystem()

    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if not isinstance(f, NonbondedForce):
            f.setForceGroup(4)

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

        for i in range(system.getNumForces()):
            if isinstance(system.getForce(i), NonbondedForce):
                system.removeForce(i)
                break

    return system, isolated_nb_force


# =============================================================================
# MINIMIZATION
# =============================================================================

def minimize_pose(context, tolerance=DEFAULT_MIN_TOL, max_iterations=DEFAULT_MIN_STEPS,
                  use_newton=False, system=None):
    """Minimize a pose using L-BFGS or Newton-Raphson.

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
        converged = True

    min_time = (time.time() - start_time) * 1000

    final_state = context.getState(getEnergy=True)
    E_final = final_state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    return E_initial, E_final, min_time, converged


# =============================================================================
# HESSIAN ANALYSIS
# =============================================================================

def compute_combined_hessian_analysis(context, grid_forces_dict, system=None,
                                       isolated_nb_force=None, temperature=300.0):
    """Compute combined Hessian analysis from bonded + grid + isolated nonbonded forces.

    Returns:
        Dict with per-atom metrics
    """
    context.getState(getEnergy=True)

    n_atoms = context.getSystem().getNumParticles()
    combined_blocks = np.zeros(n_atoms * 6)

    # Add bonded Hessian
    if system is not None:
        bonded_hessian = BondedHessian()
        bonded_hessian.initialize(system, context)
        full_H = np.array(bonded_hessian.computeHessian(context))
        n3 = 3 * n_atoms
        full_H = full_H.reshape(n3, n3)
        for i in range(n_atoms):
            base = 3 * i
            combined_blocks[6*i + 0] += full_H[base+0, base+0]
            combined_blocks[6*i + 1] += full_H[base+1, base+1]
            combined_blocks[6*i + 2] += full_H[base+2, base+2]
            combined_blocks[6*i + 3] += full_H[base+0, base+1]
            combined_blocks[6*i + 4] += full_H[base+0, base+2]
            combined_blocks[6*i + 5] += full_H[base+1, base+2]

    # Add IsolatedNonbondedForce Hessian
    if isolated_nb_force is not None:
        full_H_nb = np.array(isolated_nb_force.computeHessian(context))
        n3 = 3 * n_atoms
        full_H_nb = full_H_nb.reshape(n3, n3)
        for i in range(n_atoms):
            base = 3 * i
            combined_blocks[6*i + 0] += full_H_nb[base+0, base+0]
            combined_blocks[6*i + 1] += full_H_nb[base+1, base+1]
            combined_blocks[6*i + 2] += full_H_nb[base+2, base+2]
            combined_blocks[6*i + 3] += full_H_nb[base+0, base+1]
            combined_blocks[6*i + 4] += full_H_nb[base+0, base+2]
            combined_blocks[6*i + 5] += full_H_nb[base+1, base+2]

    # Add GridForce Hessians
    for grid_type, grid_force in grid_forces_dict.items():
        if grid_type == 'gbsa':
            # GBSAGridForce: full 3N×3N Hessian (captures cross-atom coupling)
            grid_force.computeHessian(context)
            gbsa_full = np.array(grid_force.getFullHessian(context))
            n3 = 3 * n_atoms
            gbsa_full = gbsa_full.reshape(n3, n3)
            for i in range(n_atoms):
                base = 3 * i
                combined_blocks[6*i + 0] += gbsa_full[base+0, base+0]
                combined_blocks[6*i + 1] += gbsa_full[base+1, base+1]
                combined_blocks[6*i + 2] += gbsa_full[base+2, base+2]
                combined_blocks[6*i + 3] += gbsa_full[base+0, base+1]
                combined_blocks[6*i + 4] += gbsa_full[base+0, base+2]
                combined_blocks[6*i + 5] += gbsa_full[base+1, base+2]
        else:
            grid_force.computeHessian(context)
            blocks = np.array(grid_force.getHessianBlocks(context))
            if len(blocks) == len(combined_blocks):
                combined_blocks += blocks

    if n_atoms == 0:
        raise ValueError("No atoms in system")

    # Eigenanalysis
    blocks_2d = combined_blocks.reshape(n_atoms, 6)
    kB = 8.314462618e-3
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
        dxx, dyy, dzz, dxy, dxz, dyz = blocks_2d[i]
        H = np.array([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
        eigs, _ = np.linalg.eigh(H)
        eigenvalues[i] = eigs
        mean_curvature[i] = np.mean(eigs)
        total_curvature[i] = np.sum(eigs)
        gaussian_curvature[i] = np.prod(eigs)
        min_eigenvalue[i] = eigs[0]
        num_negative[i] = np.sum(eigs < 0)

        mean_eig = np.mean(eigs)
        diff = eigs - mean_eig
        numerator = np.sum(diff**2)
        denominator = np.sum(eigs**2)
        if denominator > 1e-20:
            frac_anisotropy[i] = np.sqrt(numerator / (2 * denominator))

        if np.all(eigs > 1e-10):
            two_pi_kT = 2 * np.pi * kT
            entropy[i] = 0.5 * np.sum(1 + np.log(two_pi_kT / eigs))
        else:
            entropy[i] = np.nan

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


def compute_numerical_hessian(context, h=1e-5):
    """Compute Hessian numerically using central differences on forces."""
    from openmm import unit
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    n_atoms = len(positions)
    n_dof = 3 * n_atoms
    H = np.zeros((n_dof, n_dof))

    for i in range(n_dof):
        atom_i, dim_i = i // 3, i % 3

        pos_plus = positions.copy()
        pos_plus[atom_i, dim_i] += h
        context.setPositions(pos_plus * unit.nanometer)
        f_plus = context.getState(getForces=True).getForces(asNumpy=True)
        f_plus = f_plus.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        pos_minus = positions.copy()
        pos_minus[atom_i, dim_i] -= h
        context.setPositions(pos_minus * unit.nanometer)
        f_minus = context.getState(getForces=True).getForces(asNumpy=True)
        f_minus = f_minus.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        for j in range(n_dof):
            atom_j, dim_j = j // 3, j % 3
            H[i, j] = -(f_plus[atom_j, dim_j] - f_minus[atom_j, dim_j]) / (2 * h)

    context.setPositions(positions * unit.nanometer)
    return (H + H.T) / 2


# =============================================================================
# REFERENCE ENERGY CALCULATION
# =============================================================================

def calculate_ref_energy_by_type(lig_positions_nm, rec_positions_nm, lig_params, rec_params, grid_type):
    """Calculate total reference energy for a specific grid type using pairwise interactions."""
    lig_pos = np.asarray(lig_positions_nm)
    rec_pos = np.asarray(rec_positions_nm)

    lig_q = np.array([p[0] for p in lig_params])
    lig_s = np.array([p[1] for p in lig_params])
    lig_e = np.array([p[2] for p in lig_params])

    rec_q = np.array([p[0] for p in rec_params])
    rec_s = np.array([p[1] for p in rec_params])
    rec_e = np.array([p[2] for p in rec_params])

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


# =============================================================================
# RESULT HANDLING
# =============================================================================

def save_result(output_file, result):
    """Append a single result to the CSV file."""
    df_new = pd.DataFrame([result])
    if os.path.exists(output_file):
        df_new.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(output_file, index=False)
