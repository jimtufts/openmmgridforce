#!/usr/bin/env python
"""
Test grid generation and evaluation with triquintic interpolation using TILED mode.
BIGGER GRID version to stress-test derivative caching in tiled streaming.

Instead of increasing resolution (which would be very expensive for triquintic
due to all the derivative computations), we make the grid larger by extending
the region covered.

Uses tile-by-tile generation to test the tiled file I/O path.
"""

import sys
import gc
import numpy as np
import pytest
import gridforceplugin as gfp
from openmm.app import *
from openmm import *
from openmm.unit import *
import os
import tempfile
import time


def cuda_available():
    """Check if CUDA platform is available."""
    try:
        Platform.getPlatformByName('CUDA')
        return True
    except Exception:
        return False


# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA platform not available")

# Get CUDA platform (will only run if CUDA is available)
platform = Platform.getPlatformByName('CUDA') if cuda_available() else None

# Coulomb constant
ONE_4PI_EPS0 = 138.935456  # kJ/mol * nm / e^2

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(TEST_DIR), 'prmtopcrd')

# Grid parameters - BIGGER GRID (same resolution, larger region)
# Original: 208 x 278 x 231 at 0.0125 nm spacing = 2.6 x 3.475 x 2.8875 nm
# Bigger: 2x in each dimension = 416 x 556 x 462 = ~106 million points
GRID_ORIGIN = (0.0, -0.5, -0.5)  # nm - extend origin to cover more space
GRID_SPACING = (0.0125, 0.0125, 0.0125)  # nm (same as original)
GRID_COUNTS = (416, 556, 462)  # 2x original counts

# Tiled mode parameters
TILE_SIZE = 32
TILE_MEMORY_MB = 2048  # 2 GB tile cache

# Calculate expected memory
n_points = GRID_COUNTS[0] * GRID_COUNTS[1] * GRID_COUNTS[2]
mem_values_gb = n_points * 4 / (1024**3)  # float32 = 4 bytes
# Triquintic has 56 derivatives per point (vs 27 for B-spline)
mem_with_derivs_gb = n_points * 56 * 4 / (1024**3)

print("=" * 80)
print("BIGGER GRID TRIQUINTIC TILED TEST")
print("=" * 80)
print(f"Grid spacing: {GRID_SPACING[0]*10:.2f} Angstrom ({GRID_SPACING[0]} nm)")
print(f"Grid counts: {GRID_COUNTS[0]} x {GRID_COUNTS[1]} x {GRID_COUNTS[2]}")
print(f"Total grid points: {n_points:,}")
print(f"Estimated memory (values only): {mem_values_gb:.2f} GB")
print(f"Estimated memory (with derivs): {mem_with_derivs_gb:.2f} GB")
print(f"Tile size: {TILE_SIZE}, Tile cache: {TILE_MEMORY_MB} MB")
print("=" * 80)

receptor_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'receptor.prmtop'))
receptor_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'receptor.trans.inpcrd'))
ligand_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'ligand.prmtop'))
ligand_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'ligand.trans.inpcrd'))

receptor_atoms = list(range(receptor_prmtop.topology.getNumAtoms()))
pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer), p[2].value_in_unit(nanometer))
            for p in receptor_inpcrd.positions]

# Use a persistent directory for the grids
GRID_DIR = '/scratch/triquintic_biggrid_grids'
os.makedirs(GRID_DIR, exist_ok=True)

print("\nPhase 1: Grid Generation (TILED - tile-by-tile to disk)")
print("-" * 40)

for grid_type in ['charge', 'lja', 'ljr']:
    # Use .tiled extension for tiled format files
    grid_file = os.path.join(GRID_DIR, f'{grid_type}_triquintic_big.tiled')

    if os.path.exists(grid_file):
        print(f"  {grid_type}: Already exists, skipping generation")
        continue

    print(f"  Generating {grid_type} grid (tile-by-tile)...", flush=True)
    start_time = time.time()

    system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
    grid = gfp.GridForce()
    grid.setGridOrigin(*GRID_ORIGIN)
    grid.addGridCounts(*GRID_COUNTS)
    grid.addGridSpacing(*GRID_SPACING)
    grid.setAutoGenerateGrid(True)
    grid.setGridType(grid_type)
    grid.setComputeDerivatives(True)  # Needed for triquintic
    grid.setGridCap(1e30)  # High grid cap for triquintic
    grid.setReceptorAtoms(receptor_atoms)
    grid.setReceptorPositionsFromLists(pos_list)
    grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

    # Enable tiled output - generates tile-by-tile directly to file
    grid.setTiledOutputFile(grid_file, TILE_SIZE)

    system.addForce(grid)
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, platform)
    context.setPositions(receptor_inpcrd.positions)

    # Note: For tiled output mode, context creation generates the grid
    elapsed = time.time() - start_time
    if os.path.exists(grid_file):
        file_size_gb = os.path.getsize(grid_file) / (1024**3)
        print(f"    Done in {elapsed:.1f}s, file size: {file_size_gb:.2f} GB")
    else:
        print(f"    ERROR: Grid file was not created!")
        sys.exit(1)

    del context, integrator, system, grid
    gfp.clearGridCache()
    gc.collect()  # Force memory cleanup

print("\nPhase 2: Grid Evaluation with Tiled Streaming (Triquintic)")
print("-" * 40)

system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

for f in system.getForces():
    f.setForceGroup(31)

for i, grid_type in enumerate(['charge', 'lja', 'ljr']):
    grid_file = os.path.join(GRID_DIR, f'{grid_type}_triquintic_big.tiled')

    if not os.path.exists(grid_file):
        print(f"ERROR: Grid file not found: {grid_file}")
        sys.exit(1)

    print(f"  Loading {grid_type} grid with tiled input mode...", end=" ", flush=True)
    start_time = time.time()

    grid_force = gfp.GridForce()
    # Set grid dimensions (needed for context creation)
    grid_force.setGridOrigin(*GRID_ORIGIN)
    grid_force.addGridCounts(*GRID_COUNTS)
    grid_force.addGridSpacing(*GRID_SPACING)

    # Use tiled input file - loads tiles on demand
    grid_force.setTiledInputFile(grid_file)
    grid_force.setTiledMode(True, TILE_SIZE, TILE_MEMORY_MB)

    grid_force.setInterpolationMethod(3)  # Triquintic
    grid_force.setScalingProperty(grid_type)
    grid_force.setAutoCalculateScalingFactors(True)
    grid_force.setForceGroup(i)
    system.addForce(grid_force)

    elapsed = time.time() - start_time
    print(f"done in {elapsed:.1f}s")

print("\n  Creating context and evaluating energy...")
start_time = time.time()
integrator = VerletIntegrator(0.001)
context = Context(system, integrator, platform)
context.setPositions(ligand_inpcrd.positions)
context_time = time.time() - start_time
print(f"  Context creation: {context_time:.1f}s")

start_time = time.time()
state_total = context.getState(getEnergy=True)
E_total = state_total.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
eval_time = time.time() - start_time
print(f"  Energy evaluation: {eval_time:.3f}s")

E_charge = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
E_lja = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
E_ljr = context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
E_internal = context.getState(getEnergy=True, groups={31}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

print("\n" + "=" * 80)
print("RESULTS WITH TRIQUINTIC INTERPOLATION (TILED MODE, BIGGER GRID)")
print("=" * 80)
print(f"Total energy: {E_total:.3f} kJ/mol")
print(f"  Charge (group 0): {E_charge:.3f} kJ/mol")
print(f"  LJA (group 1):    {E_lja:.3f} kJ/mol")
print(f"  LJR (group 2):    {E_ljr:.3f} kJ/mol")
print(f"  Internal (31):    {E_internal:.3f} kJ/mol")
print(f"  Grid sum:         {E_charge + E_lja + E_ljr:.3f} kJ/mol")

# Calculate reference
print("\nCalculating reference (pairwise)...")
lig_positions = np.array(ligand_inpcrd.positions.value_in_unit(nanometer))
rec_positions = np.array(receptor_inpcrd.positions.value_in_unit(nanometer))

lig_sys = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)
rec_sys = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)

lig_nb = [f for f in lig_sys.getForces() if isinstance(f, NonbondedForce)][0]
rec_nb = [f for f in rec_sys.getForces() if isinstance(f, NonbondedForce)][0]

lig_params = []
for i in range(ligand_prmtop.topology.getNumAtoms()):
    q, s, e = lig_nb.getParticleParameters(i)
    lig_params.append((q.value_in_unit(elementary_charge),
                      s.value_in_unit(nanometer),
                      e.value_in_unit(kilojoules_per_mole)))

rec_params = []
for i in range(receptor_prmtop.topology.getNumAtoms()):
    q, s, e = rec_nb.getParticleParameters(i)
    rec_params.append((q.value_in_unit(elementary_charge),
                      s.value_in_unit(nanometer),
                      e.value_in_unit(kilojoules_per_mole)))

E_ref_charge = 0.0
E_ref_lja = 0.0
E_ref_ljr = 0.0

for i_lig, lig_pos in enumerate(lig_positions):
    lig_q, lig_s, lig_e = lig_params[i_lig]
    for i_rec, rec_pos in enumerate(rec_positions):
        rec_q, rec_s, rec_e = rec_params[i_rec]
        dx = lig_pos[0] - rec_pos[0]
        dy = lig_pos[1] - rec_pos[1]
        dz = lig_pos[2] - rec_pos[2]
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        E_ref_charge += ONE_4PI_EPS0 * lig_q * rec_q / r
        sig_ij = np.sqrt(lig_s * rec_s)
        eps_ij = np.sqrt(lig_e * rec_e)
        if eps_ij > 0:
            sr = sig_ij / r
            sr6 = sr**6
            sr12 = sr6 * sr6
            E_ref_lja += -4.0 * eps_ij * sr6
            E_ref_ljr += 4.0 * eps_ij * sr12

print("\nReference energies:")
print(f"  Charge: {E_ref_charge:.3f} kJ/mol")
print(f"  LJA:    {E_ref_lja:.3f} kJ/mol")
print(f"  LJR:    {E_ref_ljr:.3f} kJ/mol")
print(f"  Total:  {E_ref_charge + E_ref_lja + E_ref_ljr:.3f} kJ/mol")

print("\nErrors:")
print(f"  Charge: {E_charge - E_ref_charge:+.3f} kJ/mol")
print(f"  LJA:    {E_lja - E_ref_lja:+.3f} kJ/mol")
print(f"  LJR:    {E_ljr - E_ref_ljr:+.3f} kJ/mol")

print("\nRelative errors:")
rel_err_charge = 100*(E_charge - E_ref_charge)/E_ref_charge
rel_err_lja = 100*(E_lja - E_ref_lja)/E_ref_lja
rel_err_ljr = 100*(E_ljr - E_ref_ljr)/E_ref_ljr
print(f"  Charge: {rel_err_charge:+.4f}%")
print(f"  LJA:    {rel_err_lja:+.4f}%")
print(f"  LJR:    {rel_err_ljr:+.4f}%")

print("\n" + "=" * 80)
if abs(rel_err_charge) < 2.0 and abs(rel_err_lja) < 2.0 and abs(rel_err_ljr) < 2.0:
    print("PASS: All energies within 2% of reference (TILED MODE, BIGGER GRID)")
else:
    print(f"FAIL: Errors exceed 2% tolerance")
    print(f"      (charge={rel_err_charge:.4f}%, lja={rel_err_lja:.4f}%, ljr={rel_err_ljr:.4f}%)")
    sys.exit(1)
