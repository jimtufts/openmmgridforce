#!/usr/bin/env python
"""
Test grid generation and evaluation with triquintic interpolation using TILED mode.
This is a copy of test_triquintic.py with tiled streaming enabled.
"""

import sys
import numpy as np
import pytest
import gridforceplugin as gfp
from openmm.app import *
from openmm import *
from openmm.unit import *
import os
import tempfile


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

GRID_ORIGIN = (1.00175115, 0.5328844699999999, 0.8606374500000002)  # nm
GRID_SPACING = (0.0125, 0.0125, 0.0125)  # nm
GRID_COUNTS = (208, 278, 231)

# Tiled mode parameters
TILE_SIZE = 32
TILE_MEMORY_MB = 512

receptor_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'receptor.prmtop'))
receptor_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'receptor.trans.inpcrd'))
ligand_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'ligand.prmtop'))
ligand_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'ligand.trans.inpcrd'))

receptor_atoms = list(range(receptor_prmtop.topology.getNumAtoms()))
pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer), p[2].value_in_unit(nanometer))
            for p in receptor_inpcrd.positions]

print("Testing grid energy with triquintic interpolation (TILED MODE)")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Generate grids with triquintic derivatives
    print("\nGenerating grids for triquintic interpolation...")
    for grid_type in ['charge', 'lja', 'ljr']:
        system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
        grid = gfp.GridForce()
        grid.setGridOrigin(*GRID_ORIGIN)
        grid.addGridCounts(*GRID_COUNTS)
        grid.addGridSpacing(*GRID_SPACING)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(grid_type)
        grid.setComputeDerivatives(True)
        grid.setGridCap(1e30)  # High grid cap for triquintic
        grid.setReceptorAtoms(receptor_atoms)
        grid.setReceptorPositionsFromLists(pos_list)
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        system.addForce(grid)
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CUDA')
        context = Context(system, integrator, platform)
        context.setPositions(receptor_inpcrd.positions)
        state = context.getState(getEnergy=True)

        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid.saveToFile(grid_file)
        print(f"  Saved {grid_type}.grid")
        del context

    # Now load and use the grids with triquintic interpolation + tiled mode
    print("\nCreating ligand system with triquintic grid forces (TILED MODE)...")
    print(f"  Tile size: {TILE_SIZE}, Memory budget: {TILE_MEMORY_MB} MB")
    system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

    for f in system.getForces():
        f.setForceGroup(31)

    for i, grid_type in enumerate(['charge', 'lja', 'ljr']):
        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid_force = gfp.GridForce()
        grid_force.loadFromFile(grid_file)

        grid_force.setInterpolationMethod(3)  # Triquintic
        grid_force.setTiledMode(True, TILE_SIZE, TILE_MEMORY_MB)
        grid_force.setScalingProperty(grid_type)
        grid_force.setAutoCalculateScalingFactors(True)
        grid_force.setForceGroup(i)
        system.addForce(grid_force)
        print(f"  Loaded {grid_type}: interpolationMethod=3 (triquintic), tiledMode={grid_force.getTiledMode()}")

    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)
    context.setPositions(ligand_inpcrd.positions)

    state_total = context.getState(getEnergy=True)
    E_total = state_total.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    E_charge = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_lja = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr = context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_internal = context.getState(getEnergy=True, groups={31}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    print("\n" + "=" * 80)
    print("RESULTS WITH TRIQUINTIC INTERPOLATION (TILED MODE)")
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
    print(f"  Charge: {100*(E_charge - E_ref_charge)/E_ref_charge:+.2f}%")
    print(f"  LJA:    {100*(E_lja - E_ref_lja)/E_ref_lja:+.2f}%")
    print(f"  LJR:    {100*(E_ljr - E_ref_ljr)/E_ref_ljr:+.2f}%")

    print("\n" + "=" * 80)
    charge_err = abs(100*(E_charge - E_ref_charge)/E_ref_charge)
    lja_err = abs(100*(E_lja - E_ref_lja)/E_ref_lja)
    ljr_err = abs(100*(E_ljr - E_ref_ljr)/E_ref_ljr)

    if charge_err < 2.0 and lja_err < 2.0 and ljr_err < 2.0:
        print("PASS: All energies within 2% of reference (TILED MODE - memory-backed)")
    else:
        print(f"FAIL: Errors too large (charge={charge_err:.2f}%, lja={lja_err:.2f}%, ljr={ljr_err:.2f}%)")
        sys.exit(1)

    # Store reference energies for comparison with file-backed mode
    E_memory_charge = E_charge
    E_memory_lja = E_lja
    E_memory_ljr = E_ljr
    del context

# =============================================================================
# TEST 2: Tiled FILE FORMAT (file-backed mode)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: TILED FILE FORMAT (file-backed mode)")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Generate grids directly to .tiled format
    print("\nGenerating grids directly to .tiled format...")
    for grid_type in ['charge', 'lja', 'ljr']:
        system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
        grid = gfp.GridForce()
        grid.setGridOrigin(*GRID_ORIGIN)
        grid.addGridCounts(*GRID_COUNTS)
        grid.addGridSpacing(*GRID_SPACING)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(grid_type)
        grid.setComputeDerivatives(True)
        grid.setGridCap(1e30)
        grid.setReceptorAtoms(receptor_atoms)
        grid.setReceptorPositionsFromLists(pos_list)
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        # Generate directly to tiled file format
        tiled_file = os.path.join(tmpdir, f'{grid_type}.tiled')
        grid.setTiledOutputFile(tiled_file, TILE_SIZE)

        system.addForce(grid)
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CUDA')
        context = Context(system, integrator, platform)
        context.setPositions(receptor_inpcrd.positions)
        context.getState(getEnergy=True)  # Trigger generation
        print(f"  Generated {grid_type}.tiled ({os.path.getsize(tiled_file)} bytes)")
        del context

    # Load tiled files and evaluate with triquintic
    print("\nLoading tiled files with triquintic interpolation...")
    print(f"  Tile size: {TILE_SIZE}, Memory budget: {TILE_MEMORY_MB} MB")
    system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

    for f in system.getForces():
        f.setForceGroup(31)

    for i, grid_type in enumerate(['charge', 'lja', 'ljr']):
        tiled_file = os.path.join(tmpdir, f'{grid_type}.tiled')
        grid_force = gfp.GridForce()
        # Must set grid parameters for tiled input files
        grid_force.setGridOrigin(*GRID_ORIGIN)
        grid_force.addGridCounts(*GRID_COUNTS)
        grid_force.addGridSpacing(*GRID_SPACING)
        grid_force.setTiledInputFile(tiled_file)
        grid_force.setTiledMode(True, TILE_SIZE, TILE_MEMORY_MB)
        grid_force.setInterpolationMethod(3)  # Triquintic
        grid_force.setScalingProperty(grid_type)
        grid_force.setAutoCalculateScalingFactors(True)
        grid_force.setForceGroup(i)
        system.addForce(grid_force)
        print(f"  Loaded {grid_type}.tiled: interpolationMethod=3 (triquintic)")

    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)
    context.setPositions(ligand_inpcrd.positions)

    E_charge_file = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_lja_file = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr_file = context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    print("\n" + "=" * 80)
    print("RESULTS WITH TRIQUINTIC (TILED FILE FORMAT)")
    print("=" * 80)
    print(f"  Charge: {E_charge_file:.3f} kJ/mol")
    print(f"  LJA:    {E_lja_file:.3f} kJ/mol")
    print(f"  LJR:    {E_ljr_file:.3f} kJ/mol")

    print("\nComparison with memory-backed mode:")
    print(f"  Charge diff: {E_charge_file - E_memory_charge:+.6f} kJ/mol")
    print(f"  LJA diff:    {E_lja_file - E_memory_lja:+.6f} kJ/mol")
    print(f"  LJR diff:    {E_ljr_file - E_memory_ljr:+.6f} kJ/mol")

    print("\nComparison with pairwise reference:")
    charge_err_file = abs(100*(E_charge_file - E_ref_charge)/E_ref_charge)
    lja_err_file = abs(100*(E_lja_file - E_ref_lja)/E_ref_lja)
    ljr_err_file = abs(100*(E_ljr_file - E_ref_ljr)/E_ref_ljr)
    print(f"  Charge: {100*(E_charge_file - E_ref_charge)/E_ref_charge:+.2f}%")
    print(f"  LJA:    {100*(E_lja_file - E_ref_lja)/E_ref_lja:+.2f}%")
    print(f"  LJR:    {100*(E_ljr_file - E_ref_ljr)/E_ref_ljr:+.2f}%")

    print("\n" + "=" * 80)
    if charge_err_file < 2.0 and lja_err_file < 2.0 and ljr_err_file < 2.0:
        print("PASS: All energies within 2% of reference (TILED FILE FORMAT)")
    else:
        print(f"FAIL: Errors too large (charge={charge_err_file:.2f}%, lja={lja_err_file:.2f}%, ljr={ljr_err_file:.2f}%)")
        sys.exit(1)

# =============================================================================
# TEST 3: inv_power RUNTIME mode
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: INV_POWER RUNTIME MODE (trilinear)")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Generate grids WITHOUT inv_power transformation (raw values)
    print("\nGenerating raw grids (no inv_power)...")
    for grid_type in ['lja', 'ljr']:
        system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
        grid = gfp.GridForce()
        grid.setGridOrigin(*GRID_ORIGIN)
        grid.addGridCounts(*GRID_COUNTS)
        grid.addGridSpacing(*GRID_SPACING)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(grid_type)
        grid.setComputeDerivatives(False)  # Not needed for trilinear
        grid.setGridCap(1e30)
        grid.setReceptorAtoms(receptor_atoms)
        grid.setReceptorPositionsFromLists(pos_list)
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        system.addForce(grid)
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CUDA')
        context = Context(system, integrator, platform)
        context.setPositions(receptor_inpcrd.positions)
        state = context.getState(getEnergy=True)

        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid.saveToFile(grid_file)
        print(f"  Saved {grid_type}.grid")
        del context

    # Evaluate with inv_power RUNTIME mode (trilinear interpolation)
    INV_POWER = -12.0
    print(f"\nEvaluating with inv_power={INV_POWER} RUNTIME mode (trilinear)...")
    system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

    for f in system.getForces():
        f.setForceGroup(31)

    for i, grid_type in enumerate(['lja', 'ljr']):
        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid_force = gfp.GridForce()
        grid_force.loadFromFile(grid_file)

        grid_force.setInterpolationMethod(0)  # Trilinear
        grid_force.setInvPowerMode(gfp.InvPowerMode_RUNTIME, INV_POWER)
        grid_force.setScalingProperty(grid_type)
        grid_force.setAutoCalculateScalingFactors(True)
        grid_force.setForceGroup(i)
        system.addForce(grid_force)
        print(f"  Loaded {grid_type}: interpolationMethod=0 (trilinear), invPowerMode=RUNTIME, invPower={INV_POWER}")

    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)
    context.setPositions(ligand_inpcrd.positions)

    E_lja_runtime = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr_runtime = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    print("\n" + "=" * 80)
    print(f"RESULTS WITH INV_POWER={INV_POWER} RUNTIME MODE (trilinear)")
    print("=" * 80)
    print(f"  LJA (RUNTIME):  {E_lja_runtime:.3f} kJ/mol")
    print(f"  LJR (RUNTIME):  {E_ljr_runtime:.3f} kJ/mol")

    print("\nComparison with pairwise reference:")
    print(f"  LJA ref:        {E_ref_lja:.3f} kJ/mol")
    print(f"  LJR ref:        {E_ref_ljr:.3f} kJ/mol")
    print(f"  LJA diff:       {E_lja_runtime - E_ref_lja:+.3f} kJ/mol ({100*(E_lja_runtime - E_ref_lja)/E_ref_lja:+.2f}%)")
    print(f"  LJR diff:       {E_ljr_runtime - E_ref_ljr:+.3f} kJ/mol ({100*(E_ljr_runtime - E_ref_ljr)/E_ref_ljr:+.2f}%)")

    del context

# =============================================================================
# TEST 4: inv_power RUNTIME mode with TRIQUINTIC (requires derivatives)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: INV_POWER RUNTIME MODE (triquintic)")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Generate grids WITH derivatives for triquintic (but no inv_power transform)
    print("\nGenerating grids with derivatives (no inv_power)...")
    for grid_type in ['lja', 'ljr']:
        system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
        grid = gfp.GridForce()
        grid.setGridOrigin(*GRID_ORIGIN)
        grid.addGridCounts(*GRID_COUNTS)
        grid.addGridSpacing(*GRID_SPACING)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(grid_type)
        grid.setComputeDerivatives(True)  # Needed for triquintic
        grid.setGridCap(1e30)
        grid.setReceptorAtoms(receptor_atoms)
        grid.setReceptorPositionsFromLists(pos_list)
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        system.addForce(grid)
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CUDA')
        context = Context(system, integrator, platform)
        context.setPositions(receptor_inpcrd.positions)
        state = context.getState(getEnergy=True)

        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid.saveToFile(grid_file)
        print(f"  Saved {grid_type}.grid")
        del context

    # Evaluate with inv_power RUNTIME mode (triquintic interpolation)
    INV_POWER = -12.0
    print(f"\nEvaluating with inv_power={INV_POWER} RUNTIME mode (triquintic)...")
    system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

    for f in system.getForces():
        f.setForceGroup(31)

    for i, grid_type in enumerate(['lja', 'ljr']):
        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid_force = gfp.GridForce()
        grid_force.loadFromFile(grid_file)

        grid_force.setInterpolationMethod(3)  # Triquintic
        grid_force.setInvPowerMode(gfp.InvPowerMode_RUNTIME, INV_POWER)
        grid_force.setScalingProperty(grid_type)
        grid_force.setAutoCalculateScalingFactors(True)
        grid_force.setForceGroup(i)
        system.addForce(grid_force)
        print(f"  Loaded {grid_type}: interpolationMethod=3 (triquintic), invPowerMode=RUNTIME, invPower={INV_POWER}")

    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)
    context.setPositions(ligand_inpcrd.positions)

    E_lja_triquintic = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr_triquintic = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    print("\n" + "=" * 80)
    print(f"RESULTS WITH INV_POWER={INV_POWER} RUNTIME MODE (triquintic)")
    print("=" * 80)
    print(f"  LJA (RUNTIME):  {E_lja_triquintic:.3f} kJ/mol")
    print(f"  LJR (RUNTIME):  {E_ljr_triquintic:.3f} kJ/mol")

    print("\nComparison with pairwise reference:")
    print(f"  LJA ref:        {E_ref_lja:.3f} kJ/mol")
    print(f"  LJR ref:        {E_ref_ljr:.3f} kJ/mol")
    print(f"  LJA diff:       {E_lja_triquintic - E_ref_lja:+.3f} kJ/mol ({100*(E_lja_triquintic - E_ref_lja)/E_ref_lja:+.2f}%)")
    print(f"  LJR diff:       {E_ljr_triquintic - E_ref_ljr:+.3f} kJ/mol ({100*(E_ljr_triquintic - E_ref_ljr)/E_ref_ljr:+.2f}%)")

    del context

# =============================================================================
# TEST 5: inv_power via updateParametersInContext (like benchmark does)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: INV_POWER via updateParametersInContext")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Generate grids WITHOUT inv_power
    print("\nGenerating grids (no inv_power)...")
    for grid_type in ['lja', 'ljr']:
        system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
        grid = gfp.GridForce()
        grid.setGridOrigin(*GRID_ORIGIN)
        grid.addGridCounts(*GRID_COUNTS)
        grid.addGridSpacing(*GRID_SPACING)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(grid_type)
        grid.setComputeDerivatives(False)
        grid.setGridCap(1e30)
        grid.setReceptorAtoms(receptor_atoms)
        grid.setReceptorPositionsFromLists(pos_list)
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        system.addForce(grid)
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CUDA')
        context = Context(system, integrator, platform)
        context.setPositions(receptor_inpcrd.positions)
        context.getState(getEnergy=True)

        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid.saveToFile(grid_file)
        print(f"  Saved {grid_type}.grid")
        del context

    # Create context with NONE mode first, then switch to RUNTIME
    print("\nCreating context with invPowerMode=NONE initially...")
    system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

    for f in system.getForces():
        f.setForceGroup(31)

    grid_forces = []
    for i, grid_type in enumerate(['lja', 'ljr']):
        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid_force = gfp.GridForce()
        grid_force.loadFromFile(grid_file)

        grid_force.setInterpolationMethod(0)  # Trilinear
        grid_force.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)  # Start with NONE
        grid_force.setScalingProperty(grid_type)
        grid_force.setAutoCalculateScalingFactors(True)
        grid_force.setForceGroup(i)
        system.addForce(grid_force)
        grid_forces.append(grid_force)
        print(f"  Added {grid_type}: invPowerMode=NONE initially")

    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName('CUDA')
    context = Context(system, integrator, platform)
    context.setPositions(ligand_inpcrd.positions)

    # First evaluation with NONE mode
    E_lja_none = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr_none = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    print(f"\nWith invPowerMode=NONE:")
    print(f"  LJA: {E_lja_none:.3f} kJ/mol")
    print(f"  LJR: {E_ljr_none:.3f} kJ/mol")

    # Now update to RUNTIME mode via updateParametersInContext
    INV_POWER = -12.0
    print(f"\nUpdating to invPowerMode=RUNTIME, invPower={INV_POWER} via updateParametersInContext...")
    for i, (grid_force, grid_type) in enumerate(zip(grid_forces, ['lja', 'ljr'])):
        grid_force.setInvPowerMode(gfp.InvPowerMode_RUNTIME, INV_POWER)
        grid_force.updateParametersInContext(context)
        print(f"  Updated {grid_type}")

    # Second evaluation with RUNTIME mode
    E_lja_runtime_ctx = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr_runtime_ctx = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

    print(f"\nWith invPowerMode=RUNTIME (after updateParametersInContext):")
    print(f"  LJA: {E_lja_runtime_ctx:.3f} kJ/mol")
    print(f"  LJR: {E_ljr_runtime_ctx:.3f} kJ/mol")

    print("\nComparison with TEST 3 (direct RUNTIME mode):")
    print(f"  LJA diff from direct RUNTIME: {E_lja_runtime_ctx - E_lja_runtime:+.6f} kJ/mol")
    print(f"  LJR diff from direct RUNTIME: {E_ljr_runtime_ctx - E_ljr_runtime:+.6f} kJ/mol")

    print("\nComparison with reference:")
    print(f"  LJA ref:   {E_ref_lja:.3f} kJ/mol")
    print(f"  LJR ref:   {E_ref_ljr:.3f} kJ/mol")
    print(f"  LJA error: {100*(E_lja_runtime_ctx - E_ref_lja)/E_ref_lja:+.2f}%")
    print(f"  LJR error: {100*(E_ljr_runtime_ctx - E_ref_ljr)/E_ref_ljr:+.2f}%")

    del context

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
