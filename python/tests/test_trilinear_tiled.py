#!/usr/bin/env python
"""
Test grid generation and evaluation with trilinear interpolation using TILED mode.
This is a copy of test_trilinear.py with tiled streaming enabled.
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
TILE_SIZE = 32  # Core tile size (excluding overlap)
TILE_MEMORY_MB = 512  # Memory budget for tile cache

receptor_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'receptor.prmtop'))
receptor_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'receptor.trans.inpcrd'))
ligand_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'ligand.prmtop'))
ligand_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'ligand.trans.inpcrd'))

receptor_atoms = list(range(receptor_prmtop.topology.getNumAtoms()))
# Convert positions to nanometers (not Angstroms!)
pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer), p[2].value_in_unit(nanometer))
            for p in receptor_inpcrd.positions]

print("Testing grid energy with trilinear interpolation (TILED MODE)")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Generate grids (no derivatives needed for trilinear)
    print("\nGenerating grids for trilinear interpolation...")
    for grid_type in ['charge', 'lja', 'ljr']:
        system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
        grid = gfp.GridForce()
        grid.setGridOrigin(*GRID_ORIGIN)
        grid.addGridCounts(*GRID_COUNTS)
        grid.addGridSpacing(*GRID_SPACING)
        grid.setAutoGenerateGrid(True)
        grid.setGridType(grid_type)

        # Trilinear does NOT need derivatives
        grid.setComputeDerivatives(False)

        grid.setReceptorAtoms(receptor_atoms)
        grid.setReceptorPositionsFromLists(pos_list)

        # Use NONE mode (no inv_power transformation) for all grids
        grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

        system.addForce(grid)
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CUDA')
        context = Context(system, integrator, platform)
        context.setPositions(receptor_inpcrd.positions)

        # Trigger grid generation
        state = context.getState(getEnergy=True)

        # Save grid
        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid.saveToFile(grid_file)
        mode = grid.getInvPowerMode()
        power = grid.getInvPower()
        print(f"  Saved {grid_type}.grid (invPowerMode={mode}, invPower={power})")

        del context

    # Now load and use the grids with trilinear interpolation + tiled mode
    print("\nCreating ligand system with trilinear grid forces (TILED MODE)...")
    print(f"  Tile size: {TILE_SIZE}, Memory budget: {TILE_MEMORY_MB} MB")
    system = ligand_prmtop.createSystem(nonbondedMethod=NoCutoff)

    # Move internal forces to group 31
    for f in system.getForces():
        f.setForceGroup(31)

    # Load grids and set trilinear interpolation with tiled mode
    for i, grid_type in enumerate(['charge', 'lja', 'ljr']):
        grid_file = os.path.join(tmpdir, f'{grid_type}.grid')
        grid_force = gfp.GridForce()
        grid_force.loadFromFile(grid_file)

        # Set trilinear interpolation (method 0)
        grid_force.setInterpolationMethod(0)

        # Enable tiled mode
        grid_force.setTiledMode(True, TILE_SIZE, TILE_MEMORY_MB)

        # Auto-calculate scaling factors
        grid_force.setAutoCalculateScalingFactors(True)
        grid_force.setScalingProperty(grid_type)

        # Separate force groups
        grid_force.setForceGroup(i)

        system.addForce(grid_force)

        mode = grid_force.getInvPowerMode()
        power = grid_force.getInvPower()
        has_derivs = grid_force.hasDerivatives()
        tiled = grid_force.getTiledMode()
        print(f"  Loaded {grid_type}: interpolationMethod=0 (trilinear), invPowerMode={mode}, invPower={power}, hasDerivatives={has_derivs}, tiledMode={tiled}")

    # Evaluate energies
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, platform)
    context.setPositions(ligand_inpcrd.positions)

    E_charge = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_lja = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_ljr = context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_internal = context.getState(getEnergy=True, groups={31}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    E_grid_sum = E_charge + E_lja + E_ljr

    print("\n" + "=" * 80)
    print("RESULTS WITH TRILINEAR INTERPOLATION (TILED MODE)")
    print("=" * 80)
    print(f"Total energy: {E_total:.3f} kJ/mol")
    print(f"  Charge (group 0): {E_charge:.3f} kJ/mol")
    print(f"  LJA (group 1):    {E_lja:.3f} kJ/mol")
    print(f"  LJR (group 2):    {E_ljr:.3f} kJ/mol")
    print(f"  Internal (31):    {E_internal:.3f} kJ/mol")
    print(f"  Grid sum:         {E_grid_sum:.3f} kJ/mol")

    # Calculate reference energies
    print("\nCalculating reference (pairwise)...")
    lig_pos = np.array(ligand_inpcrd.positions.value_in_unit(nanometer))
    rec_pos = np.array(receptor_inpcrd.positions.value_in_unit(nanometer))

    # Get ligand NonbondedForce parameters
    lig_nb = None
    for f in system.getForces():
        if isinstance(f, NonbondedForce):
            lig_nb = f
            break

    rec_system = receptor_prmtop.createSystem(nonbondedMethod=NoCutoff)
    rec_nb = None
    for f in rec_system.getForces():
        if isinstance(f, NonbondedForce):
            rec_nb = f
            break

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

    for i_lig, lig_p in enumerate(lig_pos):
        lig_q, lig_s, lig_e = lig_params[i_lig]
        for i_rec, rec_p in enumerate(rec_pos):
            rec_q, rec_s, rec_e = rec_params[i_rec]

            dr = lig_p - rec_p
            r = np.sqrt(np.sum(dr**2))

            # Electrostatic
            E_ref_charge += ONE_4PI_EPS0 * lig_q * rec_q / r

            # LJ (geometric combining rules)
            sig_ij = np.sqrt(lig_s * rec_s)
            eps_ij = np.sqrt(lig_e * rec_e)
            if eps_ij > 0:
                sr = sig_ij / r
                sr6 = sr**6
                sr12 = sr6 * sr6
                E_ref_lja += -4.0 * eps_ij * sr6
                E_ref_ljr += 4.0 * eps_ij * sr12

    E_ref_total = E_ref_charge + E_ref_lja + E_ref_ljr

    print(f"\nReference energies:")
    print(f"  Charge: {E_ref_charge:.3f} kJ/mol")
    print(f"  LJA:    {E_ref_lja:.3f} kJ/mol")
    print(f"  LJR:    {E_ref_ljr:.3f} kJ/mol")
    print(f"  Total:  {E_ref_total:.3f} kJ/mol")

    err_charge = E_charge - E_ref_charge
    err_lja = E_lja - E_ref_lja
    err_ljr = E_ljr - E_ref_ljr

    print(f"\nErrors:")
    print(f"  Charge: {err_charge:+.3f} kJ/mol")
    print(f"  LJA:    {err_lja:+.3f} kJ/mol")
    print(f"  LJR:    {err_ljr:+.3f} kJ/mol")

    rel_err_charge = 100 * err_charge / E_ref_charge
    rel_err_lja = 100 * err_lja / E_ref_lja
    rel_err_ljr = 100 * err_ljr / E_ref_ljr

    print(f"\nRelative errors:")
    print(f"  Charge: {rel_err_charge:+.2f}%")
    print(f"  LJA:    {rel_err_lja:+.2f}%")
    print(f"  LJR:    {rel_err_ljr:+.2f}%")

    # Check pass/fail
    # Trilinear is less accurate than tricubic/triquintic, so use 5% tolerance
    tolerance = 5.0
    all_pass = (abs(rel_err_charge) < tolerance and
                abs(rel_err_lja) < tolerance and
                abs(rel_err_ljr) < tolerance)

    print("\n" + "=" * 80)
    if all_pass:
        print(f"PASS: All energies within {tolerance}% of reference (TILED MODE)")
    else:
        print(f"FAIL: Some energies exceed {tolerance}% tolerance (TILED MODE)")
        sys.exit(1)
