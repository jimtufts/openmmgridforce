#!/usr/bin/env python
"""
Test grid generation and evaluation with triquintic interpolation using TILED mode.
This is a copy of test_triquintic.py with tiled streaming enabled.
"""

import sys
import numpy as np
import gridforceplugin as gfp
from openmm.app import *
from openmm import *
from openmm.unit import *
import os
import tempfile

# Check if CUDA is available
try:
    platform = Platform.getPlatformByName('CUDA')
except Exception:
    print("CUDA platform not available, skipping test")
    sys.exit(0)

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
        print("PASS: All energies within 2% of reference (TILED MODE)")
    else:
        print(f"FAIL: Errors too large (charge={charge_err:.2f}%, lja={lja_err:.2f}%, ljr={ljr_err:.2f}%)")
        sys.exit(1)
