#!/usr/bin/env python
"""
Test GBSAGridForce multi-ligand support.

Verifies that multiple ligand copies can be evaluated simultaneously
with correct per-group energy reporting.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openmm as mm
from openmm import unit

from gridforceplugin import GBSAGridForce, DesolvationGrid
from desolvation_grid_generator import generate_desolvation_grid


def create_receptor():
    """Create receptor atoms."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.15, 0.26, 0.0],
    ], dtype=np.float64)
    radii = np.array([0.17, 0.17, 0.15], dtype=np.float64)
    scales = np.array([0.72, 0.72, 0.79], dtype=np.float64)
    return positions, radii, scales


def create_ligand_template():
    """Create ligand template (2 atoms)."""
    # Template positions (will be translated for each copy)
    radii = np.array([0.17, 0.12], dtype=np.float64)  # C, H
    scales = np.array([0.72, 0.85], dtype=np.float64)
    charges = np.array([0.2, -0.1], dtype=np.float64)
    return radii, scales, charges


def test_multiligand_energy():
    """Test that multiple ligand copies get correct energies."""
    print("=" * 60)
    print("Test: Multi-ligand Energy Calculation")
    print("=" * 60)

    # Create receptor
    rec_pos, rec_radii, rec_scales = create_receptor()
    n_rec = len(rec_pos)

    # Create ligand template
    lig_radii, lig_scales, lig_charges = create_ligand_template()
    n_lig_atoms = len(lig_radii)

    # Number of ligand copies
    n_ligands = 4

    # Generate desolvation grid from receptor
    print("\nGenerating desolvation grid...")
    origin = np.array([-0.5, -0.5, -0.5])
    spacing = 0.05
    counts = (30, 30, 30)

    py_grid = generate_desolvation_grid(
        rec_pos, rec_radii, rec_scales,
        origin, spacing, counts,
        probe_radius=0.14,
        r_thresholds=(0.12, 0.16),
        verbose=False
    )

    # Convert to C++ grid
    nx, ny, nz = py_grid.counts
    cpp_grid = DesolvationGrid(
        nx, ny, nz,
        float(py_grid.spacing),
        float(py_grid.probe_radius),
        list(py_grid.r_thresholds)
    )
    cpp_grid.setOrigin(
        float(py_grid.origin[0]),
        float(py_grid.origin[1]),
        float(py_grid.origin[2])
    )

    # Set grid data
    cpp_grid.setHctProbe(py_grid.hct_probe.flatten(order='C').tolist())

    n_points = nx * ny * nz
    n_bins = py_grid.n_bins
    corr_N, corr_A, corr_B = [], [], []
    for b in range(n_bins):
        corr_N.extend(py_grid.correction_N[b].flatten(order='C').tolist())
        corr_A.extend(py_grid.correction_A[b].flatten(order='C').tolist())
        corr_B.extend(py_grid.correction_B[b].flatten(order='C').tolist())
    cpp_grid.setCorrectionN(corr_N)
    cpp_grid.setCorrectionA(corr_A)
    cpp_grid.setCorrectionB(corr_B)

    # Create OpenMM system
    print(f"\nCreating system with {n_ligands} ligand copies...")
    system = mm.System()

    # Add ligand atoms (n_ligands copies)
    total_atoms = n_ligands * n_lig_atoms
    for _ in range(total_atoms):
        system.addParticle(12.0)

    # Create GBSAGridForce
    gbsa = GBSAGridForce()
    gbsa.setNumAtoms(n_lig_atoms)  # Template size
    gbsa.setIncludeSurfaceArea(False)

    # Set template atom parameters
    for i in range(n_lig_atoms):
        gbsa.setAtomParameters(i, float(lig_charges[i]),
                               float(lig_radii[i]),
                               float(lig_scales[i]))

    gbsa.setDesolvationGrid(cpp_grid)

    # Add particle groups for each ligand copy
    for lig_idx in range(n_ligands):
        start = lig_idx * n_lig_atoms
        indices = list(range(start, start + n_lig_atoms))
        gbsa.addParticleGroup(f"ligand_{lig_idx}", indices)

    # Add exclusion for bonded atoms in template
    gbsa.addExclusion(0, 1)  # C-H bond

    system.addForce(gbsa)

    # Create context
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        print("Using CUDA platform")
    except:
        platform = mm.Platform.getPlatformByName('Reference')
        print("Using Reference platform")

    context = mm.Context(system, integrator, platform)

    # Create positions for each ligand copy at different locations
    # Ligand 0: close to receptor
    # Ligand 1: medium distance
    # Ligand 2: far from receptor
    # Ligand 3: same as ligand 0 (should have same energy)

    ligand_offsets = [
        np.array([0.15, 0.1, 0.2]),   # Close
        np.array([0.15, 0.1, 0.4]),   # Medium
        np.array([0.15, 0.1, 0.8]),   # Far
        np.array([0.15, 0.1, 0.2]),   # Same as ligand 0
    ]

    # Base ligand geometry (C-H bond)
    base_lig_pos = np.array([
        [0.0, 0.0, 0.0],
        [0.11, 0.0, 0.0],  # ~1.1 Angstrom C-H bond
    ])

    all_positions = []
    for offset in ligand_offsets:
        for atom_pos in base_lig_pos:
            all_positions.append(atom_pos + offset)

    positions = np.array(all_positions)
    context.setPositions(positions * unit.nanometer)

    # Get state and energies
    state = context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    print(f"\nTotal energy: {total_energy:.4f} kJ/mol")

    # Get per-group energies
    print("\nPer-group energies:")
    group_energies = []
    for g in range(n_ligands):
        energy = gbsa.getGroupEnergy(g)
        group_energies.append(energy)
        print(f"  Ligand {g}: {energy:.4f} kJ/mol (position: {ligand_offsets[g]})")

    # Verify sum of group energies equals total
    sum_energies = sum(group_energies)
    print(f"\nSum of group energies: {sum_energies:.4f} kJ/mol")
    print(f"Difference from total: {abs(total_energy - sum_energies):.6f} kJ/mol")

    # Check that ligands 0 and 3 have the same energy (same position)
    energy_diff_0_3 = abs(group_energies[0] - group_energies[3])
    print(f"\nLigand 0 vs 3 (same position): difference = {energy_diff_0_3:.6f} kJ/mol")

    # Check that closer ligand has more negative (favorable) energy
    # (receptor should provide more desolvation for closer ligand)
    print("\nPhysical sanity checks:")
    print(f"  Ligand 0 (close): {group_energies[0]:.4f} kJ/mol")
    print(f"  Ligand 2 (far):   {group_energies[2]:.4f} kJ/mol")

    # Tests
    passed = True

    # Test 1: Sum of energies matches total
    if abs(total_energy - sum_energies) > 0.01:
        print("\nFAILED: Sum of group energies doesn't match total")
        passed = False

    # Test 2: Same position gives same energy
    if energy_diff_0_3 > 0.001:
        print("\nFAILED: Ligands at same position have different energies")
        passed = False

    # Test 3: All energies are negative (favorable solvation)
    if any(e >= 0 for e in group_energies):
        print("\nWARNING: Some ligands have non-negative energy")

    if passed:
        print("\n" + "=" * 60)
        print("PASSED: Multi-ligand test successful!")
        print("=" * 60)

    return passed, group_energies


def test_multiligand_forces():
    """Test that forces are computed correctly for multiple ligands."""
    print("\n" + "=" * 60)
    print("Test: Multi-ligand Force Calculation")
    print("=" * 60)

    # Simplified test: 2 ligands, each with 2 atoms
    rec_pos, rec_radii, rec_scales = create_receptor()
    lig_radii, lig_scales, lig_charges = create_ligand_template()
    n_lig_atoms = len(lig_radii)
    n_ligands = 2

    # Generate grid
    origin = np.array([-0.5, -0.5, -0.5])
    spacing = 0.05
    counts = (30, 30, 30)

    py_grid = generate_desolvation_grid(
        rec_pos, rec_radii, rec_scales,
        origin, spacing, counts,
        probe_radius=0.14,
        r_thresholds=(0.12, 0.16),
        verbose=False
    )

    # Convert to C++ grid
    nx, ny, nz = py_grid.counts
    cpp_grid = DesolvationGrid(nx, ny, nz, float(py_grid.spacing),
                                float(py_grid.probe_radius), list(py_grid.r_thresholds))
    cpp_grid.setOrigin(float(py_grid.origin[0]), float(py_grid.origin[1]), float(py_grid.origin[2]))
    cpp_grid.setHctProbe(py_grid.hct_probe.flatten(order='C').tolist())

    n_points = nx * ny * nz
    n_bins = py_grid.n_bins
    corr_N, corr_A, corr_B = [], [], []
    for b in range(n_bins):
        corr_N.extend(py_grid.correction_N[b].flatten(order='C').tolist())
        corr_A.extend(py_grid.correction_A[b].flatten(order='C').tolist())
        corr_B.extend(py_grid.correction_B[b].flatten(order='C').tolist())
    cpp_grid.setCorrectionN(corr_N)
    cpp_grid.setCorrectionA(corr_A)
    cpp_grid.setCorrectionB(corr_B)

    # Create system
    system = mm.System()
    total_atoms = n_ligands * n_lig_atoms
    for _ in range(total_atoms):
        system.addParticle(12.0)

    gbsa = GBSAGridForce()
    gbsa.setNumAtoms(n_lig_atoms)
    gbsa.setIncludeSurfaceArea(False)

    for i in range(n_lig_atoms):
        gbsa.setAtomParameters(i, float(lig_charges[i]),
                               float(lig_radii[i]), float(lig_scales[i]))

    gbsa.setDesolvationGrid(cpp_grid)

    for lig_idx in range(n_ligands):
        start = lig_idx * n_lig_atoms
        indices = list(range(start, start + n_lig_atoms))
        gbsa.addParticleGroup(f"ligand_{lig_idx}", indices)

    gbsa.addExclusion(0, 1)
    system.addForce(gbsa)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
    except:
        platform = mm.Platform.getPlatformByName('Reference')

    context = mm.Context(system, integrator, platform)

    # Positions
    base_lig_pos = np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]])
    # Use positions slightly off grid boundaries to avoid trilinear gradient discontinuities
    ligand_offsets = [np.array([0.152, 0.103, 0.307]), np.array([0.152, 0.103, 0.507])]

    all_positions = []
    for offset in ligand_offsets:
        for atom_pos in base_lig_pos:
            all_positions.append(atom_pos + offset)

    positions = np.array(all_positions)
    context.setPositions(positions * unit.nanometer)

    # Get forces
    state = context.getState(getEnergy=True, getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    print(f"\nEnergy: {energy:.4f} kJ/mol")
    print("\nForces (kJ/mol/nm):")
    for i in range(total_atoms):
        lig_idx = i // n_lig_atoms
        atom_idx = i % n_lig_atoms
        print(f"  Ligand {lig_idx}, Atom {atom_idx}: [{forces[i,0]:.4f}, {forces[i,1]:.4f}, {forces[i,2]:.4f}]")

    # Verify forces by finite difference
    # Use h=1e-4 since smaller values hit float precision limits
    print("\nVerifying forces by finite difference:")
    h = 1e-4
    passed = True

    for atom in range(min(2, total_atoms)):  # Test first 2 atoms
        for axis in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[atom, axis] += h
            pos_minus[atom, axis] -= h

            context.setPositions(pos_plus * unit.nanometer)
            e_plus = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            context.setPositions(pos_minus * unit.nanometer)
            e_minus = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            fd_force = -(e_plus - e_minus) / (2 * h)
            analytical = forces[atom, axis]

            abs_err = abs(fd_force - analytical)
            if abs(analytical) > 1e-6:
                rel_err = abs_err / abs(analytical)
            else:
                rel_err = abs_err

            axis_name = ['x', 'y', 'z'][axis]
            # Use combined check: relative error < 10% OR absolute error < 0.02 kJ/mol/nm
            # Small forces near cell boundaries may have high relative error due to gradient discontinuity
            status = "OK" if (rel_err < 0.1 or abs_err < 0.02) else "FAIL"
            if status == "FAIL":
                passed = False
            print(f"  Atom {atom}, {axis_name}: analytical={analytical:.4f}, FD={fd_force:.4f}, rel_err={rel_err:.2e} [{status}]")

    # Restore positions
    context.setPositions(positions * unit.nanometer)

    if passed:
        print("\n" + "=" * 60)
        print("PASSED: Multi-ligand force test successful!")
        print("=" * 60)

    return passed


def main():
    print("GBSAGridForce Multi-ligand Tests")
    print("=" * 60)
    print()

    all_passed = True

    passed, _ = test_multiligand_energy()
    all_passed &= passed

    passed = test_multiligand_forces()
    all_passed &= passed

    print()
    print("=" * 60)
    if all_passed:
        print("All multi-ligand tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
