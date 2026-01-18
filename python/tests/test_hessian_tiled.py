#!/usr/bin/env python
"""
Test tiled Hessian computation.

Validates that the tiled Hessian kernel produces identical results to
the non-tiled Hessian kernel. Tests with both simple systems and real
ligand data.
"""

import os
import sys
import numpy as np
import openmm as mm
from openmm.app import AmberPrmtopFile, AmberInpcrdFile
from openmm import Platform, Context, VerletIntegrator
from openmm.unit import nanometer, kilojoules_per_mole
import tempfile

import gridforceplugin as gfp

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(TEST_DIR), 'prmtopcrd')

# Coulomb constant
ONE_4PI_EPS0 = 138.935456  # kJ/mol * nm / e^2


def compute_pairwise_hessian_block(lig_pos, rec_pos, lig_params, rec_params, grid_type):
    """Compute analytical Hessian 3x3 block for a single ligand atom from pairwise interactions.

    Args:
        lig_pos: (3,) ligand atom position in nm
        rec_pos: (n_rec, 3) receptor positions in nm
        lig_params: (charge, sigma, epsilon) for this ligand atom
        rec_params: list of (charge, sigma, epsilon) for receptor atoms
        grid_type: 'charge', 'ljr', or 'lja'

    Returns:
        (3, 3) Hessian block for this atom
    """
    n_rec = len(rec_pos)
    H_block = np.zeros((3, 3))

    lig_q, lig_s, lig_e = lig_params

    for j in range(n_rec):
        rec_q, rec_s, rec_e = rec_params[j]

        # Vector from receptor atom j to ligand atom
        r_vec = lig_pos - rec_pos[j]
        r = np.linalg.norm(r_vec)

        if r < 0.01:  # Skip very close atoms
            continue

        # Outer product: r_i * r_j
        rr = np.outer(r_vec, r_vec)
        I = np.eye(3)

        if grid_type == 'charge':
            # E = k * q1 * q2 / r
            # d²E/dr_i dr_j = k * q1 * q2 * [3 * r_i * r_j / r⁵ - δ_ij / r³]
            k = ONE_4PI_EPS0
            q1q2 = lig_q * rec_q
            H_block += k * q1q2 * (3.0 * rr / r**5 - I / r**3)

        elif grid_type == 'ljr':
            # E = 4 * eps * (sig/r)^12
            # d²E/dr_i dr_j = 48 * eps * sig^12 * [14 * r_i * r_j / r^16 - δ_ij / r^14]
            sig = np.sqrt(lig_s * rec_s)
            eps = np.sqrt(lig_e * rec_e)
            sig12 = sig**12
            H_block += 48.0 * eps * sig12 * (14.0 * rr / r**16 - I / r**14)

        elif grid_type == 'lja':
            # E = -4 * eps * (sig/r)^6
            # d²E/dr_i dr_j = -24 * eps * sig^6 * [8 * r_i * r_j / r^10 - δ_ij / r^8]
            sig = np.sqrt(lig_s * rec_s)
            eps = np.sqrt(lig_e * rec_e)
            sig6 = sig**6
            H_block += -24.0 * eps * sig6 * (8.0 * rr / r**10 - I / r**8)

    return H_block


def generate_test_grid(grid_file, grid_type='charge', spacing=0.05, counts=(52, 70, 58)):
    """Generate a test grid from receptor data."""
    receptor_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'receptor.prmtop'))
    receptor_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'receptor.trans.inpcrd'))

    receptor_atoms = list(range(receptor_prmtop.topology.getNumAtoms()))
    pos_list = [(p[0].value_in_unit(nanometer), p[1].value_in_unit(nanometer),
                 p[2].value_in_unit(nanometer)) for p in receptor_inpcrd.positions]

    grid_origin = (1.00175115, 0.5328844699999999, 0.8606374500000002)

    system = receptor_prmtop.createSystem()
    grid = gfp.GridForce()
    grid.setGridOrigin(*grid_origin)
    grid.addGridCounts(*counts)
    grid.addGridSpacing(spacing, spacing, spacing)
    grid.setAutoGenerateGrid(True)
    grid.setGridType(grid_type)
    grid.setComputeDerivatives(True)
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

    grid.saveToFile(grid_file)
    del context

    return grid_origin, counts, spacing


def numerical_hessian_block(context, atom_idx, h=1e-5):
    """Compute numerical 3x3 Hessian block for a single atom using finite differences."""
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)

    H = np.zeros((3, 3))

    for i in range(3):
        # +h perturbation
        pos_plus = positions.copy()
        pos_plus[atom_idx, i] += h
        context.setPositions(pos_plus)
        f_plus = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
            kilojoules_per_mole / nanometer)

        # -h perturbation
        pos_minus = positions.copy()
        pos_minus[atom_idx, i] -= h
        context.setPositions(pos_minus)
        f_minus = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
            kilojoules_per_mole / nanometer)

        # H[i, j] = -dF_j/dx_i
        for j in range(3):
            H[i, j] = -(f_plus[atom_idx, j] - f_minus[atom_idx, j]) / (2 * h)

    # Restore original positions
    context.setPositions(positions)

    # Symmetrize
    H = (H + H.T) / 2
    return H


def test_tiled_vs_nontiled_simple():
    """Test tiled vs non-tiled Hessian with a simple 2-atom system."""
    print("\n" + "=" * 70)
    print("TEST: Tiled vs Non-tiled Hessian (Simple 2-atom system)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        grid_file = os.path.join(tmpdir, 'charge.grid')
        print("  Generating test grid...")
        generate_test_grid(grid_file, 'charge')

        platform = Platform.getPlatformByName('CUDA')
        q1, q2 = 0.3, -0.3
        pos1 = np.array([1.5, 2.0, 1.8])
        pos2 = pos1 + np.array([0.15, 0.0, 0.0])
        positions = np.array([pos1, pos2])

        # Non-tiled
        system_nontiled = mm.System()
        system_nontiled.addParticle(12.0)
        system_nontiled.addParticle(12.0)

        gridforce_nontiled = gfp.GridForce()
        gridforce_nontiled.loadFromFile(grid_file)
        gridforce_nontiled.setInterpolationMethod(3)  # Triquintic
        gridforce_nontiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_nontiled.addScalingFactor(q1)
        gridforce_nontiled.addScalingFactor(q2)
        system_nontiled.addForce(gridforce_nontiled)

        integrator_nontiled = VerletIntegrator(0.001)
        context_nontiled = Context(system_nontiled, integrator_nontiled, platform)
        context_nontiled.setPositions(positions)

        state_nontiled = context_nontiled.getState(getEnergy=True)
        energy_nontiled = state_nontiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_nontiled.computeHessian(context_nontiled)
        H_nontiled = np.array(gridforce_nontiled.getHessianMatrices(context_nontiled))
        del context_nontiled

        # Tiled
        system_tiled = mm.System()
        system_tiled.addParticle(12.0)
        system_tiled.addParticle(12.0)

        gridforce_tiled = gfp.GridForce()
        gridforce_tiled.loadFromFile(grid_file)
        gridforce_tiled.setInterpolationMethod(3)  # Triquintic
        gridforce_tiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_tiled.setTiledMode(True, 16, 100)  # Enable tiled mode
        gridforce_tiled.addScalingFactor(q1)
        gridforce_tiled.addScalingFactor(q2)
        system_tiled.addForce(gridforce_tiled)

        integrator_tiled = VerletIntegrator(0.001)
        context_tiled = Context(system_tiled, integrator_tiled, platform)
        context_tiled.setPositions(positions)

        state_tiled = context_tiled.getState(getEnergy=True)
        energy_tiled = state_tiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_tiled.computeHessian(context_tiled)
        H_tiled = np.array(gridforce_tiled.getHessianMatrices(context_tiled))
        del context_tiled

        # Compare energies
        energy_diff = abs(energy_tiled - energy_nontiled)
        print(f"  Non-tiled energy: {energy_nontiled:.6f} kJ/mol")
        print(f"  Tiled energy: {energy_tiled:.6f} kJ/mol")
        print(f"  Energy difference: {energy_diff:.2e} kJ/mol")

        # Compare Hessians
        hessian_diff = np.abs(H_tiled - H_nontiled)
        max_diff = hessian_diff.max()
        mean_diff = hessian_diff.mean()

        print(f"  Hessian max |diff|: {max_diff:.2e}")
        print(f"  Hessian mean |diff|: {mean_diff:.2e}")

        passed = energy_diff < 1e-6 and max_diff < 1e-6
        print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
        return passed


def test_tiled_vs_nontiled_ligand():
    """Test tiled vs non-tiled Hessian with a real ligand."""
    print("\n" + "=" * 70)
    print("TEST: Tiled vs Non-tiled Hessian (Real Ligand)")
    print("=" * 70)

    ligand_prmtop = os.path.join(DATA_DIR, 'ligand.prmtop')
    ligand_inpcrd = os.path.join(DATA_DIR, 'ligand.trans.inpcrd')

    if not os.path.exists(ligand_prmtop):
        print("  Ligand files not found, skipping test")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        grid_file = os.path.join(tmpdir, 'charge.grid')
        print("  Generating test grid...")
        generate_test_grid(grid_file, 'charge')

        # Load ligand
        ligand_prmtop_obj = AmberPrmtopFile(ligand_prmtop)
        ligand_inpcrd_obj = AmberInpcrdFile(ligand_inpcrd)
        n_atoms = ligand_prmtop_obj.topology.getNumAtoms()
        print(f"  Ligand has {n_atoms} atoms")

        platform = Platform.getPlatformByName('CUDA')

        # Get charges
        lig_sys = ligand_prmtop_obj.createSystem()
        nb_force = None
        for i in range(lig_sys.getNumForces()):
            force = lig_sys.getForce(i)
            if isinstance(force, mm.NonbondedForce):
                nb_force = force
                break

        charges = []
        for i in range(n_atoms):
            q, sigma, epsilon = nb_force.getParticleParameters(i)
            charges.append(q.value_in_unit_system(mm.unit.md_unit_system))

        # Non-tiled
        system_nontiled = mm.System()
        for i in range(n_atoms):
            system_nontiled.addParticle(12.0)

        gridforce_nontiled = gfp.GridForce()
        gridforce_nontiled.loadFromFile(grid_file)
        gridforce_nontiled.setInterpolationMethod(3)  # Triquintic
        gridforce_nontiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        for q in charges:
            gridforce_nontiled.addScalingFactor(q)
        system_nontiled.addForce(gridforce_nontiled)

        integrator_nontiled = VerletIntegrator(0.001)
        context_nontiled = Context(system_nontiled, integrator_nontiled, platform)
        context_nontiled.setPositions(ligand_inpcrd_obj.positions)

        state_nontiled = context_nontiled.getState(getEnergy=True)
        energy_nontiled = state_nontiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_nontiled.computeHessian(context_nontiled)
        H_nontiled = np.array(gridforce_nontiled.getHessianMatrices(context_nontiled))
        del context_nontiled

        # Tiled
        system_tiled = mm.System()
        for i in range(n_atoms):
            system_tiled.addParticle(12.0)

        gridforce_tiled = gfp.GridForce()
        gridforce_tiled.loadFromFile(grid_file)
        gridforce_tiled.setInterpolationMethod(3)  # Triquintic
        gridforce_tiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_tiled.setTiledMode(True, 16, 100)  # Enable tiled mode
        for q in charges:
            gridforce_tiled.addScalingFactor(q)
        system_tiled.addForce(gridforce_tiled)

        integrator_tiled = VerletIntegrator(0.001)
        context_tiled = Context(system_tiled, integrator_tiled, platform)
        context_tiled.setPositions(ligand_inpcrd_obj.positions)

        state_tiled = context_tiled.getState(getEnergy=True)
        energy_tiled = state_tiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_tiled.computeHessian(context_tiled)
        H_tiled = np.array(gridforce_tiled.getHessianMatrices(context_tiled))
        del context_tiled

        # Compare energies
        energy_diff = abs(energy_tiled - energy_nontiled)
        print(f"  Non-tiled energy: {energy_nontiled:.6f} kJ/mol")
        print(f"  Tiled energy: {energy_tiled:.6f} kJ/mol")
        print(f"  Energy difference: {energy_diff:.2e} kJ/mol")

        # Compare Hessians
        hessian_diff = np.abs(H_tiled - H_nontiled)
        max_diff = hessian_diff.max()
        mean_diff = hessian_diff.mean()

        print(f"  Hessian max |diff|: {max_diff:.2e}")
        print(f"  Hessian mean |diff|: {mean_diff:.2e}")

        passed = energy_diff < 1e-4 and max_diff < 1e-4
        print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
        return passed


def test_tiled_vs_numerical():
    """Validate tiled Hessian against numerical finite differences."""
    print("\n" + "=" * 70)
    print("TEST: Tiled Hessian vs Numerical Finite Differences")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        grid_file = os.path.join(tmpdir, 'charge.grid')
        print("  Generating test grid...")
        generate_test_grid(grid_file, 'charge')

        platform = Platform.getPlatformByName('CUDA')
        q1, q2 = 0.3, -0.3
        pos1 = np.array([1.5, 2.0, 1.8])
        pos2 = pos1 + np.array([0.15, 0.0, 0.0])
        positions = np.array([pos1, pos2])

        # Tiled system
        system = mm.System()
        system.addParticle(12.0)
        system.addParticle(12.0)

        gridforce = gfp.GridForce()
        gridforce.loadFromFile(grid_file)
        gridforce.setInterpolationMethod(3)  # Triquintic
        gridforce.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce.setTiledMode(True, 16, 100)  # Enable tiled mode
        gridforce.addScalingFactor(q1)
        gridforce.addScalingFactor(q2)
        system.addForce(gridforce)

        integrator = VerletIntegrator(0.001)
        context = Context(system, integrator, platform)
        context.setPositions(positions)

        # Analytical Hessian from tiled kernel
        gridforce.computeHessian(context)
        H_analytical = np.array(gridforce.getHessianMatrices(context))

        # Numerical Hessian
        print("  Computing numerical Hessian (finite differences)...")
        H_numerical = np.zeros((2, 3, 3))
        for atom_idx in range(2):
            H_numerical[atom_idx] = numerical_hessian_block(context, atom_idx, h=1e-5)

        del context

        # Compare
        diff = np.abs(H_analytical - H_numerical)
        max_diff = diff.max()
        max_val = np.max(np.abs(H_numerical))
        rel_error = max_diff / (max_val + 1e-10)

        print(f"  Analytical Hessian range: [{H_analytical.min():.2e}, {H_analytical.max():.2e}]")
        print(f"  Numerical Hessian range: [{H_numerical.min():.2e}, {H_numerical.max():.2e}]")
        print(f"  Max |diff|: {max_diff:.2e}")
        print(f"  Max |H_num|: {max_val:.2e}")
        print(f"  Relative error: {rel_error:.4f}")

        # Allow 1% relative error for numerical vs analytical
        passed = rel_error < 0.01
        print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
        return passed


def test_tiled_bspline():
    """Test tiled Hessian with bspline interpolation."""
    print("\n" + "=" * 70)
    print("TEST: Tiled vs Non-tiled Hessian (B-spline interpolation)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        grid_file = os.path.join(tmpdir, 'charge.grid')
        print("  Generating test grid...")
        generate_test_grid(grid_file, 'charge')

        platform = Platform.getPlatformByName('CUDA')
        q1, q2 = 0.3, -0.3
        pos1 = np.array([1.5, 2.0, 1.8])
        pos2 = pos1 + np.array([0.15, 0.0, 0.0])
        positions = np.array([pos1, pos2])

        # Non-tiled with bspline
        system_nontiled = mm.System()
        system_nontiled.addParticle(12.0)
        system_nontiled.addParticle(12.0)

        gridforce_nontiled = gfp.GridForce()
        gridforce_nontiled.loadFromFile(grid_file)
        gridforce_nontiled.setInterpolationMethod(1)  # bspline
        gridforce_nontiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_nontiled.addScalingFactor(q1)
        gridforce_nontiled.addScalingFactor(q2)
        system_nontiled.addForce(gridforce_nontiled)

        integrator_nontiled = VerletIntegrator(0.001)
        context_nontiled = Context(system_nontiled, integrator_nontiled, platform)
        context_nontiled.setPositions(positions)

        state_nontiled = context_nontiled.getState(getEnergy=True)
        energy_nontiled = state_nontiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_nontiled.computeHessian(context_nontiled)
        H_nontiled = np.array(gridforce_nontiled.getHessianMatrices(context_nontiled))
        del context_nontiled

        # Tiled with bspline
        system_tiled = mm.System()
        system_tiled.addParticle(12.0)
        system_tiled.addParticle(12.0)

        gridforce_tiled = gfp.GridForce()
        gridforce_tiled.loadFromFile(grid_file)
        gridforce_tiled.setInterpolationMethod(1)  # bspline
        gridforce_tiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_tiled.setTiledMode(True, 16, 100)  # Enable tiled mode
        gridforce_tiled.addScalingFactor(q1)
        gridforce_tiled.addScalingFactor(q2)
        system_tiled.addForce(gridforce_tiled)

        integrator_tiled = VerletIntegrator(0.001)
        context_tiled = Context(system_tiled, integrator_tiled, platform)
        context_tiled.setPositions(positions)

        state_tiled = context_tiled.getState(getEnergy=True)
        energy_tiled = state_tiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_tiled.computeHessian(context_tiled)
        H_tiled = np.array(gridforce_tiled.getHessianMatrices(context_tiled))
        del context_tiled

        # Compare
        energy_diff = abs(energy_tiled - energy_nontiled)
        hessian_diff = np.abs(H_tiled - H_nontiled)
        max_diff = hessian_diff.max()

        print(f"  Non-tiled energy: {energy_nontiled:.6f} kJ/mol")
        print(f"  Tiled energy: {energy_tiled:.6f} kJ/mol")
        print(f"  Energy difference: {energy_diff:.2e} kJ/mol")
        print(f"  Hessian max |diff|: {max_diff:.2e}")

        passed = energy_diff < 1e-6 and max_diff < 1e-6
        print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
        return passed


def test_tiled_triquintic():
    """Test tiled Hessian with triquintic interpolation."""
    print("\n" + "=" * 70)
    print("TEST: Tiled vs Non-tiled Hessian (Triquintic interpolation)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        grid_file = os.path.join(tmpdir, 'charge.grid')
        print("  Generating test grid...")
        generate_test_grid(grid_file, 'charge')

        platform = Platform.getPlatformByName('CUDA')
        q1, q2 = 0.3, -0.3
        pos1 = np.array([1.5, 2.0, 1.8])
        pos2 = pos1 + np.array([0.15, 0.0, 0.0])
        positions = np.array([pos1, pos2])

        # Non-tiled with triquintic
        system_nontiled = mm.System()
        system_nontiled.addParticle(12.0)
        system_nontiled.addParticle(12.0)

        gridforce_nontiled = gfp.GridForce()
        gridforce_nontiled.loadFromFile(grid_file)
        gridforce_nontiled.setInterpolationMethod(3)  # triquintic
        gridforce_nontiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_nontiled.addScalingFactor(q1)
        gridforce_nontiled.addScalingFactor(q2)
        system_nontiled.addForce(gridforce_nontiled)

        integrator_nontiled = VerletIntegrator(0.001)
        context_nontiled = Context(system_nontiled, integrator_nontiled, platform)
        context_nontiled.setPositions(positions)

        state_nontiled = context_nontiled.getState(getEnergy=True)
        energy_nontiled = state_nontiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_nontiled.computeHessian(context_nontiled)
        H_nontiled = np.array(gridforce_nontiled.getHessianMatrices(context_nontiled))
        del context_nontiled

        # Tiled with triquintic
        system_tiled = mm.System()
        system_tiled.addParticle(12.0)
        system_tiled.addParticle(12.0)

        gridforce_tiled = gfp.GridForce()
        gridforce_tiled.loadFromFile(grid_file)
        gridforce_tiled.setInterpolationMethod(3)  # triquintic
        gridforce_tiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_tiled.setTiledMode(True, 16, 100)  # Enable tiled mode
        gridforce_tiled.addScalingFactor(q1)
        gridforce_tiled.addScalingFactor(q2)
        system_tiled.addForce(gridforce_tiled)

        integrator_tiled = VerletIntegrator(0.001)
        context_tiled = Context(system_tiled, integrator_tiled, platform)
        context_tiled.setPositions(positions)

        state_tiled = context_tiled.getState(getEnergy=True)
        energy_tiled = state_tiled.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        gridforce_tiled.computeHessian(context_tiled)
        H_tiled = np.array(gridforce_tiled.getHessianMatrices(context_tiled))
        del context_tiled

        # Compare
        energy_diff = abs(energy_tiled - energy_nontiled)
        hessian_diff = np.abs(H_tiled - H_nontiled)
        max_diff = hessian_diff.max()

        print(f"  Non-tiled energy: {energy_nontiled:.6f} kJ/mol")
        print(f"  Tiled energy: {energy_tiled:.6f} kJ/mol")
        print(f"  Energy difference: {energy_diff:.2e} kJ/mol")
        print(f"  Hessian max |diff|: {max_diff:.2e}")

        passed = energy_diff < 1e-6 and max_diff < 1e-6
        print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
        return passed


def test_different_tile_sizes():
    """Test tiled Hessian with different tile sizes."""
    print("\n" + "=" * 70)
    print("TEST: Tiled Hessian with Different Tile Sizes")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        grid_file = os.path.join(tmpdir, 'charge.grid')
        print("  Generating test grid...")
        generate_test_grid(grid_file, 'charge')

        platform = Platform.getPlatformByName('CUDA')
        q1, q2 = 0.3, -0.3
        pos1 = np.array([1.5, 2.0, 1.8])
        pos2 = pos1 + np.array([0.15, 0.0, 0.0])
        positions = np.array([pos1, pos2])

        # Get reference (non-tiled)
        system_ref = mm.System()
        system_ref.addParticle(12.0)
        system_ref.addParticle(12.0)

        gridforce_ref = gfp.GridForce()
        gridforce_ref.loadFromFile(grid_file)
        gridforce_ref.setInterpolationMethod(3)
        gridforce_ref.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
        gridforce_ref.addScalingFactor(q1)
        gridforce_ref.addScalingFactor(q2)
        system_ref.addForce(gridforce_ref)

        integrator_ref = VerletIntegrator(0.001)
        context_ref = Context(system_ref, integrator_ref, platform)
        context_ref.setPositions(positions)

        gridforce_ref.computeHessian(context_ref)
        H_ref = np.array(gridforce_ref.getHessianMatrices(context_ref))
        del context_ref

        tile_sizes = [8, 16, 32]
        all_passed = True

        for tile_size in tile_sizes:
            system_tiled = mm.System()
            system_tiled.addParticle(12.0)
            system_tiled.addParticle(12.0)

            gridforce_tiled = gfp.GridForce()
            gridforce_tiled.loadFromFile(grid_file)
            gridforce_tiled.setInterpolationMethod(3)
            gridforce_tiled.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)
            gridforce_tiled.setTiledMode(True, tile_size, 100)
            gridforce_tiled.addScalingFactor(q1)
            gridforce_tiled.addScalingFactor(q2)
            system_tiled.addForce(gridforce_tiled)

            integrator_tiled = VerletIntegrator(0.001)
            context_tiled = Context(system_tiled, integrator_tiled, platform)
            context_tiled.setPositions(positions)

            gridforce_tiled.computeHessian(context_tiled)
            H_tiled = np.array(gridforce_tiled.getHessianMatrices(context_tiled))
            del context_tiled

            max_diff = np.abs(H_tiled - H_ref).max()
            passed = max_diff < 1e-6
            all_passed = all_passed and passed

            print(f"  Tile size {tile_size}: max |diff| = {max_diff:.2e} {'PASS' if passed else 'FAIL'}")

        print(f"  STATUS: {'PASS' if all_passed else 'FAIL'}")
        return all_passed


def test_grid_vs_analytical_comprehensive():
    """Compare grid Hessian against analytical pairwise Hessian for all grid types."""
    print("\n" + "=" * 70)
    print("TEST: Grid Hessian vs Analytical Pairwise (All Grid Types)")
    print("=" * 70)

    # Load receptor and ligand data
    receptor_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'receptor.prmtop'))
    receptor_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'receptor.trans.inpcrd'))
    ligand_prmtop = AmberPrmtopFile(os.path.join(DATA_DIR, 'ligand.prmtop'))
    ligand_inpcrd = AmberInpcrdFile(os.path.join(DATA_DIR, 'ligand.trans.inpcrd'))

    n_lig = ligand_prmtop.topology.getNumAtoms()
    n_rec = receptor_prmtop.topology.getNumAtoms()
    print(f"  Ligand: {n_lig} atoms, Receptor: {n_rec} atoms")

    # Get positions
    lig_pos = np.array(ligand_inpcrd.positions.value_in_unit(nanometer))
    rec_pos = np.array(receptor_inpcrd.positions.value_in_unit(nanometer))

    # Get parameters
    lig_sys = ligand_prmtop.createSystem()
    rec_sys = receptor_prmtop.createSystem()

    lig_nb = [f for f in lig_sys.getForces() if isinstance(f, mm.NonbondedForce)][0]
    rec_nb = [f for f in rec_sys.getForces() if isinstance(f, mm.NonbondedForce)][0]

    lig_params = []
    for i in range(n_lig):
        q, s, e = lig_nb.getParticleParameters(i)
        lig_params.append((q.value_in_unit_system(mm.unit.md_unit_system),
                          s.value_in_unit(nanometer),
                          e.value_in_unit(kilojoules_per_mole)))

    rec_params = []
    for i in range(n_rec):
        q, s, e = rec_nb.getParticleParameters(i)
        rec_params.append((q.value_in_unit_system(mm.unit.md_unit_system),
                          s.value_in_unit(nanometer),
                          e.value_in_unit(kilojoules_per_mole)))

    platform = Platform.getPlatformByName('CUDA')

    # Test configurations
    grid_types = ['charge', 'ljr', 'lja']
    methods = [(1, 'bspline'), (3, 'triquintic')]

    results = {}
    all_passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        receptor_atoms = list(range(n_rec))
        rec_pos_list = [(p[0], p[1], p[2]) for p in rec_pos]

        # Grid centered on ligand
        center = lig_pos.mean(axis=0)
        origin = tuple(center - 1.5)
        spacing = 0.02  # 0.2 Angstrom
        counts = (150, 150, 150)

        for grid_type in grid_types:
            print(f"\n  --- Grid type: {grid_type} ---")

            # Generate grid
            grid_file = os.path.join(tmpdir, f'{grid_type}.grid')

            gen_sys = receptor_prmtop.createSystem()
            grid = gfp.GridForce()
            grid.setGridOrigin(*origin)
            grid.addGridCounts(*counts)
            grid.addGridSpacing(spacing, spacing, spacing)
            grid.setAutoGenerateGrid(True)
            grid.setGridType(grid_type)
            grid.setComputeDerivatives(True)
            grid.setGridCap(1e30)
            grid.setReceptorAtoms(receptor_atoms)
            grid.setReceptorPositionsFromLists(rec_pos_list)
            grid.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

            gen_sys.addForce(grid)
            gen_integrator = VerletIntegrator(0.001)
            gen_context = Context(gen_sys, gen_integrator, platform)
            gen_context.setPositions(receptor_inpcrd.positions)
            gen_context.getState(getEnergy=True)
            grid.saveToFile(grid_file)
            del gen_context

            # Compute analytical pairwise Hessian for this grid type
            H_analytical = np.zeros((n_lig, 3, 3))
            for i in range(n_lig):
                H_analytical[i] = compute_pairwise_hessian_block(
                    lig_pos[i], rec_pos, lig_params[i], rec_params, grid_type)

            analytical_eigs = np.array([np.linalg.eigvalsh(H_analytical[i]) for i in range(n_lig)])

            # Compute scaling factors based on grid type
            if grid_type == 'charge':
                scaling = [p[0] for p in lig_params]  # charge
            elif grid_type == 'ljr':
                # LJR scaling: sqrt(eps) * 2 * sig^6
                scaling = [np.sqrt(p[2]) * 2.0 * (p[1]**6) for p in lig_params]
            elif grid_type == 'lja':
                # LJA scaling: sqrt(eps) * sqrt(2) * sig^3
                scaling = [np.sqrt(p[2]) * np.sqrt(2.0) * (p[1]**3) for p in lig_params]

            for method_id, method_name in methods:
                # Create evaluation system
                eval_sys = mm.System()
                for i in range(n_lig):
                    eval_sys.addParticle(12.0)

                gridforce = gfp.GridForce()
                gridforce.loadFromFile(grid_file)
                gridforce.setInterpolationMethod(method_id)
                gridforce.setInvPowerMode(gfp.InvPowerMode_NONE, 0.0)

                for s in scaling:
                    gridforce.addScalingFactor(s)

                eval_sys.addForce(gridforce)

                eval_integrator = VerletIntegrator(0.001)
                eval_context = Context(eval_sys, eval_integrator, platform)
                eval_context.setPositions(ligand_inpcrd.positions)

                # Compute grid Hessian
                gridforce.computeHessian(eval_context)
                H_grid = np.array(gridforce.getHessianMatrices(eval_context))
                del eval_context

                # Compare eigenvalues
                grid_eigs = np.array([np.linalg.eigvalsh(H_grid[i]) for i in range(n_lig)])

                eig_diff = np.abs(grid_eigs - analytical_eigs)
                rel_err = eig_diff / (np.abs(analytical_eigs) + 1e-10)

                mean_err = rel_err.mean() * 100
                max_err = rel_err.max() * 100

                results[(grid_type, method_name)] = (mean_err, max_err)
                print(f"    {method_name}: mean err = {mean_err:.2f}%, max err = {max_err:.2f}%")

    # Summary table
    print("\n  Summary (mean relative error %):")
    print("  " + "-" * 40)
    print(f"  {'Grid Type':<10} {'bspline':<12} {'triquintic':<12}")
    print("  " + "-" * 40)
    for grid_type in grid_types:
        bspline_err = results[(grid_type, 'bspline')][0]
        triquintic_err = results[(grid_type, 'triquintic')][0]
        winner = "←" if bspline_err < triquintic_err else "→"
        print(f"  {grid_type:<10} {bspline_err:>10.2f}%  {triquintic_err:>10.2f}% {winner}")
    print("  " + "-" * 40)

    # Pass if all mean errors are under 30%
    for key, (mean_err, max_err) in results.items():
        if mean_err > 30:
            all_passed = False

    print(f"\n  STATUS: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def main():
    """Run all tests."""
    results = []

    results.append(("Simple 2-atom (tiled vs non-tiled)", test_tiled_vs_nontiled_simple()))
    results.append(("Real ligand (tiled vs non-tiled)", test_tiled_vs_nontiled_ligand()))
    results.append(("Tiled vs numerical FD", test_tiled_vs_numerical()))
    results.append(("B-spline interpolation", test_tiled_bspline()))
    results.append(("Triquintic interpolation", test_tiled_triquintic()))
    results.append(("Different tile sizes", test_different_tile_sizes()))
    results.append(("Grid vs analytical (all types)", test_grid_vs_analytical_comprehensive()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(f"  {name}: {status}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
