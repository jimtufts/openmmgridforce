"""
Tests for automatic grid generation on GPU platforms (CUDA, OpenCL).
These tests are skipped if the GPU platforms are not available.
"""
from __future__ import print_function
import pytest
from openmm import app
import openmm as omm
from openmm import unit
import gridforceplugin
import numpy as np


def platform_available(platform_name):
    """Check if a platform is available."""
    try:
        omm.Platform.getPlatformByName(platform_name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not platform_available('CUDA'), reason="CUDA platform not available")
def test_cuda_autogrid_generation():
    """Test automatic grid generation on CUDA platform."""
    # Simple test system with one charged particle
    system = omm.System()
    system.addParticle(1.0)

    # Create NonbondedForce with one charged particle
    nb_force = omm.NonbondedForce()
    nb_force.addParticle(1.0, 0.3, 0.1)  # charge, sigma, epsilon
    system.addForce(nb_force)

    # Create grid with auto-generation
    force = gridforceplugin.GridForce()
    force.addGridCounts(8, 8, 8)
    force.addGridSpacing(0.1, 0.1, 0.1)
    force.setGridOrigin(0.0, 0.0, 0.0)
    force.setAutoGenerateGrid(True)
    force.setGridType("charge")
    force.setReceptorAtoms([0])
    force.setReceptorPositionsFromLists([(0.4, 0.4, 0.4)])

    # Enable auto-scaling
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("charge")

    system.addForce(force)

    # Initialize on CUDA platform
    platform = omm.Platform.getPlatformByName('CUDA')
    integrator = omm.VerletIntegrator(1.0*unit.femtoseconds)
    context = omm.Context(system, integrator, platform)
    context.setPositions([(0.2, 0.2, 0.2)])

    # Get energy to verify the force is working
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Verify energy is finite (grid was generated and evaluated)
    assert np.isfinite(energy), "Energy should be finite"
    assert -1e6 < energy < 1e6, f"Energy out of reasonable range: {energy}"

    print(f"[PASS] CUDA auto-generation (energy={energy:.2f} kJ/mol)")

    del context


@pytest.mark.skipif(not platform_available('OpenCL'), reason="OpenCL platform not available")
def test_opencl_autogrid_generation():
    """Test automatic grid generation on OpenCL platform."""
    # Simple test system with one charged particle
    system = omm.System()
    system.addParticle(1.0)

    # Create NonbondedForce with one charged particle
    nb_force = omm.NonbondedForce()
    nb_force.addParticle(1.0, 0.3, 0.1)  # charge, sigma, epsilon
    system.addForce(nb_force)

    # Create grid with auto-generation
    force = gridforceplugin.GridForce()
    force.addGridCounts(8, 8, 8)
    force.addGridSpacing(0.1, 0.1, 0.1)
    force.setGridOrigin(0.0, 0.0, 0.0)
    force.setAutoGenerateGrid(True)
    force.setGridType("charge")
    force.setReceptorAtoms([0])
    force.setReceptorPositionsFromLists([(0.4, 0.4, 0.4)])

    # Enable auto-scaling
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("charge")

    system.addForce(force)

    # Initialize on OpenCL platform
    platform = omm.Platform.getPlatformByName('OpenCL')
    integrator = omm.VerletIntegrator(1.0*unit.femtoseconds)
    context = omm.Context(system, integrator, platform)
    context.setPositions([(0.2, 0.2, 0.2)])

    # Get energy to verify the force is working
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Verify energy is finite (grid was generated and evaluated)
    assert np.isfinite(energy), "Energy should be finite"
    assert -1e6 < energy < 1e6, f"Energy out of reasonable range: {energy}"

    print(f"[PASS] OpenCL auto-generation (energy={energy:.2f} kJ/mol)")

    del context


@pytest.mark.skipif(not platform_available('CUDA'), reason="CUDA platform not available")
def test_cuda_all_grid_types():
    """Test all grid types (charge, ljr, lja) on CUDA platform."""
    # Simple test system
    system = omm.System()
    system.addParticle(1.0)

    # Create NonbondedForce
    nb_force = omm.NonbondedForce()
    nb_force.addParticle(0.5, 0.3, 0.1)  # charge, sigma, epsilon
    system.addForce(nb_force)

    platform = omm.Platform.getPlatformByName('CUDA')

    for grid_type in ['charge', 'ljr', 'lja']:
        # Create fresh system copy
        test_system = omm.System()
        test_system.addParticle(1.0)
        test_nb = omm.NonbondedForce()
        test_nb.addParticle(0.5, 0.3, 0.1)
        test_system.addForce(test_nb)

        # Create grid
        force = gridforceplugin.GridForce()
        force.addGridCounts(6, 6, 6)
        force.addGridSpacing(0.1, 0.1, 0.1)
        force.setAutoGenerateGrid(True)
        force.setGridType(grid_type)
        force.setReceptorAtoms([0])
        force.setReceptorPositionsFromLists([(0.3, 0.3, 0.3)])
        force.setAutoCalculateScalingFactors(True)
        force.setScalingProperty(grid_type)

        test_system.addForce(force)

        # Test
        integrator = omm.VerletIntegrator(1.0*unit.femtoseconds)
        context = omm.Context(test_system, integrator, platform)
        context.setPositions([(0.15, 0.15, 0.15)])

        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        assert np.isfinite(energy), f"Grid {grid_type}: Energy should be finite"
        print(f"[PASS] CUDA {grid_type} grid (energy={energy:.2f} kJ/mol)")

        del context, integrator


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
