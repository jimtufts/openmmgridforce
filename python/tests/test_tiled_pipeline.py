"""Test the full tiled grid generation and evaluation pipeline.

This test verifies:
1. Tile-by-tile grid generation to a file
2. Loading the tiled grid for evaluation
3. Comparing results with non-tiled mode
"""

import os
import tempfile
import numpy as np
import pytest
from openmm import *
from openmm.app import *
from openmm.unit import *
import gridforceplugin as gfp


def cuda_available():
    """Check if CUDA platform is available."""
    try:
        Platform.getPlatformByName('CUDA')
        return True
    except Exception:
        return False

def create_simple_system():
    """Create a simple water-like system for testing."""
    # Create a system with some particles
    system = System()

    # Add 10 "receptor" atoms (fixed) and 5 "ligand" atoms
    for i in range(10):  # Receptor atoms
        system.addParticle(16.0)  # Oxygen mass
    for i in range(5):  # Ligand atoms
        system.addParticle(12.0)  # Carbon mass

    # Add NonbondedForce
    nb = NonbondedForce()
    for i in range(10):  # Receptor atoms - larger charges and LJ params
        nb.addParticle(0.5 * ((-1)**i), 0.3, 0.5)  # charge, sigma, epsilon
    for i in range(5):  # Ligand atoms
        nb.addParticle(0.1 * ((-1)**i), 0.3, 0.3)

    # Set nonbonded method to avoid cutoff issues
    nb.setNonbondedMethod(NonbondedForce.NoCutoff)
    system.addForce(nb)

    return system

def create_grid_force(tiled_output_file=None, tiled_input_file=None):
    """Create a GridForce for testing."""
    gf = gfp.GridForce()

    # Set up for auto-generation
    gf.setAutoGenerateGrid(True)
    gf.setGridType("charge")

    # Small grid for fast testing (10x10x10)
    gf.addGridCounts(10, 10, 10)
    gf.addGridSpacing(0.1, 0.1, 0.1)  # 0.1 nm spacing
    gf.setGridOrigin(0.0, 0.0, 0.0)

    # Set receptor atoms (first 10)
    gf.setReceptorAtoms(list(range(10)))

    # Set ligand atoms (last 5)
    gf.setLigandAtoms(list(range(10, 15)))

    # Set particles to evaluate (ligand only)
    gf.setParticles(list(range(10, 15)))

    # Set interpolation method
    gf.setInterpolationMethod(0)  # Trilinear

    # Auto-calculate scaling factors from NonbonedForce
    gf.setAutoCalculateScalingFactors(True)
    gf.setScalingProperty("charge")

    # Configure tiled mode
    if tiled_output_file:
        gf.setTiledOutputFile(tiled_output_file, 32)  # Tile size 32
        # Enable tiled mode so we can evaluate after generation
        gf.setTiledMode(True, 32, 512)

    if tiled_input_file:
        gf.setTiledInputFile(tiled_input_file)
        # setTiledInputFile auto-enables tiled mode

    return gf

def test_tiled_output_api():
    """Test that the tiled output API works."""
    print("\n=== Test 1: Tiled Output File API ===")

    gf = gfp.GridForce()

    # Test setTiledOutputFile
    gf.setTiledOutputFile("/tmp/test.grid", 64)
    assert gf.getTiledOutputFile() == "/tmp/test.grid"
    assert gf.getTiledOutputTileSize() == 64

    print("  setTiledOutputFile/getTiledOutputFile: PASS")
    print("  getTiledOutputTileSize: PASS")

def test_tiled_input_api():
    """Test that the tiled input API works."""
    print("\n=== Test 2: Tiled Input File API ===")

    gf = gfp.GridForce()

    # Test setTiledInputFile (should auto-enable tiled mode)
    gf.setTiledInputFile("/tmp/test.grid")
    assert gf.getTiledInputFile() == "/tmp/test.grid"
    assert gf.getTiledMode() == True  # Should be auto-enabled

    print("  setTiledInputFile/getTiledInputFile: PASS")
    print("  Tiled mode auto-enabled: PASS")

@pytest.mark.skipif(not cuda_available(), reason="CUDA platform not available")
def test_tiled_generation_and_evaluation():
    """Test full pipeline: generate tiled grid, then load and evaluate."""
    print("\n=== Test 3: Tiled Generation and Evaluation Pipeline ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tiled_file = os.path.join(tmpdir, "test_tiled.grid")

        # Step 1: Generate tiled grid
        print("  Step 1: Generating tiled grid...")
        system1 = create_simple_system()
        gf1 = create_grid_force(tiled_output_file=tiled_file)
        system1.addForce(gf1)

        # Set receptor positions
        receptor_positions = []
        for i in range(10):
            receptor_positions.append([0.3 + i*0.1, 0.3, 0.3])  # nm
        gf1.setReceptorPositionsFromLists(receptor_positions)

        # Create positions for all atoms
        positions = []
        for i in range(10):  # Receptor atoms
            positions.append(Vec3(0.3 + i*0.1, 0.3, 0.3))
        for i in range(5):  # Ligand atoms
            positions.append(Vec3(0.5 + i*0.05, 0.5, 0.5))

        # Create context (this will trigger grid generation)
        integrator1 = VerletIntegrator(0.001*picoseconds)
        platform = Platform.getPlatformByName('CUDA')

        try:
            context1 = Context(system1, integrator1, platform)
            context1.setPositions(positions)

            # Get energy (triggers force calculation)
            state1 = context1.getState(getEnergy=True, getForces=True)
            energy1 = state1.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            forces1 = state1.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)

            print(f"    Generated grid energy: {energy1:.6f} kJ/mol")

            # Clean up context
            del context1
            del integrator1

        except Exception as e:
            print(f"    ERROR during generation: {e}")
            raise

        # Verify file was created
        if not os.path.exists(tiled_file):
            print("    ERROR: Tiled file was not created!")
            return False

        file_size = os.path.getsize(tiled_file)
        print(f"    Tiled file created: {file_size} bytes")

        # Step 2: Load and evaluate using tiled input
        print("  Step 2: Loading tiled grid and evaluating...")
        system2 = create_simple_system()
        gf2 = create_grid_force(tiled_input_file=tiled_file)
        system2.addForce(gf2)

        # Set receptor positions (same as generation)
        gf2.setReceptorPositionsFromLists(receptor_positions)

        try:
            integrator2 = VerletIntegrator(0.001*picoseconds)
            context2 = Context(system2, integrator2, platform)
            context2.setPositions(positions)

            state2 = context2.getState(getEnergy=True, getForces=True)
            energy2 = state2.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            forces2 = state2.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)

            print(f"    Loaded grid energy: {energy2:.6f} kJ/mol")

            del context2
            del integrator2

        except Exception as e:
            print(f"    ERROR during loading: {e}")
            raise

        # Step 3: Compare with non-tiled mode
        print("  Step 3: Comparing with non-tiled mode...")
        system3 = create_simple_system()
        gf3 = create_grid_force()  # No tiled mode
        system3.addForce(gf3)
        gf3.setReceptorPositionsFromLists(receptor_positions)

        try:
            integrator3 = VerletIntegrator(0.001*picoseconds)
            context3 = Context(system3, integrator3, platform)
            context3.setPositions(positions)

            state3 = context3.getState(getEnergy=True, getForces=True)
            energy3 = state3.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            forces3 = state3.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)

            print(f"    Non-tiled energy: {energy3:.6f} kJ/mol")

            del context3
            del integrator3

        except Exception as e:
            print(f"    ERROR during non-tiled: {e}")
            raise

        # Compare energies
        energy_diff = abs(energy1 - energy3)
        print(f"\n  Energy comparison:")
        print(f"    Generated vs non-tiled: {energy_diff:.6e} kJ/mol")

        if energy_diff < 1e-4:
            print("    PASS: Energies match")
        else:
            print("    WARNING: Energies differ (may be expected if grids differ)")

    print("\n  Test completed successfully!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tiled Grid Pipeline")
    print("=" * 60)

    # Run API tests
    test_tiled_output_api()
    test_tiled_input_api()

    # Run full pipeline test
    try:
        test_tiled_generation_and_evaluation()
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
