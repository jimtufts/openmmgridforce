"""
Tests for automatic grid generation and file I/O.
"""
from __future__ import print_function
import pytest
from openmm import app
import openmm as omm
from openmm import unit
import gridforceplugin
import numpy as np
from netCDF4 import Dataset
import os
import sys
import tempfile

# Add parent directory to path for grid_io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import grid_io

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture
def receptor_system():
    """Load receptor system for grid generation."""
    prmtop = app.AmberPrmtopFile('../prmtopcrd/receptor.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/receptor.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff,
                                 constraints=None,
                                 implicitSolvent=None)
    return prmtop, inpcrd, system


@pytest.fixture
def reference_grids():
    """Load reference grids from NetCDF files."""
    grids = {}
    for grid_type in ['charge', 'ljr', 'lja']:
        # Map to actual file names
        file_map = {'charge': 'direct_ele', 'ljr': 'LJr', 'lja': 'LJa'}
        nc_file = f'../grids/{file_map[grid_type]}.nc'

        if os.path.exists(nc_file):
            grids[grid_type] = grid_io.read_netcdf(nc_file)
        else:
            pytest.skip(f"Reference grid {nc_file} not found")

    return grids


def test_binary_file_roundtrip():
    """Test saving and loading grid in binary format."""
    # Create a simple test grid
    force = gridforceplugin.GridForce()
    force.addGridCounts(5, 5, 5)
    force.addGridSpacing(0.1, 0.1, 0.1)
    force.setGridOrigin(1.0, 2.0, 3.0)
    force.setInvPower(6.0)
    force.setGridType("charge")

    # Add test values
    test_values = np.arange(125, dtype=np.float64) * 0.5
    for v in test_values:
        force.addGridValue(float(v))

    # Save to binary file
    with tempfile.NamedTemporaryFile(suffix='.grid', delete=False) as tmp:
        tmp_file = tmp.name

    try:
        force.saveToFile(tmp_file)
        assert os.path.exists(tmp_file), "Binary file was not created"

        # Load back
        force2 = gridforceplugin.GridForce()
        force2.loadFromFile(tmp_file)

        # Verify parameters
        counts, spacing, vals, sf = force2.getGridParameters()

        assert list(counts) == [5, 5, 5], f"Counts mismatch: {counts}"
        assert np.allclose(spacing, [0.1, 0.1, 0.1]), f"Spacing mismatch: {spacing}"
        assert len(vals) == 125, f"Value count mismatch: {len(vals)}"
        assert np.allclose(vals, test_values, rtol=1e-10), "Values don't match"

        # Verify grid origin
        ox, oy, oz = force2.getGridOrigin()
        assert np.allclose([ox, oy, oz], [1.0, 2.0, 3.0]), "Origin mismatch"

        # Verify grid type and inv_power
        assert force2.getGridType() == "charge", "Grid type mismatch"
        assert force2.getInvPower() == 6.0, "InvPower mismatch"

        print(f"[PASS] Binary round-trip test")

    finally:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)


def test_netcdf_roundtrip():
    """Test writing and reading NetCDF format."""
    # Create test data
    counts = (10, 10, 10)
    spacing = (0.15, 0.15, 0.15)
    origin = (0.5, 1.0, 1.5)
    vals = np.random.randn(1000) * 10.0

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_file = tmp.name

    try:
        # Write NetCDF
        grid_io.write_netcdf(tmp_file, counts, spacing, vals, origin)
        assert os.path.exists(tmp_file), "NetCDF file was not created"

        # Read back
        data = grid_io.read_netcdf(tmp_file)

        assert data['counts'] == counts, "Counts mismatch"
        assert np.allclose(data['spacing'], spacing), "Spacing mismatch"
        assert np.allclose(data['origin'], origin), "Origin mismatch"
        assert np.allclose(data['vals'], vals), "Values mismatch"

        print(f"[PASS] NetCDF round-trip test")

    finally:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)


def test_auto_grid_generation(receptor_system):
    """Test automatic grid generation from receptor."""
    prmtop, inpcrd, system = receptor_system

    # Get receptor positions and convert to nm (strip units)
    receptor_positions = inpcrd.positions.value_in_unit(unit.nanometer)
    receptor_positions = [(p[0], p[1], p[2]) for p in receptor_positions]

    # Get all receptor atoms
    receptor_atoms = list(range(prmtop.topology.getNumAtoms()))

    # Define grid parameters (smaller grid for testing)
    nx, ny, nz = 20, 20, 20
    dx, dy, dz = 0.25 * 0.1, 0.25 * 0.1, 0.25 * 0.1  # Convert Å to nm

    # Get NonbondedForce to access particle parameters
    nonbonded_force = None
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), omm.NonbondedForce):
            nonbonded_force = system.getForce(i)
            break

    # Test charge grid generation
    for grid_type in ['charge', 'ljr', 'lja']:
        force = gridforceplugin.GridForce()
        force.addGridCounts(nx, ny, nz)
        force.addGridSpacing(dx, dy, dz)
        force.setAutoGenerateGrid(True)
        force.setGridType(grid_type)
        force.setReceptorAtoms(receptor_atoms)
        force.setReceptorPositionsFromLists(receptor_positions)

        # Enable auto-calculation of scaling factors
        force.setAutoCalculateScalingFactors(True)
        if grid_type == 'charge':
            force.setScalingProperty("charge")
        elif grid_type == 'ljr':
            force.setScalingProperty("ljr")
        else:  # lja
            force.setScalingProperty("lja")

        # Add to system to trigger generation
        system.addForce(force)

        # Create context to initialize (this triggers grid generation)
        integrator = omm.VerletIntegrator(1.0*unit.femtoseconds)
        platform = omm.Platform.getPlatformByName('Reference')

        try:
            context = omm.Context(system, integrator, platform)
            context.setPositions(inpcrd.positions)

            # Get energy to verify the force is working
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            # Verify energy is finite (grid was generated and evaluated)
            assert np.isfinite(energy), f"Grid {grid_type}: Energy is not finite"

            print(f"[PASS] Auto-generation test for {grid_type} grid (energy={energy:.2f} kJ/mol)")

        finally:
            # Clean up
            if 'context' in locals():
                del context
            del integrator
            system.removeForce(system.getNumForces() - 1)


def test_binary_vs_netcdf_equivalence():
    """Test that binary and NetCDF formats store the same data."""
    # Create test grid
    counts = (8, 8, 8)
    spacing = (0.2, 0.2, 0.2)
    origin = (0.0, 0.0, 0.0)
    vals = np.random.randn(512) * 5.0

    # Save as NetCDF
    nc_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
    grid_io.write_netcdf(nc_file, counts, spacing, vals, origin)

    # Convert to binary via GridForce
    force = gridforceplugin.GridForce()
    nc_data = grid_io.read_netcdf(nc_file)
    force.addGridCounts(*nc_data['counts'])
    force.addGridSpacing(*nc_data['spacing'])
    for v in nc_data['vals']:
        force.addGridValue(float(v))

    binary_file = tempfile.NamedTemporaryFile(suffix='.grid', delete=False).name
    force.saveToFile(binary_file)

    try:
        # Load binary back
        force2 = gridforceplugin.GridForce()
        force2.loadFromFile(binary_file)
        bin_counts, bin_spacing, bin_vals, _ = force2.getGridParameters()

        # Compare
        assert list(bin_counts) == list(counts), "Counts differ between formats"
        assert np.allclose(bin_spacing, spacing), "Spacing differs between formats"
        assert np.allclose(bin_vals, vals, rtol=1e-10), "Values differ between formats"

        print(f"[PASS] Binary vs NetCDF equivalence test")

    finally:
        os.unlink(nc_file)
        os.unlink(binary_file)


def test_grid_generation_with_custom_origin():
    """Test grid generation with non-zero origin."""
    # Create simple system
    from openmm import app
    system = omm.System()
    system.addParticle(1.0)

    # Create NonbondedForce with one charged particle
    nb_force = omm.NonbondedForce()
    nb_force.addParticle(1.0, 0.3, 0.1)  # charge, sigma, epsilon
    system.addForce(nb_force)

    # Create grid with custom origin
    force = gridforceplugin.GridForce()
    force.addGridCounts(5, 5, 5)
    force.addGridSpacing(0.1, 0.1, 0.1)
    force.setGridOrigin(1.0, 2.0, 3.0)
    force.setAutoGenerateGrid(True)
    force.setGridType("charge")
    force.setReceptorAtoms([0])
    force.setReceptorPositionsFromLists([(1.5, 2.5, 3.5)])  # nm

    # Enable auto-scaling
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("charge")

    system.addForce(force)

    # Initialize to trigger generation
    integrator = omm.VerletIntegrator(1.0*unit.femtoseconds)
    context = omm.Context(system, integrator, omm.Platform.getPlatformByName('Reference'))
    context.setPositions([(1.5, 2.5, 3.5)])

    # Verify origin was preserved
    ox, oy, oz = force.getGridOrigin()
    assert np.allclose([ox, oy, oz], [1.0, 2.0, 3.0]), "Custom origin not preserved"

    # Verify grid is working by getting energy
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert np.isfinite(energy), "Energy should be finite"

    print(f"[PASS] Custom origin test (energy={energy:.2f} kJ/mol)")

    del context


def test_nc_converter():
    """Test the nc_converter utility."""
    import nc_converter

    # Create a test NetCDF file
    nc_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
    grid_file = tempfile.NamedTemporaryFile(suffix='.grid', delete=False).name

    # Write test NetCDF (in Angstroms and kcal/mol as AlGDock uses)
    counts = (6, 6, 6)
    spacing_angstrom = (1.0, 1.0, 1.0)  # Å
    vals_kcal = np.arange(216, dtype=np.float64)  # kcal/mol

    grid_io.write_netcdf(nc_file, counts, spacing_angstrom, vals_kcal)

    try:
        # Convert
        nc_converter.nc_to_binary(nc_file, grid_file)
        assert os.path.exists(grid_file), "Converter didn't create output file"

        # Load and verify conversion
        force = gridforceplugin.GridForce()
        force.loadFromFile(grid_file)
        out_counts, out_spacing, out_vals, _ = force.getGridParameters()

        # Verify counts
        assert list(out_counts) == list(counts), "Counts not preserved"

        # Verify spacing was converted Å -> nm
        expected_spacing = np.array(spacing_angstrom) * 0.1
        assert np.allclose(out_spacing, expected_spacing), "Spacing conversion incorrect"

        # Verify values were converted kcal/mol -> kJ/mol
        expected_vals = vals_kcal * 4.184
        assert np.allclose(out_vals, expected_vals), "Value conversion incorrect"

        print(f"[PASS] NC converter test")

    finally:
        os.unlink(nc_file)
        if os.path.exists(grid_file):
            os.unlink(grid_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
