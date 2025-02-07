from __future__ import print_function
import pytest
from openmm import app
import openmm as omm
from openmm import unit
from openmm.vec3 import Vec3
import sys
import gridforceplugin
import numpy as np

# Suppress expected warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="numpy.ndarray size changed")

def test_plugin_loading():
    """Verify that the GridForce plugin is properly loaded"""
    try:
        force = gridforceplugin.GridForce()
        # Print available methods for debugging
        print("\nAvailable methods on GridForce:")
        print(dir(force))
        # Verify we can at least create the force object
        assert isinstance(force, gridforceplugin.GridForce)
    except Exception as e:
        assert False, f"Failed to create GridForce: {str(e)}"

def grid_read(FN):
    """
    Reads a grid in a netcdf format
    The multiplier affects the origin and spacing
    """
    if FN is None:
        raise Exception('File is not defined')
    elif FN.endswith('.nc'):
        from netCDF4 import Dataset
        grid_nc = Dataset(FN, 'r')
        data = {}
        for key in list(grid_nc.variables):
            data[key] = np.array(grid_nc.variables[key][:][0][:])
        grid_nc.close()
    else:
        raise Exception('File type not supported')
    return data

def getGridForce(FN, unit_conversion):
    data = grid_read(FN)
    force = gridforceplugin.GridForce()
    nx = int(data['counts'][0])
    ny = int(data['counts'][1])
    nz = int(data['counts'][2])
    force.addGridCounts(nx, ny, nz)
    data['spacing'] *= 0.1
    force.addGridSpacing(data['spacing'][0],
                        data['spacing'][1],
                        data['spacing'][2])
    data['vals'] *= unit_conversion
    for val in data['vals']:
        force.addGridValue(val)
    return force

@pytest.fixture
def simulation_setup():
    """Fixture to set up the simulation environment"""
    prmtop = app.AmberPrmtopFile('../prmtopcrd/ligand.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/ligand.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff,
                                constraints=app.HBonds,
                                implicitSolvent=None)
    return prmtop, inpcrd, system

def test_grid_force_creation(simulation_setup):
    """Test the creation of grid forces"""
    prmtop, _, system = simulation_setup
    
    initial_forces = system.getNumForces()
    unit_conversion = 4.184
    force = getGridForce('../grids/direct_ele.nc', unit_conversion)
    
    # Print available methods for debugging
    print("\nAvailable methods on created force:")
    print(dir(force))
    
    # Just verify we can add the force without error
    for chg in prmtop._prmtop.getCharges():
        force.addScalingFactor(chg)
    system.addForce(force)
    
    assert system.getNumForces() == initial_forces + 1

def test_lj_forces(simulation_setup):
    """Test the creation and addition of Lennard-Jones forces"""
    prmtop, _, system = simulation_setup
    initial_forces = system.getNumForces()
    
    # Test LJr force
    unit_conversion = np.sqrt(4.184)*1.0e6
    ljr_force = getGridForce('../grids/LJr.nc', unit_conversion)
    
    # Just verify we can add scaling factors and the force
    for rVdw, eps in prmtop._prmtop.getNonbondTerms():
        ljr_scale = np.sqrt(eps)*(2.0*rVdw)**6
        ljr_force.addScalingFactor(ljr_scale)
    system.addForce(ljr_force)
    assert system.getNumForces() == initial_forces + 1
    
    # Test LJa force
    unit_conversion = np.sqrt(4.184)*1.0e3
    lja_force = getGridForce('../grids/LJa.nc', unit_conversion)
    
    for rVdw, eps in prmtop._prmtop.getNonbondTerms():
        lja_scale = np.sqrt(eps)*(2.0*rVdw)**3
        lja_force.addScalingFactor(lja_scale)
    system.addForce(lja_force)
    assert system.getNumForces() == initial_forces + 2

def test_full_simulation(simulation_setup):
    """Test the full simulation setup and energy calculation"""
    prmtop, inpcrd, system = simulation_setup
    initial_forces = system.getNumForces()
    
    # Add forces
    forces_to_add = [
        ('../grids/direct_ele.nc', 4.184),
        ('../grids/LJr.nc', np.sqrt(4.184)*1.0e6),
        ('../grids/LJa.nc', np.sqrt(4.184)*1.0e3)
    ]
    
    for grid_file, unit_conv in forces_to_add:
        force = getGridForce(grid_file, unit_conv)
        
        if 'direct_ele' in grid_file:
            for chg in prmtop._prmtop.getCharges():
                force.addScalingFactor(chg)
        else:
            for rVdw, eps in prmtop._prmtop.getNonbondTerms():
                scale = np.sqrt(eps)*(2.0*rVdw)**(6 if 'LJr' in grid_file else 3)
                force.addScalingFactor(scale)
        system.addForce(force)
    
    assert system.getNumForces() == initial_forces + 3
    
    # Set up integrator and platform
    integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                       1.0/unit.picoseconds,
                                       2.0*unit.femtoseconds)
    platform = omm.Platform.getPlatformByName('Reference')
    
    # Create simulation
    simulation = app.Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    
    # Get and verify energy
    state = simulation.context.getState(getForces=True, getEnergy=True, getPositions=False)
    potential_energy = state.getPotentialEnergy()
    
    # Check energy is finite
    energy_in_kj = potential_energy.value_in_unit(unit.kilojoules_per_mole)
    assert np.isfinite(energy_in_kj), "Energy should be finite"

def test_grid_read():
    """Test the grid reading functionality"""
    with pytest.raises(Exception, match='File is not defined'):
        grid_read(None)
    
    with pytest.raises(Exception, match='File type not supported'):
        grid_read('invalid.txt')
    
    # Test successful read if file exists
    try:
        data = grid_read('../grids/direct_ele.nc')
        assert isinstance(data, dict), "Grid data should be a dictionary"
        assert all(key in data for key in ['counts', 'spacing', 'vals']), "Missing expected keys in grid data"
    except FileNotFoundError:
        pytest.skip("Test file not found - skipping positive test case")

if __name__ == '__main__':
    pytest.main([__file__])
