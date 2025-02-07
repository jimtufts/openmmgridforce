from __future__ import print_function
import pytest
from openmm import app
import openmm as omm
from openmm import unit
from openmm.vec3 import Vec3
import sys
import gridforceplugin
import numpy as np

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
    # data['spacing] length unit is A
    # A -> nm: 0.1
    data['spacing'] *= 0.1
    force.addGridSpacing(data['spacing'][0],
                        data['spacing'][1],
                        data['spacing'][2])
    # vals:
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
    
    # Test direct electrostatic force
    unit_conversion = 4.184  # kcal/mol/e --> kJ/mol/e
    force = getGridForce('../grids/direct_ele.nc', unit_conversion)
    for chg in prmtop._prmtop.getCharges():
        force.addScalingFactor(chg)
    system.addForce(force)
    
    assert system.getNumForces() > 0
    assert isinstance(system.getForce(system.getNumForces()-1), gridforceplugin.GridForce)

def test_lj_forces(simulation_setup):
    """Test the creation and addition of Lennard-Jones forces"""
    prmtop, _, system = simulation_setup
    
    # Test LJr force
    unit_conversion = np.sqrt(4.184)*1.0e6  # sqrt(kcal/mol)/A^6 --> sqrt(kJ/mol)/nm^6
    force = getGridForce('../grids/LJr.nc', unit_conversion)
    for rVdw, eps in prmtop._prmtop.getNonbondTerms():
        ljr_scale = np.sqrt(eps)*(2.0*rVdw)**6
        force.addScalingFactor(ljr_scale)
    system.addForce(force)
    
    # Test LJa force
    unit_conversion = np.sqrt(4.184)*1.0e3  # sqrt(kcal/mol)/A^3 --> sqrt(kJ/mol)/nm^3
    force = getGridForce('../grids/LJa.nc', unit_conversion)
    for rVdw, eps in prmtop._prmtop.getNonbondTerms():
        lja_scale = np.sqrt(eps)*(2.0*rVdw)**3
        force.addScalingFactor(lja_scale)
    system.addForce(force)
    
    assert system.getNumForces() > 0

def test_full_simulation(simulation_setup):
    """Test the full simulation setup and energy calculation"""
    prmtop, inpcrd, system = simulation_setup
    
    # Add all forces
    # Direct electrostatic
    unit_conversion = 4.184
    force = getGridForce('../grids/direct_ele.nc', unit_conversion)
    for chg in prmtop._prmtop.getCharges():
        force.addScalingFactor(chg)
    system.addForce(force)
    
    # LJr
    unit_conversion = np.sqrt(4.184)*1.0e6
    force = getGridForce('../grids/LJr.nc', unit_conversion)
    for rVdw, eps in prmtop._prmtop.getNonbondTerms():
        ljr_scale = np.sqrt(eps)*(2.0*rVdw)**6
        force.addScalingFactor(ljr_scale)
    system.addForce(force)
    
    # LJa
    unit_conversion = np.sqrt(4.184)*1.0e3
    force = getGridForce('../grids/LJa.nc', unit_conversion)
    for rVdw, eps in prmtop._prmtop.getNonbondTerms():
        lja_scale = np.sqrt(eps)*(2.0*rVdw)**3
        force.addScalingFactor(lja_scale)
    system.addForce(force)
    
    # Set up integrator and platform
    integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                       1.0/unit.picoseconds,
                                       2.0*unit.femtoseconds)
    platform = omm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(
        prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    
    # Get and test energy
    state = simulation.context.getState(
        getForces=True, getEnergy=True, getPositions=False)
    potential_energy = state.getPotentialEnergy()
    
    # Assertions
    assert potential_energy is not None
    assert potential_energy.value_in_unit(unit.kilojoules_per_mole) > -np.inf
    assert potential_energy.value_in_unit(unit.kilojoules_per_mole) < np.inf

def test_grid_read():
    """Test the grid reading functionality"""
    with pytest.raises(Exception):
        grid_read(None)
    
    with pytest.raises(Exception):
        grid_read('invalid.txt')

if __name__ == '__main__':
    pytest.main([__file__])
