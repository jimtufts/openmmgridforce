"""
Test auto-calculation of scaling factors feature
"""
from __future__ import print_function
import pytest
from openmm import app
import openmm as omm
from openmm import unit
import gridforceplugin
import numpy as np


def test_auto_calculate_charge_scaling():
    """Test automatic calculation of scaling factors from charges"""
    # Create a simple system
    prmtop = app.AmberPrmtopFile('../prmtopcrd/ligand.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/ligand.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)

    # Create a simple grid (just for testing)
    force = gridforceplugin.GridForce()
    force.addGridCounts(10, 10, 10)
    force.addGridSpacing(0.1, 0.1, 0.1)
    for i in range(10*10*10):
        force.addGridValue(1.0)  # Simple uniform grid

    # Enable auto-calculation for charges
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("charge")

    # Verify settings
    assert force.getAutoCalculateScalingFactors() == True
    assert force.getScalingProperty() == "charge"

    # Add force to system
    system.addForce(force)

    # Create context - this will trigger auto-calculation
    integrator = omm.VerletIntegrator(0.001*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)

    # Success if we got here without errors
    print("Auto-calculation of charge scaling factors succeeded")


def test_auto_calculate_ljr_scaling():
    """Test automatic calculation of LJ repulsive scaling factors"""
    prmtop = app.AmberPrmtopFile('../prmtopcrd/ligand.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/ligand.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)

    # Create grid force
    force = gridforceplugin.GridForce()
    force.addGridCounts(10, 10, 10)
    force.addGridSpacing(0.1, 0.1, 0.1)
    for i in range(10*10*10):
        force.addGridValue(1.0)

    # Enable auto-calculation for LJ repulsive
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("ljr")

    assert force.getScalingProperty() == "ljr"

    system.addForce(force)

    integrator = omm.VerletIntegrator(0.001*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)

    print("Auto-calculation of LJ repulsive scaling factors succeeded")


def test_auto_calculate_lja_scaling():
    """Test automatic calculation of LJ attractive scaling factors"""
    prmtop = app.AmberPrmtopFile('../prmtopcrd/ligand.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/ligand.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)

    # Create grid force
    force = gridforceplugin.GridForce()
    force.addGridCounts(10, 10, 10)
    force.addGridSpacing(0.1, 0.1, 0.1)
    for i in range(10*10*10):
        force.addGridValue(1.0)

    # Enable auto-calculation for LJ attractive
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("lja")

    assert force.getScalingProperty() == "lja"

    system.addForce(force)

    integrator = omm.VerletIntegrator(0.001*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)

    print("Auto-calculation of LJ attractive scaling factors succeeded")


def test_manual_scaling_still_works():
    """Test that manual scaling factor addition still works (backward compatibility)"""
    prmtop = app.AmberPrmtopFile('../prmtopcrd/ligand.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/ligand.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)

    # Create grid force
    force = gridforceplugin.GridForce()
    force.addGridCounts(10, 10, 10)
    force.addGridSpacing(0.1, 0.1, 0.1)
    for i in range(10*10*10):
        force.addGridValue(1.0)

    # Manually add scaling factors (old way)
    for chg in prmtop._prmtop.getCharges():
        force.addScalingFactor(chg)

    # Auto-calculation should be disabled by default
    assert force.getAutoCalculateScalingFactors() == False

    system.addForce(force)

    integrator = omm.VerletIntegrator(0.001*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)

    print("Manual scaling factor addition still works")


def test_invalid_scaling_property():
    """Test that invalid scaling property raises an error during Context creation"""
    prmtop = app.AmberPrmtopFile('../prmtopcrd/ligand.prmtop')
    inpcrd = app.AmberInpcrdFile('../prmtopcrd/ligand.trans.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)

    # Create grid force with invalid property
    force = gridforceplugin.GridForce()
    force.addGridCounts(10, 10, 10)
    force.addGridSpacing(0.1, 0.1, 0.1)
    for i in range(10*10*10):
        force.addGridValue(1.0)

    # Set invalid property (this should not raise an error yet)
    force.setAutoCalculateScalingFactors(True)
    force.setScalingProperty("invalid_property")
    system.addForce(force)

    # Error should be raised when creating Context (during kernel initialization)
    integrator = omm.VerletIntegrator(0.001*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('Reference')

    with pytest.raises(Exception) as exc_info:
        context = omm.Context(system, integrator, platform)

    # Verify the error message contains our validation message
    assert "Invalid scaling property" in str(exc_info.value)
    print("Invalid scaling property correctly detected during Context creation")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
