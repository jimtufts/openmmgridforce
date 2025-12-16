"""
NetCDF grid file converter for GridForce plugin.
"""

def nc_to_binary(nc_file, grid_file):
    """Convert NetCDF to binary grid file."""
    import numpy as np
    from netCDF4 import Dataset
    import gridforceplugin

    # Read NetCDF
    with Dataset(nc_file, 'r') as nc:
        counts = tuple(int(c) for c in nc.variables['counts'][0][:])
        spacing = tuple(float(s) * 0.1 for s in nc.variables['spacing'][0][:])  # Å to nm
        vals = np.array(nc.variables['vals'][0][:]) * 4.184  # kcal/mol to kJ/mol

    # Create and populate GridForce
    force = gridforceplugin.GridForce()
    force.addGridCounts(*counts)
    force.addGridSpacing(*spacing)

    for v in vals:
        force.addGridValue(float(v))

    # Save
    force.saveToFile(grid_file)
    print(f"Converted {nc_file} -> {grid_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python nc_converter.py input.nc output.grid")
        sys.exit(1)

    nc_to_binary(sys.argv[1], sys.argv[2])
