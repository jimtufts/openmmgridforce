"""Simple grid I/O utilities for GridForce."""

import numpy as np


def read_netcdf(filename):
    """Read NetCDF grid file and return dict."""
    from netCDF4 import Dataset
    
    with Dataset(filename, 'r') as nc:
        data = {}
        counts = nc.variables['counts'][:]
        data['counts'] = tuple(int(c) for c in (counts[0][:] if len(counts.shape) > 1 else counts))
        
        spacing = nc.variables['spacing'][:]
        data['spacing'] = tuple(float(s) for s in (spacing[0][:] if len(spacing.shape) > 1 else spacing))
        
        if 'origin' in nc.variables:
            origin = nc.variables['origin'][:]
            data['origin'] = tuple(float(o) for o in (origin[0][:] if len(origin.shape) > 1 else origin))
        else:
            data['origin'] = (0.0, 0.0, 0.0)
        
        vals = nc.variables['vals'][:]
        data['vals'] = np.array(vals[0][:] if len(vals.shape) > 1 else vals, dtype=np.float64)
    
    return data


def write_netcdf(filename, counts, spacing, vals, origin=(0.0, 0.0, 0.0)):
    """Write grid to NetCDF file."""
    from netCDF4 import Dataset
    
    with Dataset(filename, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', 1)
        nc.createDimension('data', len(vals))
        nc.createDimension('xyz', 3)
        
        counts_var = nc.createVariable('counts', 'i4', ('time', 'xyz'))
        spacing_var = nc.createVariable('spacing', 'f8', ('time', 'xyz'))
        origin_var = nc.createVariable('origin', 'f8', ('time', 'xyz'))
        vals_var = nc.createVariable('vals', 'f8', ('time', 'data'))
        
        counts_var[0, :] = counts
        spacing_var[0, :] = spacing
        origin_var[0, :] = origin
        vals_var[0, :] = vals


def read_dx(filename):
    """
    Read grid in .dx format.

    Returns
    -------
    dict with keys: 'counts', 'spacing', 'origin', 'vals'
    """
    import gzip

    if filename.endswith('.gz'):
        F = gzip.open(filename, 'rt')
    else:
        F = open(filename, 'r')

    # Read the header
    line = F.readline()
    while line and 'object' not in line:
        line = F.readline()

    if not line:
        raise ValueError("Invalid .dx file format")

    header = {}
    header['counts'] = [int(x) for x in line.split()[-3:]]

    for name in ['origin', 'd0', 'd1', 'd2']:
        line = F.readline()
        header[name] = [float(x) for x in line.split()[-3:]]

    F.readline()  # Skip gridconnections line

    line = F.readline()
    header['npts'] = int(line.split()[-3])

    # Read the data values
    vals = np.ndarray(shape=header['npts'], dtype=float)
    index = 0
    while index < header['npts']:
        line = F.readline()
        if not line or 'object' in line:
            break
        items = [float(item) for item in line.split()]
        vals[index:index + len(items)] = items
        index = index + len(items)

    F.close()

    data = {
        'origin': np.array(header['origin']),
        'spacing': np.array([header['d0'][0], header['d1'][1], header['d2'][2]]),
        'counts': np.array(header['counts']),
        'vals': vals
    }
    return data


def write_dx(filename, counts, spacing, vals, origin=(0.0, 0.0, 0.0), convert_to_angstrom=True):
    """
    Write grid in .dx format for visualization in VMD, PyMOL, Chimera, etc.

    Parameters
    ----------
    filename : str
        Output filename (.dx or .dx.gz)
    counts : tuple of 3 ints
        Number of grid points in each dimension
    spacing : tuple of 3 floats
        Grid spacing in each dimension (in nm for OpenMM)
    vals : array-like
        Flattened grid values
    origin : tuple of 3 floats
        Grid origin coordinates (in nm for OpenMM)
    convert_to_angstrom : bool
        If True, convert origin and spacing from nm to Angstrom (default: True)
        Most molecular visualization tools expect Angstroms
    """
    import gzip

    n_points = counts[0] * counts[1] * counts[2]

    # Convert nm to Angstrom for molecular visualization
    if convert_to_angstrom:
        origin_out = tuple(o * 10.0 for o in origin)
        spacing_out = tuple(s * 10.0 for s in spacing)
    else:
        origin_out = origin
        spacing_out = spacing

    if filename.endswith('.dx.gz'):
        F = gzip.open(filename, 'wt')
    else:
        F = open(filename, 'w')

    # Write header
    F.write("""object 1 class gridpositions counts {0[0]} {0[1]} {0[2]}
origin {1[0]} {1[1]} {1[2]}
delta {2[0]} 0.0 0.0
delta 0.0 {2[1]} 0.0
delta 0.0 0.0 {2[2]}
object 2 class gridconnections counts {0[0]} {0[1]} {0[2]}
object 3 class array type double rank 0 items {3} data follows
""".format(counts, origin_out, spacing_out, n_points))

    # Write data values (3 per line)
    for start_n in range(0, len(vals), 3):
        F.write(' '.join(['%6e' % c for c in vals[start_n:start_n + 3]]) + '\n')

    # Write footer
    F.write('object 4 class field\n')
    F.write('component "positions" value 1\n')
    F.write('component "connections" value 2\n')
    F.write('component "data" value 3\n')

    F.close()


def save_grid_as_dx(grid_force, filename):
    """
    Convenience function to save a GridForce to .dx format.

    Parameters
    ----------
    grid_force : GridForce
        The GridForce object to save
    filename : str
        Output filename (.dx or .dx.gz)
    """
    counts, spacing, vals, scaling = grid_force.getGridParameters()

    # Get origin
    origin = grid_force.getGridOrigin()

    write_dx(filename, counts, spacing, vals, origin)


print("grid_io module loaded successfully")
