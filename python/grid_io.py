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


print("grid_io module loaded successfully")
