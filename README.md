GridForce plugin for OpenMM
===========================

`openmmgridforce` is a plugin for the `OpenMM` toolkit for molecular simulations using a potential energy on grid points. 
Currently, the `openmmgridforce` plugin is available within the "reference implementation" of the `OpenMM` toolkit.

## How to install from source

### Requirements
* conda create -n omm3
* conda activate omm3
* conda install -c conda-forge openmm cmake swig netcdf4

#### on macOS-64
* conda install -c conda-forge clang_osx-64 clangxx_osx-64

#### on linux-64
* conda install -c conda-forge gcc_linux-64 gxx_linux-64


### (Compile and install C++ codes) ###
* pip install .
* Now you can testify the Test*py.

python TestReferenceGridForce.py
