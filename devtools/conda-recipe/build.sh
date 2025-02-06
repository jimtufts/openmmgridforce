#!/bin/bash

# Configure build
CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"

if [[ "$target_platform" == osx-64 ]]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT}"
fi

# OpenMM is installed in the conda environment
CMAKE_FLAGS="$CMAKE_FLAGS -DOPENMM_DIR=$PREFIX"

# Build C++ components
mkdir build
cd build
cmake $CMAKE_FLAGS ..
make -j$CPU_COUNT
make install

# Build and install Python wrappers
cd ../python
swig -python -c++ -o GridForcePluginWrapper.cpp -I$PREFIX/include gridforceplugin.i
$PYTHON setup.py build
$PYTHON setup.py install
