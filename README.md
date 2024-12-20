# OpenMM GridForce Plugin

[![License](https://img.shields.io/badge/License-BSD_2_Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenMM](https://img.shields.io/badge/openmm-7.0+-blue.svg)](http://openmm.org)

A plugin for OpenMM that implements grid-based forces.

## Prerequisites

Before installing the plugin, ensure you have the following:

- Python 3.7 or later
- OpenMM 7.0 or later
- C++ compiler:
  - Linux: GCC
  - macOS: Clang
- SWIG 3.0.12 (specifically this version for OpenMM compatibility)
- CMake 3.16 or later
- Ninja build system
- scikit-build (for pip installation)

## Installation

### 1. Create a Conda Environment

First, create and activate a new conda environment:

```bash
conda create -n gridforce python
conda activate gridforce
```

### 2. Install Dependencies

Install OpenMM and build dependencies:

```bash
# Core dependencies
conda install -c conda-forge openmm cmake ninja
conda install -c conda-forge "swig=3.0.12"

# Build dependencies
pip install scikit-build

# Additional dependencies for testing
conda install -c conda-forge netcdf4
```

### 3. Install Platform-Specific Compiler

On Linux:
```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64
```

On macOS:
```bash
conda install -c conda-forge clang_osx-64 clangxx_osx-64
```

### 4. Install the Plugin

```bash
pip install .
```

## Verification

To verify the installation:

```python
import openmm
import openmmgridforce
print("Installation successful!")
```

Run the tests:
```bash
python python/TestReferenceGridForce.py
```

## Troubleshooting

If you encounter issues during installation:

1. Verify you're in the correct conda environment:
   ```bash
   conda activate gridforce
   ```

2. Check that required packages are installed:
   ```bash
   conda list openmm  # Should show OpenMM 7.0 or later
   swig -version      # Should show version 3.0.12
   cmake --version    # Should show version 3.16 or later
   ```

3. Verify compiler installation:
   - Linux: `gcc --version`
   - macOS: `clang --version`

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

