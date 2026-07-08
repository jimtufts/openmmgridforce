# openmm-gridforce

An [OpenMM](https://openmm.org) plugin for grid-based potentials. The plugin
adds a small set of `Force` and integrator classes that evaluate energies and
forces on a precomputed Cartesian grid of receptor potentials, which is the
core building block for grid-based docking, free-energy calculations, and
implicit-solvent models built on top of OpenMM.

## What's in the plugin

Force / integrator classes (all OpenMM `Force`-compatible, all platforms):

| Class | Purpose |
|---|---|
| `GridForce` | Trilinear / tricubic / triquintic interpolation of a precomputed potential grid (charge, LJa, LJr). |
| `IsolatedNonbondedForce` | NB energy of a ligand against a frozen receptor, with per-group isolation. |
| `IsolatedBondedForce`, `IsolatedSiteForce` | Companion bonded / site terms with the same isolation semantics. |
| `IsolatedGBSAForce`, `GBSAGridForce` | OBC2 generalized-Born + surface-area, including a receptor-pairwise mode and analytical Hessians. |
| `MultiGroupHMCIntegrator`, `MultiGroupNUTSIntegrator` | HMC / NUTS integrators that move each particle group independently. |
| `NewtonMinimizer` | Trust-region Newton minimizer that consumes analytical Hessians from `IsolatedGBSAForce`. |
| `CudaSmartDartingPool`, `CudaBATConverter` | GPU helpers for smart-darting MC and BAT coordinate transforms. |

Backends:

| Platform | Status |
|---|---|
| Reference | Full support, all force / integrator / interpolation paths. |
| CPU       | Multithreaded; full support for all interpolation methods and grid types. |
| CUDA      | Highest performance; analytical Hessians require sm_60+ for the double-precision atomic path. |
| OpenCL    | Supported for the runtime force kernels. |

Interpolation methods exposed via `setInterpolationMethod(int)`:

```
INTERP_TRILINEAR          = 0
INTERP_TRICUBIC_BSPLINE   = 1   # tricubic B-spline (prefilter at load)
INTERP_TRICUBIC_HERMITE   = 2   # requires stored derivatives
INTERP_TRIQUINTIC_HERMITE = 3   # requires stored derivatives (27 per corner)
INTERP_TRIQUINTIC_BSPLINE = 4   # quintic B-spline (prefilter at load)
```

The Hermite paths consume grids generated with `setComputeDerivatives(True)`;
the B-spline paths use values only and apply a prefilter at load time.

## Install

### From conda-forge (recommended)

```bash
conda install -c conda-forge openmm-gridforce
```

This pulls a prebuilt package with the Reference, CPU, and OpenCL backends.
The CUDA backend is omitted from the conda-forge build (it's gated behind
`GRIDFORCE_BUILD_CUDA_LIB` in CMake); CUDA-using workloads should build from
source on a node with the CUDA toolkit installed.

### From source

Dependencies (versions used in current development builds):

| Tool | Tested |
|---|---|
| OpenMM | 8.5 (any 8.x) |
| Python | 3.12 (any 3.10+) |
| CMake | 4.2 (any 3.x or 4.x) |
| SWIG | 4.3 (any 4.x) |
| GCC / G++ | 11.4 (any C++11-capable) |
| CUDA (optional) | 12.x |
| OpenCL (optional) | any |

The fastest way to get all build dependencies is from conda-forge:

```bash
conda create -n omm-gf -c conda-forge \
  python openmm swig cmake make gcc_linux-64 gxx_linux-64 pip
conda activate omm-gf
```

CUDA, OpenCL, and Python wrappers are auto-detected and built if the
corresponding toolchain is present.

```bash
# 1. Get the source
git clone https://github.com/jimtufts/openmmgridforce.git
cd openmmgridforce

# 2. Configure
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DCMAKE_BUILD_TYPE=Release

# 3. Build and install the C++ libraries
cmake --build . --target install -j

# 4. Build and install the Python module (gridforceplugin)
cd python
make PythonInstall
```

The `PythonInstall` target re-runs `cmake --install` to refresh the C++
libraries, then `pip install --no-build-isolation .` to install the SWIG
wrapper into the active environment.

#### Building on a compute node without system headers

If `/usr/bin/c++` exists but has no `libstdc++-devel` headers (typical on
shared-FS compute nodes), use the conda toolchain instead. Inside a
SLURM/PBS job:

```bash
conda activate <env>
CONDA_CXX=$(which x86_64-conda-linux-gnu-c++)
CONDA_CC=$(which x86_64-conda-linux-gnu-cc)
export CUDAHOSTCXX="$CONDA_CXX"

cmake .. \
  -DCMAKE_C_COMPILER="$CONDA_CC" \
  -DCMAKE_CXX_COMPILER="$CONDA_CXX" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-Xcompiler --sysroot=$CONDA_BUILD_SYSROOT" \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --target install -j$(nproc)
(cd python && make PythonInstall)
```

#### Build options

```
-DGRIDFORCE_BUILD_CUDA_LIB=ON|OFF        # default ON if CUDA found
-DGRIDFORCE_BUILD_OPENCL_LIB=ON|OFF      # default ON if OpenCL found
-DGRIDFORCE_BUILD_PYTHON_WRAPPERS=ON|OFF # default ON if Python+SWIG found
-DOPENMM_DIR=<path>                      # OpenMM install prefix
-DCMAKE_INSTALL_PREFIX=<path>            # where to install (set to conda env)
```

## Verifying the install

```python
import openmm as mm
import gridforceplugin as gfp

# Plugin loaded on all available backends
for i in range(mm.Platform.getNumPlatforms()):
    print(mm.Platform.getPlatform(i).getName())
# -> Reference, CPU, CUDA (if built), OpenCL (if built)

# Interpolation method constants exposed
print(gfp.INTERP_TRILINEAR, gfp.INTERP_TRICUBIC_BSPLINE,
      gfp.INTERP_TRIQUINTIC_HERMITE)
```

## Tests

The CMake build copies `python/tests/`, `python/prmtopcrd/`, and
`python/grids/` into `build/python/`. Run the smoke tests with:

```bash
cd build/python
python -m pytest tests/
```

Individual scripts (e.g. `tests/test_triquintic.py`,
`tests/test_bspline_tiled.py`, `tests/benchmark_hessian_analysis.py`) run
standalone and exercise specific interpolation paths against analytical
references.

## License

MIT (see `LICENSE`).
