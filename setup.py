import os
import sys
import platform
from skbuild import setup

def setup_build_environment():
    """Setup build environment to include conda packages"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        raise RuntimeError("This package must be built within a conda environment")
    
    # Add conda environment's site-packages to PYTHONPATH
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(conda_prefix, "lib", f"python{python_version}", "site-packages")
    
    # Print debug info
    print(f"Conda prefix: {conda_prefix}")
    print(f"Python version: {python_version}")
    print(f"Site packages: {site_packages}")
    
    if not os.path.exists(site_packages):
        raise RuntimeError(f"Site packages directory not found: {site_packages}")
    
    os.environ['PYTHONPATH'] = site_packages
    if 'PYTHONPATH' in os.environ:
        print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

def get_cmake_args():
    """Get CMake arguments"""
    cmake_args = []
    
    # Add our cmake modules directory
    cmake_dir = os.path.join(os.path.dirname(__file__), 'cmake')
    cmake_args.append(f'-DCMAKE_MODULE_PATH:PATH={cmake_dir}')
    
    # Add conda environment paths
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        cmake_args.extend([
            f'-DCMAKE_PREFIX_PATH={conda_prefix}',
            f'-DPython_EXECUTABLE={os.path.join(conda_prefix, "bin", "python")}',
            f'-DOpenMM_DIR={conda_prefix}',
        ])
    
    # Platform-specific settings
    if platform.system() == "Darwin":
        cmake_args.extend([
            "-DCMAKE_CXX_FLAGS=-stdlib=libc++ -mmacosx-version-min=10.7",
            "-DCMAKE_LINK_FLAGS=-stdlib=libc++ -mmacosx-version-min=10.7",
        ])
    
    return cmake_args

# Setup the build environment
setup_build_environment()

setup(
    name="openmmgridforce",
    version="0.1.0",
    author="Your Name",
    description="OpenMM GridForce Plugin",
    packages=['openmmgridforce'],
    package_dir={'openmmgridforce': 'openmmgridforce'},
    cmake_install_dir='openmmgridforce',
    cmake_args=get_cmake_args(),
    cmake_source_dir='.',
    include_package_data=True,
    zip_safe=False,
)
