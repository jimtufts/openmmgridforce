from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
gridforceplugin_header_dir = '@GRIDFORCEPLUGIN_HEADER_DIR@'
gridforceplugin_library_dir = '@GRIDFORCEPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']

# Add rpath to find OpenMM libraries
if platform.system() == 'Linux':
    extra_link_args += ['-Wl,-rpath,' + os.path.join(openmm_dir, 'lib')]
    extra_link_args += ['-Wl,-rpath,' + gridforceplugin_library_dir]

# The CUDA implementation library (OpenMMGridForceCUDA) and its SWIG
# bindings are only built when CUDA is available. Link it only when the
# build was configured with CUDA, so the wrapper links cleanly on CUDA-less
# builds (e.g. conda-forge CI). Must stay in sync with the
# -DGRIDFORCE_BUILD_CUDA flag passed to swig when generating the wrapper.
build_cuda = os.environ.get('GRIDFORCE_BUILD_CUDA', '').lower() in ('1', 'on', 'true', 'yes')

libraries = ['OpenMM', 'OpenMMGridForce']
library_dirs = [os.path.join(openmm_dir, 'lib'),
                os.path.join(openmm_dir, 'lib', 'plugins'),
                gridforceplugin_library_dir]
if build_cuda:
    libraries.append('OpenMMGridForceCUDA')
    library_dirs.append(os.path.join(gridforceplugin_library_dir, 'platforms', 'cuda'))
    # The SWIG-generated wrapper guards the CUDA headers with this macro, so
    # the compiler needs it defined to pull in the CudaBATConverter /
    # CudaSmartDartingPool declarations the CUDA wrapper functions reference.
    extra_compile_args.append('-DGRIDFORCE_BUILD_CUDA')

extension = Extension(name='_gridforceplugin',
                     sources=['GridForcePluginWrapper.cpp'],
                     libraries=libraries,
                     include_dirs=[os.path.join(openmm_dir, 'include'), gridforceplugin_header_dir],
                     library_dirs=library_dirs,
                     runtime_library_dirs=library_dirs,
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args
                    )

setup(name='gridforceplugin',
      version='0.7.1',
      py_modules=['gridforceplugin', 'grid_io', 'nc_converter'],
      ext_modules=[extension],
      install_requires=[],
      zip_safe=False
)
