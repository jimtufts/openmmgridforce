from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '/home/jim/anaconda3/envs/openmmgridforce'
gridforceplugin_header_dir = '/home/jim/src/p311/openmmgridforce/openmmapi/include'
gridforceplugin_library_dir = '/home/jim/src/p311/openmmgridforce/build'

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

extension = Extension(name='_gridforceplugin',
                     sources=['GridForcePluginWrapper.cpp'],
                     libraries=['OpenMM', 'OpenMMGridForce'],
                     include_dirs=[os.path.join(openmm_dir, 'include'), gridforceplugin_header_dir],
                     library_dirs=[os.path.join(openmm_dir, 'lib'), gridforceplugin_library_dir],
                     runtime_library_dirs=[os.path.join(openmm_dir, 'lib'), gridforceplugin_library_dir],
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args
                    )

setup(name='gridforceplugin',
      version='0.1',
      py_modules=['gridforceplugin'],
      ext_modules=[extension],
      install_requires=[],
      zip_safe=False
)
