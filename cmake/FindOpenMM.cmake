# Find OpenMM using Python package location
#
# Defines:
# OpenMM_FOUND
# OpenMM_INCLUDE_DIR
# OpenMM_LIBRARY

# Find OpenMM using Python package location
message(STATUS "====== Finding OpenMM ======")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# Get Python site-packages from CONDA_PREFIX
if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "This package must be built within a conda environment")
endif()

# Get Python path info
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys, os; print('\\n'.join(sys.path))"
    OUTPUT_VARIABLE PYTHON_PATH_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python path info:\n${PYTHON_PATH_INFO}")

# Explicitly add conda site-packages to Python path and try to import OpenMM
execute_process(
    COMMAND "${Python_EXECUTABLE}" 
    -c "import sys, os; conda_site_packages=os.path.join(os.environ['CONDA_PREFIX'], 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'); sys.path.insert(0, conda_site_packages); import openmm; print(os.path.dirname(openmm.__file__))"
    OUTPUT_VARIABLE OPENMM_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE OPENMM_PYTHON_RESULT
    ERROR_VARIABLE OPENMM_PYTHON_ERROR
)

if(NOT OPENMM_PYTHON_RESULT EQUAL 0)
    message(STATUS "Error trying to import openmm: ${OPENMM_PYTHON_ERROR}")
    message(FATAL_ERROR "Could not find OpenMM Python package. Please install it first using: conda install openmm")
endif()

message(STATUS "Found OpenMM Python package at: ${OPENMM_PYTHON_DIR}")

# Find include directory
set(OpenMM_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
if(NOT EXISTS "${OpenMM_INCLUDE_DIR}/OpenMM.h")
    message(FATAL_ERROR "Could not find OpenMM.h in ${OpenMM_INCLUDE_DIR}")
endif()
message(STATUS "Found OpenMM headers: ${OpenMM_INCLUDE_DIR}")

# Find library
set(OpenMM_LIBRARY "$ENV{CONDA_PREFIX}/lib/libOpenMM${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(NOT EXISTS "${OpenMM_LIBRARY}")
    message(FATAL_ERROR "Could not find OpenMM library at ${OpenMM_LIBRARY}")
endif()
message(STATUS "Found OpenMM library: ${OpenMM_LIBRARY}")

# Set up OpenMM target
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMM
    REQUIRED_VARS 
        OpenMM_LIBRARY 
        OpenMM_INCLUDE_DIR
)

if(OpenMM_FOUND AND NOT TARGET OpenMM::OpenMM)
    add_library(OpenMM::OpenMM UNKNOWN IMPORTED)
    set_target_properties(OpenMM::OpenMM PROPERTIES
        IMPORTED_LOCATION "${OpenMM_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenMM_INCLUDE_DIR}"
    )
endif()

message(STATUS "====== End Finding OpenMM ======")
