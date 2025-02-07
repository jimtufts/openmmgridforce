set(WRAP_FILE GridForcePluginWrapper.cpp)
set(MODULE_NAME gridforceplugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/gridforceplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gridforceplugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(GRIDFORCEPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(GRIDFORCEPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install . 
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Copy test files and data
file(COPY ${CMAKE_SOURCE_DIR}/python/tests
         ${CMAKE_SOURCE_DIR}/python/prmtopcrd
         ${CMAKE_SOURCE_DIR}/python/grids
     DESTINATION ${CMAKE_BINARY_DIR}/python)

add_test(NAME TestPluginLoad
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python/tests)
