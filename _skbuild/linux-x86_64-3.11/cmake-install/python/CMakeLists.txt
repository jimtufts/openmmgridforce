# Set up SWIG wrapper
set(MODULE_NAME gridforceplugin)

# Configure SWIG
include_directories(${OpenMM_INCLUDE_DIR})
set_source_files_properties(gridforceplugin.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(gridforceplugin.i PROPERTIES SWIG_FLAGS "-threads")

# Add SWIG module
swig_add_library(${MODULE_NAME}
    TYPE MODULE
    LANGUAGE python
    SOURCES gridforceplugin.i
)

# Handle Python module properties
if (APPLE)
    set_target_properties(${MODULE_NAME} PROPERTIES 
        PREFIX ""
        OUTPUT_NAME "_gridforceplugin"
        SUFFIX ".so"
        INSTALL_RPATH "@loader_path/../lib"
    )
else()
    set_target_properties(${MODULE_NAME} PROPERTIES 
        PREFIX ""
        OUTPUT_NAME "_gridforceplugin"
        SUFFIX ".so"
    )
endif()

# Link against libraries
swig_link_libraries(${MODULE_NAME}
    ${SHARED_GRIDFORCE_TARGET}
    OpenMM::OpenMM
    Python::Python
)

# Copy and rename the Python wrapper file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/gridforceplugin.py"
    "${CMAKE_CURRENT_BINARY_DIR}/gridforceplugin.py"
    COPYONLY
)

# Install both the SWIG module and Python wrapper
install(TARGETS ${MODULE_NAME}
    LIBRARY DESTINATION openmmgridforce
    RUNTIME DESTINATION openmmgridforce
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gridforceplugin.py"
    DESTINATION openmmgridforce
)
