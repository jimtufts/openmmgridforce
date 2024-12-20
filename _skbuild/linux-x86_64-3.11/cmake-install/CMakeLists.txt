cmake_minimum_required(VERSION 3.16)
project(OpenMMGridForcePlugin)

# Add our cmake modules to the path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMM REQUIRED)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Handle platform-specific settings
if(APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    set(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
endif()

# Set installation directory
if(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH "Where to install the plugin" FORCE)
    endif()
endif()

# Source directories setup
set(GRIDFORCE_PLUGIN_SOURCE_SUBDIRS openmmapi)
set(GRIDFORCE_LIBRARY_NAME OpenMMGridForce)
set(SHARED_GRIDFORCE_TARGET ${GRIDFORCE_LIBRARY_NAME})

# API includes
set(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")
set(API_INCLUDE_FILES)
foreach(dir ${API_INCLUDE_DIRS})
    file(GLOB fullpaths ${dir}/*.h)
    set(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
endforeach()

# Collect source files
set(SOURCE_FILES)
set(SOURCE_INCLUDE_FILES)
foreach(subdir ${GRIDFORCE_PLUGIN_SOURCE_SUBDIRS})
    file(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    file(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    set(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})
    set(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
endforeach()

# Create the library
add_library(${SHARED_GRIDFORCE_TARGET} SHARED 
    ${SOURCE_FILES} 
    ${SOURCE_INCLUDE_FILES} 
    ${API_INCLUDE_FILES}
)

set_target_properties(${SHARED_GRIDFORCE_TARGET} PROPERTIES
    COMPILE_FLAGS "-DALGDOCK_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}"
)

target_link_libraries(${SHARED_GRIDFORCE_TARGET} OpenMM::OpenMM)

# Install targets
install(TARGETS ${SHARED_GRIDFORCE_TARGET}
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install headers
install(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
file(GLOB API_ONLY_INCLUDE_FILES_OPENMM "openmmapi/include/internal/*.h")
install(FILES ${API_ONLY_INCLUDE_FILES_OPENMM} DESTINATION include/internal)

# Python wrapper
if(Python_Interpreter_FOUND AND SWIG_FOUND)
    add_subdirectory(python)
endif()
