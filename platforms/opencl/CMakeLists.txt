#---------------------------------------------------
# OpenMM GridForce Plugin OpenCL Platform
#----------------------------------------------------

SET(OPENMM_GRIDFORCE_OPENCL_LIBRARY_NAME OpenMMGridForceOpenCL)
SET(SHARED_TARGET ${OPENMM_GRIDFORCE_OPENCL_LIBRARY_NAME})

# These are all the places to search for header files
SET(API_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)

SET(OPENMM_SOURCE_SUBDIRS . ../common)
FOREACH(subdir ${OPENMM_SOURCE_SUBDIRS})
    FILE(GLOB_RECURSE src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.c)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    # Exclude the CommonGridForceKernelSources.cpp stub from source directory
    LIST(FILTER src_files EXCLUDE REGEX "CommonGridForceKernelSources\\.cpp$")
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
ENDFOREACH(subdir)

# Add the GENERATED kernel source file from build directory
SET(COMMON_KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/../common/src/CommonGridForceKernelSources.cpp)
SET(SOURCE_FILES ${SOURCE_FILES} ${COMMON_KERNELS_CPP})

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/opencl/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/common/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/common/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/../common/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/../common/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/openmmapi/include)

# Create the library
SET_SOURCE_FILES_PROPERTIES(${COMMON_KERNELS_CPP} PROPERTIES GENERATED TRUE)
ADD_LIBRARY(${SHARED_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
ADD_DEPENDENCIES(${SHARED_TARGET} CommonKernels)

TARGET_LINK_LIBRARIES(${SHARED_TARGET} OpenMM OpenMMOpenCL OpenMMGridForce)

SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_FLAGS "-DOPENMM_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")

INSTALL(TARGETS ${SHARED_TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/plugins)

# Ensure that links to the main OpenCL library will be resolved.
IF (APPLE)
    SET(OPENCL_LIBRARY libOpenMMOpenCL.dylib)
    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${OPENCL_LIBRARY} @loader_path/${OPENCL_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
ENDIF (APPLE)
