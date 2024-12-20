# Install script for directory: /home/jim/src/p310/openmmgridforce/platforms/reference

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local/openmm")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/jim/anaconda3/envs/openmm/bin/x86_64-conda-linux-gnu-objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/openmm/lib/plugins" TYPE SHARED_LIBRARY FILES "/home/jim/src/p310/openmmgridforce/openmmgridforce_build/platforms/reference/libOpenMMGridForceReference.so")
  if(EXISTS "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so"
         OLD_RPATH "/home/jim/src/p310/openmmgridforce/openmmgridforce_build:/usr/local/openmm/lib:/usr/local/openmm/lib/plugins:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/jim/anaconda3/envs/openmm/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libOpenMMGridForceReference.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

