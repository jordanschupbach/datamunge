cmake_minimum_required(VERSION 3.16)

project(
  datamungeoctave
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-octave, scripts/prebuild_bindings.py octave):
# src/datamunge_octave_wrap*.cpp (the oct-file) and the package beside this file
# (inst/+datamunge: one classdef or function .m file per class, enum, function and constant),
# generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_OCTAVE_PACKAGE "${CMAKE_CURRENT_LIST_DIR}/inst/+datamunge")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_octave_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${DATAMUNGE_OCTAVE_PACKAGE}/Contents.m")
  message(
    FATAL_ERROR
      "No generated Octave binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-octave`"
  )
endif()

get_target_property(
  DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES
)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(
    DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH
  )
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# Octave's headers and libraries, as mkoctfile reports them (the interpreter must match).
find_program(MKOCTFILE_EXECUTABLE mkoctfile REQUIRED)
foreach(setting OCTINCLUDEDIR OCTLIBDIR)
  execute_process(
    COMMAND "${MKOCTFILE_EXECUTABLE}" -p ${setting}
    OUTPUT_VARIABLE OCTAVE_${setting}
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
  )
endforeach()
get_filename_component(OCTAVE_INCLUDE_PARENT "${OCTAVE_OCTINCLUDEDIR}" DIRECTORY)
find_library(
  OCTAVE_OCTINTERP_LIBRARY
  NAMES octinterp
  PATHS "${OCTAVE_OCTLIBDIR}" REQUIRED
  NO_DEFAULT_PATH
)
find_library(
  OCTAVE_OCTAVE_LIBRARY
  NAMES octave
  PATHS "${OCTAVE_OCTLIBDIR}" REQUIRED
  NO_DEFAULT_PATH
)

# The oct-file: its one DEFMETHOD_DLD is __datamunge__, which the .m files of +datamunge call, so it
# must be named __datamunge__.oct and sit on the load path with the package.
add_library(datamunge_octave MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_octave
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "__datamunge__"
             SUFFIX ".oct"
)
# The generated sources include datamungeoctave/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_octave PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_include_directories(
  datamunge_octave SYSTEM PRIVATE "${OCTAVE_OCTINCLUDEDIR}" "${OCTAVE_INCLUDE_PARENT}"
)
target_link_libraries(
  datamunge_octave PRIVATE datamunge::datamunge "${OCTAVE_OCTINTERP_LIBRARY}"
                           "${OCTAVE_OCTAVE_LIBRARY}"
)

# <prefix>/share/octave/site/m: __datamunge__.oct and +datamunge/. `addpath` that directory, then
# datamunge.DataFrame (...), datamunge.seql (...), help datamunge.DataFrame.filter.
set(DATAMUNGE_OCTAVE_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/octave/site/m")
install(TARGETS datamunge_octave LIBRARY DESTINATION "${DATAMUNGE_OCTAVE_INSTALL_DIR}")
install(DIRECTORY "${DATAMUNGE_OCTAVE_PACKAGE}" DESTINATION "${DATAMUNGE_OCTAVE_INSTALL_DIR}")
