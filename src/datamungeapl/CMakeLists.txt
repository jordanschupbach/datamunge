cmake_minimum_required(VERSION 3.16)

project(
  datamungeapl
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)
find_package(Threads REQUIRED)

# The sources come from Rosetta (just prebuild-apl, scripts/prebuild_bindings.py apl):
# src/datamunge_apl_wrap*.cpp (one GNU APL native function, the MODULE library libdatamunge_gnu_apl.so)
# and the GNU APL script apl/datamunge.apl beside this file, which ⎕FX's that library and defines the
# dm∆ functions, documented with ⍝ comments, generated from rosetta/bindings.hpp and
# rosetta/policy.json. The script loads ./libdatamunge_gnu_apl.so, or the file named by the
# environment variable DATAMUNGE_APL_LIBRARY (the dev shells and the justfile set it).
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_APL_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/apl/datamunge.apl")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_apl_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${DATAMUNGE_APL_SCRIPT}")
  message(
    FATAL_ERROR "No generated APL binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-apl`"
  )
endif()

# Native functions compile against GNU APL's own headers: a GNU APL 1.9 source tree configured like
# the interpreter (config.h, src/Native_interface.hh). The dev shells and nix/datamungeapl.nix set
# ROSETTA_GNU_APL_SOURCE_DIR, as Rosetta's flake does.
set(ROSETTA_GNU_APL_SOURCE_DIR
    "$ENV{ROSETTA_GNU_APL_SOURCE_DIR}"
    CACHE PATH "Configured GNU APL source tree"
)
if(NOT EXISTS "${ROSETTA_GNU_APL_SOURCE_DIR}/config.h"
   OR NOT EXISTS "${ROSETTA_GNU_APL_SOURCE_DIR}/src/Native_interface.hh"
)
  message(
    FATAL_ERROR
      "ROSETTA_GNU_APL_SOURCE_DIR must be a configured GNU APL source tree (with config.h): use the apl dev shell"
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

# libdatamunge_gnu_apl.so (libdatamunge is the C++ library itself). GNU APL loads it with ⎕FX and
# supplies its own symbols, so it links only datamunge.
add_library(datamunge_gnu_apl MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_gnu_apl
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX "lib"
             OUTPUT_NAME "datamunge_gnu_apl"
             SUFFIX ".so"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
)
target_compile_definitions(datamunge_gnu_apl PRIVATE HAVE_CONFIG_H)
# The generated sources include datamungeapl/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_gnu_apl PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_include_directories(
  datamunge_gnu_apl SYSTEM PRIVATE "${ROSETTA_GNU_APL_SOURCE_DIR}"
                                   "${ROSETTA_GNU_APL_SOURCE_DIR}/src"
)
target_link_libraries(datamunge_gnu_apl PRIVATE datamunge::datamunge Threads::Threads)
if(APPLE)
  target_link_options(datamunge_gnu_apl PRIVATE -undefined dynamic_lookup)
endif()

# <prefix>/lib/libdatamunge_gnu_apl.so and the script <prefix>/share/apl/datamunge.apl: run
# `DATAMUNGE_APL_LIBRARY=<prefix>/lib/libdatamunge_gnu_apl.so apl --script -f
# <prefix>/share/apl/datamunge.apl -f program.apl`.
install(TARGETS datamunge_gnu_apl LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${DATAMUNGE_APL_SCRIPT}" DESTINATION "${CMAKE_INSTALL_DATADIR}/apl")
