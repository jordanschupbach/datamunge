cmake_minimum_required(VERSION 3.16)

project(
  datamungetcl
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-tcl, scripts/prebuild_bindings.py tcl):
# src/datamunge_tcl_wrap*.cpp (the stubs-enabled extension, Datamunge_Init) and the package files
# pkg/datamunge/{datamunge.tcl,pkgIndex.tcl} beside this file. datamunge.tcl loads the extension from
# its own directory and carries the documentation, generated from rosetta/bindings.hpp and
# rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DATAMUNGE_TCL_PACKAGE_SRC "${CMAKE_CURRENT_LIST_DIR}/pkg/datamunge")
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_tcl_wrap*.cpp")
if(NOT GENERATED_SOURCES OR NOT EXISTS "${DATAMUNGE_TCL_PACKAGE_SRC}/datamunge.tcl")
  message(
    FATAL_ERROR "No generated Tcl binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-tcl`"
  )
endif()

# Tcl: version, shared-library suffix and install prefix from tclsh (as Rosetta's
# cmake/TclExample.cmake does), then tcl.h and the stub library under that prefix. The extension
# links only the stub library, so it loads into any Tcl 8.6 or 9 interpreter.
find_program(DATAMUNGE_TCLSH_EXECUTABLE NAMES tclsh9.0 tclsh tclsh8.6 REQUIRED)
set(tcl_config_script "${CMAKE_CURRENT_BINARY_DIR}/tcl_config.tcl")
file(
  WRITE "${tcl_config_script}"
  "puts [info patchlevel]\nputs [info sharedlibextension]\nputs [file dirname [file dirname [file normalize [info nameofexecutable]]]]\n"
)
execute_process(
  COMMAND "${DATAMUNGE_TCLSH_EXECUTABLE}"
  INPUT_FILE "${tcl_config_script}"
  OUTPUT_VARIABLE tcl_config
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" tcl_config "${tcl_config}")
list(GET tcl_config 0 TCL_PATCHLEVEL)
list(GET tcl_config 1 TCL_SHLIB_EXTENSION)
list(GET tcl_config 2 TCL_PREFIX)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" TCL_MAJOR_MINOR "${TCL_PATCHLEVEL}")
find_path(
  DATAMUNGE_TCL_INCLUDE_DIR tcl.h
  HINTS "${TCL_PREFIX}/include"
  PATH_SUFFIXES "tcl${TCL_MAJOR_MINOR}" tcl9 tcl
  REQUIRED NO_DEFAULT_PATH
)
find_library(
  DATAMUNGE_TCL_STUB_LIBRARY
  NAMES tclstub "tclstub${TCL_MAJOR_MINOR}"
  HINTS "${TCL_PREFIX}/lib"
  REQUIRED NO_DEFAULT_PATH
)
message(STATUS "Tcl ${TCL_PATCHLEVEL}: ${DATAMUNGE_TCL_INCLUDE_DIR}, ${DATAMUNGE_TCL_STUB_LIBRARY}")

get_target_property(
  DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES
)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(
    DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH
  )
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The package directory datamunge/ in the build tree: the extension datamunge<ext> beside copies of
# datamunge.tcl and pkgIndex.tcl, so TCLLIBPATH=<build dir> works without installing.
set(DATAMUNGE_TCL_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/datamunge")
add_library(datamunge_tcl MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge_tcl
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge"
             SUFFIX "${TCL_SHLIB_EXTENSION}"
             LIBRARY_OUTPUT_DIRECTORY "${DATAMUNGE_TCL_PACKAGE_DIR}"
)
target_compile_definitions(datamunge_tcl PRIVATE USE_TCL_STUBS)
# The generated sources include datamungetcl/rosetta/bindings.hpp relative to src/.
target_include_directories(
  datamunge_tcl PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}"
)
target_include_directories(datamunge_tcl SYSTEM PRIVATE "${DATAMUNGE_TCL_INCLUDE_DIR}")
target_link_libraries(datamunge_tcl PRIVATE datamunge::datamunge "${DATAMUNGE_TCL_STUB_LIBRARY}")
foreach(package_file datamunge.tcl pkgIndex.tcl)
  configure_file(
    "${DATAMUNGE_TCL_PACKAGE_SRC}/${package_file}" "${DATAMUNGE_TCL_PACKAGE_DIR}/${package_file}"
    COPYONLY
  )
endforeach()

# <prefix>/lib/datamunge: pkgIndex.tcl, datamunge.tcl and datamunge<ext>. `package require datamunge`
# with TCLLIBPATH=<prefix>/lib (the dev shells set it) or <prefix>/lib on auto_path.
install(TARGETS datamunge_tcl LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/datamunge")
install(FILES "${DATAMUNGE_TCL_PACKAGE_SRC}/datamunge.tcl" "${DATAMUNGE_TCL_PACKAGE_SRC}/pkgIndex.tcl"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/datamunge"
)
