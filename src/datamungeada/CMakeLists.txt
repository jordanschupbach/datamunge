cmake_minimum_required(VERSION 3.16)

project(
  datamungeada
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-ada, scripts/prebuild_bindings.py ada):
# src/datamunge_ada_wrap*.cpp (the C ABI library) and the Ada package ada/datamunge.ads, .adb and .gpr
# beside this file (GNATdoc comments), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_ada_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Ada binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-ada`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The package links this library (pragma Linker_Options "-ldatamunge_ada"); programs add -L<prefix>/lib.
add_library(datamunge_ada SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_ada
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_ada"
)

# The generated sources include datamungeada/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_ada PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_ada PRIVATE datamunge::datamunge)

find_program(DATAMUNGE_GNATMAKE gnatmake HINTS "$ENV{DATAMUNGE_GNAT_ROOT}/bin" REQUIRED)

# The compiled package (datamunge.ali, datamunge.o; about a minute): programs compile with
# `gnatmake -gnat2022 -aI<prefix>/lib/ada/datamunge -aO<prefix>/lib/ada/datamunge program.adb
#  -largs -L<prefix>/lib -Wl,-rpath,<prefix>/lib`. The .ali is read-only, so gnatmake uses it as it is.
set(package_out "${CMAKE_CURRENT_BINARY_DIR}/package")
set(package_src "${CMAKE_CURRENT_LIST_DIR}/ada")
add_custom_command(
  OUTPUT "${package_out}/datamunge.ali" "${package_out}/datamunge.o"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${package_out}"
  COMMAND "${CMAKE_COMMAND}" -E chdir "${package_out}" "${DATAMUNGE_GNATMAKE}" -q -c -gnat2022 -O1 "${package_src}/datamunge.adb"
  DEPENDS "${package_src}/datamunge.ads" "${package_src}/datamunge.adb"
  COMMENT "Compiling the Ada package Datamunge"
  VERBATIM)
add_custom_target(datamunge_ada_package ALL DEPENDS "${package_out}/datamunge.ali")

install(TARGETS datamunge_ada LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${package_src}/datamunge.ads" "${package_src}/datamunge.adb" "${package_src}/datamunge.gpr" "${package_out}/datamunge.o"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/ada/datamunge")
install(FILES "${package_out}/datamunge.ali" DESTINATION "${CMAKE_INSTALL_LIBDIR}/ada/datamunge"
        PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
