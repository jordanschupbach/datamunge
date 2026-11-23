cmake_minimum_required(VERSION 3.16)

project(
  datamungepascal
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-pascal, scripts/prebuild_bindings.py pascal):
# src/datamunge_pascal_wrap*.cpp (the C ABI library) and the Free Pascal unit pascal/datamunge.pas
# beside this file (XML doc comments), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_pascal_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Pascal binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-pascal`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The unit links this library ({$linklib datamunge_pascal}); programs add -Fl<prefix>/lib.
add_library(datamunge_pascal SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_pascal
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_pascal"
)

# The generated sources include datamungepascal/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_pascal PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_pascal PRIVATE datamunge::datamunge)

find_program(DATAMUNGE_FPC fpc HINTS "$ENV{DATAMUNGE_FPC_ROOT}/bin" REQUIRED)

# The compiled unit (datamunge.ppu, datamunge.o): programs compile with
# `fpc -Fu<prefix>/lib/fpc/datamunge -Fl<prefix>/lib -k-rpath -k<prefix>/lib program.pas`.
set(unit_out "${CMAKE_CURRENT_BINARY_DIR}/unit")
add_custom_command(
  OUTPUT "${unit_out}/datamunge.ppu" "${unit_out}/datamunge.o"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${unit_out}"
  COMMAND "${DATAMUNGE_FPC}" -O2 -vw "-FU${unit_out}" "${CMAKE_CURRENT_LIST_DIR}/pascal/datamunge.pas"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/pascal/datamunge.pas"
  COMMENT "Compiling the Free Pascal unit datamunge"
  VERBATIM)
add_custom_target(datamunge_pascal_unit ALL DEPENDS "${unit_out}/datamunge.ppu")

install(TARGETS datamunge_pascal LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${unit_out}/datamunge.ppu" "${unit_out}/datamunge.o" "${CMAKE_CURRENT_LIST_DIR}/pascal/datamunge.pas"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/fpc/datamunge")
