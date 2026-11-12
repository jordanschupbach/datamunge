cmake_minimum_required(VERSION 3.16)

project(
  datamungecobol
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-cobol, scripts/prebuild_bindings.py cobol):
# src/datamunge_cobol_wrap*.cpp (the native library) and copybooks/datamunge{,-callbacks}.cpy,
# generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_cobol_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated COBOL binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-cobol`")
endif()
set(COPYBOOKS "${CMAKE_CURRENT_LIST_DIR}/copybooks/datamunge.cpy" "${CMAKE_CURRENT_LIST_DIR}/copybooks/datamunge-callbacks.cpy")

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The stubs COBOL programs CALL (cobc -fstatic-call ... -ldatamungecobol): a shared library
# that carries its own dependencies, so programs link it alone.
add_library(datamungecobol SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamungecobol
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamungecobol"
)

# The generated sources include datamungecobol/rosetta/bindings.hpp relative to src/.
target_include_directories(datamungecobol PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamungecobol PRIVATE datamunge::datamunge)

# Programs COPY "datamunge.cpy" with <prefix>/share/cobol/copy on COBCPY (or cobc -I).
install(TARGETS datamungecobol LIBRARY DESTINATION lib)
install(FILES ${COPYBOOKS} DESTINATION "${CMAKE_INSTALL_DATADIR}/cobol/copy")
