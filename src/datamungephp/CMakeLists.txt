cmake_minimum_required(VERSION 3.16)

project(
  datamungephp
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-php, scripts/prebuild_bindings.py php):
# src/datamunge_php_wrap*.cpp (the C ABI library) and php/datamunge.php beside this file (the
# namespace Datamunge over PHP FFI), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_php_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated PHP binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-php`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# php/datamunge.php loads this library with FFI::cdef: from DATAMUNGE_PHP_LIBRARY (the dev shells
# set it), beside the PHP file, or from lib/ or ../lib.
add_library(datamunge_php SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_php
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_php"
)

# The generated sources include datamungephp/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_php PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_php PRIVATE datamunge::datamunge)

# <prefix>/lib/libdatamunge_php.so and <prefix>/share/php/datamunge/datamunge.php (which finds
# the library in ../../../lib only through DATAMUNGE_PHP_LIBRARY or the loader path).
install(TARGETS datamunge_php LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/php/datamunge.php" DESTINATION "${CMAKE_INSTALL_DATADIR}/php/datamunge")
