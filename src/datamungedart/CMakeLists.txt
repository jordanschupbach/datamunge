cmake_minimum_required(VERSION 3.16)

project(
  datamungedart
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-dart, scripts/prebuild_bindings.py dart):
# src/datamunge_dart_wrap*.cpp (the C ABI library) and the Dart package datamunge/ beside this file
# (dartdoc comments), generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_dart_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Dart binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-dart`")
endif()

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# The package loads this library through dart:ffi: DATAMUNGE_DART_LIBRARY names it.
add_library(datamunge_dart SHARED ${GENERATED_SOURCES})
set_target_properties(
  datamunge_dart
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME "datamunge_dart"
)

# The generated sources include datamungedart/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge_dart PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge_dart PRIVATE datamunge::datamunge)

install(TARGETS datamunge_dart LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
# The Dart package, for a pubspec dependency `datamunge: {path: <prefix>/share/dart/datamunge}`.
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/datamunge/" DESTINATION share/dart/datamunge)
