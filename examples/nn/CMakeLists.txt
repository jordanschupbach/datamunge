# datamunge::nn ports of the tflow example catalogue (~/projects/tflow_examples).
#
# The directory layout mirrors tflow's src/: every .cpp under a numbered
# section directory is one executable, named after the file, written to
# <build>/bin/<name>. Build and run with `just build-nn-examples` and
# `just run-nn <name>` (or `just run-all-nn`).

cmake_minimum_required(VERSION 3.14...3.22)

project(DatamungeNnExamples LANGUAGES C CXX)

include(../../cmake/CPM.cmake)

option(BUILD_W_INSTALLED "Build against the installed datamunge (pkg-config)" OFF)
if(BUILD_W_INSTALLED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(datamunge REQUIRED IMPORTED_TARGET datamunge)
  set(DATAMUNGE_TARGET PkgConfig::datamunge)
else()
  CPMAddPackage(NAME datamunge SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
  set(DATAMUNGE_TARGET datamunge::datamunge)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB_RECURSE NN_EXAMPLES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[0-9]*/*.cpp)
foreach(source ${NN_EXAMPLES})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  set_target_properties(${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE ${DATAMUNGE_TARGET})
  target_compile_options(${name} PRIVATE -Wall)
endforeach()
