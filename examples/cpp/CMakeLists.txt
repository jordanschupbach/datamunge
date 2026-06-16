cmake_minimum_required(VERSION 3.14...3.22)

project(OctraExamples LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL ""
)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

include(../../cmake/tools.cmake)
include(../../cmake/CPM.cmake)

# find_package(doctest REQUIRED) BUG: environment issue? cpmaddpackage("gh:doctest/ldoctest@2.4.9")
# cpmaddpackage("gh:TheLartians/Format.cmake@1.7.3")

option(BUILD_W_INSTALLED "Build w/ the installed version (assumes pkg-config)" OFF)

message(STATUS "BUILD_W_INSTALLED: ${BUILD_W_INSTALLED}")

if(BUILD_W_INSTALLED)
  # find_package(octra REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(octra REQUIRED IMPORTED_TARGET octra)
  # if(NOT PkgConfig::octra) message(FATAL_ERROR "Could not find installed octra package via
  # pkg-config") endif()
else()
  CPMAddPackage(NAME octra SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
endif()

# CPMAddPackage(NAME octra SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOG_LEVEL_BUILD=LOG_SEV_NOTICE -Wall")

include(cmake/lib_sources.cmake)
foreach(ENTRY ${EXAMPLES_})
  get_filename_component(EXAMPLE_NAME ${ENTRY} NAME_WE)
  add_executable(${EXAMPLE_NAME} ${ENTRY})
  set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 17)
  set_property(TARGET ${EXAMPLE_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  # target_link_libraries(${EXAMPLE_NAME} octra::octra)
  if(BUILD_W_INSTALLED)
    target_link_libraries(${EXAMPLE_NAME} PkgConfig::octra)
  else()
    target_link_libraries(${EXAMPLE_NAME} octra::octra)
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE DEBUG_BUILD)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES COMPILE_FLAGS "-O1")
  endif()
endforeach()
