cmake_minimum_required(VERSION 3.14...3.22)

project(DatamungeExamples LANGUAGES C CXX)

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
  # find_package(datamunge REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(datamunge REQUIRED IMPORTED_TARGET datamunge)
  # if(NOT PkgConfig::datamunge) message(FATAL_ERROR "Could not find installed datamunge package via
  # pkg-config") endif()
else()
  CPMAddPackage(NAME datamunge SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
endif()

# CPMAddPackage(NAME datamunge SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOG_LEVEL_BUILD=LOG_SEV_NOTICE -Wall")

include(cmake/lib_sources.cmake)
foreach(ENTRY ${EXAMPLES_})
  get_filename_component(EXAMPLE_NAME ${ENTRY} NAME_WE)
  add_executable(${EXAMPLE_NAME} ${ENTRY})
  set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 17)
  set_property(TARGET ${EXAMPLE_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  if(EXAMPLE_NAME STREQUAL "playground" OR EXAMPLE_NAME STREQUAL "courier_offscreen")
    # The interactive API sketch uses the standard C++23 <print> header.
    set_property(TARGET ${EXAMPLE_NAME} PROPERTY CXX_STANDARD 23)
  endif()
  # target_link_libraries(${EXAMPLE_NAME} datamunge::datamunge)
  if(BUILD_W_INSTALLED)
    target_link_libraries(${EXAMPLE_NAME} PkgConfig::datamunge)
  else()
    target_link_libraries(${EXAMPLE_NAME} datamunge::datamunge)
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE DEBUG_BUILD)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES COMPILE_FLAGS "-O1")
  endif()
endforeach()

# Migrated NN sections retain their numbered layout and original target names.
option(DATAMUNGE_BUILD_NN_EXAMPLES "Build CNN and autoencoder examples" ON)
if(DATAMUNGE_BUILD_NN_EXAMPLES)
  file(GLOB_RECURSE NN_SECTION_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/source/nn/[12]_*/*.cpp"
  )
  foreach(source ${NN_SECTION_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
    target_include_directories(
      ${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source/nn ${CMAKE_CURRENT_SOURCE_DIR}/../nn
    )
    if(BUILD_W_INSTALLED)
      target_link_libraries(${name} PRIVATE PkgConfig::datamunge)
    else()
      target_link_libraries(${name} PRIVATE datamunge::datamunge)
    endif()
  endforeach()
endif()
