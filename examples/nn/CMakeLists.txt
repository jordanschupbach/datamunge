# datamunge::nn ports of the tflow example catalogue (~/projects/tflow_examples).
#
# The directory layout mirrors tflow's src/: every .cpp under a numbered section directory is one
# executable, named after the file, written to <build>/bin/<name>. Build and run with `just
# build-nn-examples` and `just run-nn <name>` (or `just run-all-nn`).

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

option(DATAMUNGE_NN_EXAMPLES_ALE "Enable the optional ALE emulator example" OFF)
if(DATAMUNGE_NN_EXAMPLES_ALE)
  find_package(ale CONFIG REQUIRED)
endif()

file(GLOB_RECURSE NN_EXAMPLES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[0-9]*/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/source/nn/[12]_*/*.cpp
)
set(NN_MIGRATED_TARGETS "")
foreach(source ${NN_EXAMPLES})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  if(source MATCHES "/cpp/source/nn/")
    list(APPEND NN_MIGRATED_TARGETS ${name})
  endif()
  set_target_properties(${name} PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  target_include_directories(
    ${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/source/nn
  )
  target_link_libraries(${name} PRIVATE ${DATAMUNGE_TARGET})
  file(STRINGS "${source}" ale_headers REGEX "#include.*common/rl/ale_env[.]hpp")
  if(ale_headers)
    if(DATAMUNGE_NN_EXAMPLES_ALE)
      target_link_libraries(${name} PRIVATE ale::ale-lib)
      get_target_property(ALE_INCLUDE_DIRS ale::ale INTERFACE_INCLUDE_DIRECTORIES)
      foreach(ale_include ${ALE_INCLUDE_DIRS})
        get_filename_component(ale_include_root "${ale_include}" DIRECTORY)
        target_include_directories(${name} PRIVATE "${ale_include_root}")
      endforeach()
    else()
      target_compile_definitions(${name} PRIVATE DATAMUNGE_EXAMPLES_DISABLE_ALE=1)
    endif()
  endif()
  target_compile_options(${name} PRIVATE -Wall)
endforeach()

add_custom_target(nn_sections_1_2 DEPENDS ${NN_MIGRATED_TARGETS})
