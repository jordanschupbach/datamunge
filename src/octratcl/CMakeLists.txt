cmake_minimum_required(VERSION 3.16)

project(
  octratcl
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(TCL REQUIRED)
find_package(octra CONFIG REQUIRED)

find_program(SWIG_EXECUTABLE swig REQUIRED)

set(SWIG_INTERFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/swig/octratcl.i")
set(GENERATED_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/octra_tcl_wrap.cxx")

get_target_property(OCTRA_INSTALLED_INCLUDE_DIRS octra::octra INTERFACE_INCLUDE_DIRECTORIES)
if(NOT OCTRA_INSTALLED_INCLUDE_DIRS)
  get_filename_component(OCTRA_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(OCTRA_INSTALLED_INCLUDE_DIRS "${OCTRA_REPO_INCLUDE_DIR}")
endif()

set(OCTRA_SWIG_INCLUDE_FLAGS "")
foreach(incdir IN LISTS OCTRA_INSTALLED_INCLUDE_DIRS)
  list(APPEND OCTRA_SWIG_INCLUDE_FLAGS "-I${incdir}")
endforeach()

add_custom_command(
  OUTPUT "${GENERATED_WRAPPER}"
  COMMAND "${SWIG_EXECUTABLE}" -tcl8 -c++ ${OCTRA_SWIG_INCLUDE_FLAGS} -o "${GENERATED_WRAPPER}"
          "${SWIG_INTERFACE_FILE}"
  DEPENDS "${SWIG_INTERFACE_FILE}"
  VERBATIM
)

add_library(Octra MODULE "${GENERATED_WRAPPER}")
set_target_properties(
  Octra
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "Octra"
)

target_include_directories(Octra PRIVATE ${OCTRA_INSTALLED_INCLUDE_DIRS} ${TCL_INCLUDE_PATH})

target_link_libraries(Octra PRIVATE octra::octra ${TCL_LIBRARY})
