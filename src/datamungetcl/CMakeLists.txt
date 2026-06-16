cmake_minimum_required(VERSION 3.16)

project(
  datamungetcl
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(TCL REQUIRED)
find_package(datamunge CONFIG REQUIRED)

find_program(SWIG_EXECUTABLE swig REQUIRED)

set(SWIG_INTERFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/swig/datamungetcl.i")
set(GENERATED_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/datamunge_tcl_wrap.cxx")

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

set(DATAMUNGE_SWIG_INCLUDE_FLAGS "")
foreach(incdir IN LISTS DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  list(APPEND DATAMUNGE_SWIG_INCLUDE_FLAGS "-I${incdir}")
endforeach()

add_custom_command(
  OUTPUT "${GENERATED_WRAPPER}"
  COMMAND "${SWIG_EXECUTABLE}" -tcl8 -c++ ${DATAMUNGE_SWIG_INCLUDE_FLAGS} -o "${GENERATED_WRAPPER}"
          "${SWIG_INTERFACE_FILE}"
  DEPENDS "${SWIG_INTERFACE_FILE}"
  VERBATIM
)

add_library(Datamunge MODULE "${GENERATED_WRAPPER}")
set_target_properties(
  Datamunge
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "Datamunge"
)

target_include_directories(Datamunge PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} ${TCL_INCLUDE_PATH})

target_link_libraries(Datamunge PRIVATE datamunge::datamunge ${TCL_LIBRARY})
