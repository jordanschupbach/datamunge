cmake_minimum_required(VERSION 3.16)

project(
  datamungelisp
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(datamunge CONFIG REQUIRED)

# The sources come from Rosetta (just prebuild-lisp, scripts/prebuild_bindings.py lisp):
# src/datamunge_lisp_wrap*.cpp (the native library) and datamunge.{lisp,asd} (the ASDF system),
# generated from rosetta/bindings.hpp and rosetta/policy.json.
get_filename_component(DATAMUNGE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${DATAMUNGE_SRC_DIR}/datamunge_lisp_wrap*.cpp")
if(NOT GENERATED_SOURCES)
  message(FATAL_ERROR "No generated Common Lisp binding sources in ${DATAMUNGE_SRC_DIR}: run `just prebuild-lisp`")
endif()
set(LISP_FILES "${CMAKE_CURRENT_LIST_DIR}/datamunge.lisp" "${CMAKE_CURRENT_LIST_DIR}/datamunge.asd")

get_target_property(DATAMUNGE_INSTALLED_INCLUDE_DIRS datamunge::datamunge INTERFACE_INCLUDE_DIRECTORIES)
if(NOT DATAMUNGE_INSTALLED_INCLUDE_DIRS)
  get_filename_component(DATAMUNGE_REPO_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" REALPATH)
  set(DATAMUNGE_INSTALLED_INCLUDE_DIRS "${DATAMUNGE_REPO_INCLUDE_DIR}")
endif()

# A plain shared library (no Lisp headers): the .lisp file loads it through CFFI.
add_library(datamunge MODULE ${GENERATED_SOURCES})
set_target_properties(
  datamunge
  PROPERTIES CXX_STANDARD 23
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "datamunge"
)
if(APPLE)
  # CFFI looks for MODULE.dylib.
  set_target_properties(datamunge PROPERTIES SUFFIX ".dylib")
endif()

# The generated sources include datamungelisp/rosetta/bindings.hpp relative to src/.
target_include_directories(datamunge PRIVATE ${DATAMUNGE_INSTALLED_INCLUDE_DIRS} "${DATAMUNGE_SRC_DIR}")
target_link_libraries(datamunge PRIVATE datamunge::datamunge)

# The system directory holds the .asd, the .lisp and the library they load from beside them;
# put <prefix>/share/common-lisp/source/ on CL_SOURCE_REGISTRY to find it.
set(DATAMUNGE_LISP_DIR "${CMAKE_INSTALL_DATADIR}/common-lisp/source/datamunge")
install(TARGETS datamunge DESTINATION "${DATAMUNGE_LISP_DIR}")
install(FILES ${LISP_FILES} DESTINATION "${DATAMUNGE_LISP_DIR}")
